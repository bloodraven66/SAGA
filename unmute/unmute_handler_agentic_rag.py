"""Agentic tool-use RAG handler.

State machine (see CLAUDE.md for the full design writeup):

  TRANSCRIPT_FINAL
    -> QUERY_DECOMPOSE(i): LLM call, sees full persistent chat_history + this
       loop's own ephemeral tool-call/result scratch state. Either returns
       tool_calls, or (if empty) we're done deciding and move to the answer.
    -> RETRIEVE_AND_NOTE(i): tool_calls execute against the RAG server AND a
       spoken note is generated, in parallel. The note is sent to the TTS
       immediately (continuous feeding -- see below).
    -> loop, up to max_turns
    -> ANSWER: generate the final answer as a NORMAL completion with a
       system-prompt override that tells the model it already said the note(s)
       aloud and to continue naturally (this mirrors the reference pipeline
       Speech_Research_RAG/rag/pipeline.py, which works reliably). On turns that
       already have results the answer is streamed straight to the TTS as it
       generates. Then EOS.
    -> TURN_COMPLETE: chat_history ends up with the user question + everything
       actually voiced (notes + answer), synced for free by the base handler's
       _tts_loop word-by-word append.

TTS feeding (continuous -- the fix for the 20s-gap deadlock):
  A single continuous TTS connection carries the whole turn so prosody flows
  naturally between notes and the answer. Text is fed to the TTS *continuously*
  as it becomes ready (each note the moment it's generated, then the answer
  word-by-word), with a single EOS at the very end -- exactly like the base
  UnmuteHandler. The previous playback-gated "drop unheard notes" design
  deadlocked against the Kyutai delayed-streams TTS (which won't emit a chunk's
  tail until it gets more text / an EOS flush), producing a ~20s silent gap on
  every item boundary. See docs/agentic_rag_debug_log.md items 8-11 and CLAUDE.md.

Answer length: NOT hard-capped/truncated -- this is a live conversational
system, so the model speaks its full reply. Brevity is nudged via the system
prompt only (see agentic_rag_tools.py). Truncating the stream at a sentence
budget cut answers off mid-thought (e.g. right after a "Yes, exactly!" opener),
which is the opposite of good conversation -- removed.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from functools import partial
from logging import getLogger
from typing import Any

import numpy as np
import websockets

import unmute.openai_realtime_api_events as ora
from unmute.agentic_rag_tools import (
    AgenticRagClient,
    FORCE_RETRIEVAL_DIRECTIVE,
    SPECULATIVE_NOTE_PROMPT,
    SPOKEN_NOTE_PROMPT,
    SYSTEM_PROMPT,
    _continuation_prompt,
    _speculative_note_user_msg,
    _spoken_note_user_msg,
)
from unmute.endpointer import Endpointer
from unmute.kyutai_constants import (
    AGENTIC_LLM_MODEL,
    AGENTIC_LLM_SERVER,
    SAMPLE_RATE,
    SPEECH_RAG_API_KEY,
    SPEECH_RAG_SERVER,
)
from unmute.llm.llm_utils import get_openai_client
from unmute.llm.system_prompt import AgenticRagInstructions
from unmute.quest_manager import Quest
from unmute.service_discovery import find_instance
from unmute.tts.text_to_speech import (
    TextToSpeech,
    TTSAudioMessage,
    TTSClientEosMessage,
)
from unmute.unmute_handler import UnmuteHandler

logger = getLogger(__name__)

MAX_LOOP_TURNS = 4

# ── Anticipation / speculative-note constants ────────────────────────────────
# The endpoint anticipator (port 8093) streams user_end_probability at 12.5Hz.
# When it crosses the threshold while the user is still speaking, we speculatively
# synthesize ONE short spoken note off the partial transcript and buffer its audio,
# so the instant real VAD fires we can play it with ~0 latency (sub-600ms filler).
# See docs/agentic_speculative_note_design.md. Deliberately a COMPOSED add-on: the
# handler still subclasses UnmuteHandler (not UnmuteHandlerSpeculative), so if
# anticipation never fires the behavior is byte-for-byte the current handler.
ANTICIPATE_THRESHOLD: float = 0.5
ANTICIPATE_WINDOW_SEC: float = 0.960
ANTICIPATE_COOLDOWN_SEC: float = ANTICIPATE_WINDOW_SEC
ANTICIPATOR_FRAME_SAMPLES: int = 960
ANTICIPATOR_RECONNECT_BASE_SEC: float = 0.5
ANTICIPATOR_RECONNECT_MAX_SEC: float = 4.0
SPEC_NOTE_MAX_TOKENS: int = 32  # keep the filler short so its audio buffers fast

# Per-request timeouts for the vLLM calls. The OpenAI SDK defaults to a 600s
# request timeout, so a wedged stream (observed live: a tool_choice="required"
# decompose that accepted the request then never streamed a token -- a known
# guided-decoding stall) hangs the whole turn effectively forever. These bound
# the inter-chunk (httpx read) interval so a stall raises within ~N seconds and
# the turn degrades gracefully instead of getting stuck. Because it's an
# inter-chunk bound, a steadily-progressing stream never trips it -- it only fires
# if the server goes silent this long. Set above the normal ~2s / slow ~4.5s
# first-token latency but low enough a genuine stall doesn't feel like forever.
DECOMPOSE_TIMEOUT_SEC: float = 12.0
ANSWER_TIMEOUT_SEC: float = 20.0
NOTE_TIMEOUT_SEC: float = 10.0

# Fillers are capped STRUCTURALLY, not by playback gating. The agentic loop speaks
# at most ONE bridge note (turn 1); later search turns stay silent. This avoids the
# "let me also check... let me also check..." pile-up without tracking playback
# progress or blocking the TTS feed -- gating the TTS against the delayed-streams
# model is exactly what caused the 20s-gap/stall saga (see docs/agentic_rag_debug_log.md
# item 11). The speculative opener + one short bridge already buffer ~4-5s of speech,
# which comfortably covers decompose+retrieval+answer generation for the common
# 1-2 search-round turn; the answer is then fed continuously behind it.
MAX_LOOP_NOTES: int = 1


@dataclass
class SpeculativeNote:
    """A single pre-VAD speculative filler: its text and buffered TTS audio.

    Deliberately much slimmer than UnmuteHandlerSpeculative.SpeculativeState --
    we never continue this as an utterance (no continue_final_message / AR
    handoff). It's a self-contained filler; the grounded answer follows on the
    main TTS behind it.
    """

    trigger_transcript: str
    text_tokens: list[str] = field(default_factory=list)
    audio_chunks: list[np.ndarray] = field(default_factory=list)
    tts_connection: TextToSpeech | None = None
    tts_receive_task: asyncio.Task | None = None
    task: asyncio.Task | None = None
    committed: bool = False
    discarded: bool = False

    @property
    def text(self) -> str:
        return "".join(self.text_tokens)

# Grace period after a turn starts (conversation_state flips to "bot_speaking")
# during which STT-triggered barge-in is ignored. Needed because the STT
# server's own flush can emit a trailing word of the SAME utterance that
# triggered this turn slightly after conversation_state already flipped --
# indistinguishable from a genuine new utterance to _should_interrupt_on_stt_message.
# The base handler rarely hits this since it starts streaming audio almost
# immediately; this handler's agentic loop can take 1-5+ seconds before
# anything is actually said, making the race far more likely. Confirmed via
# docs/agentic_rag_debug_log.md item 4 (observed race window: ~80ms).
AGENTIC_INTERRUPT_GRACE_SEC = 0.6

# Keep the LLM prompt well under Qwen's 32768-token context. Long conversations
# plus big RAG payloads (a broad search_author_topics can return 40+ papers with
# abstracts) would otherwise overflow and 400 the whole turn. Defense in depth:
# cap persistent history, compact each tool result, and drop the oldest in-loop
# tool results if the assembled prompt still estimates over budget.
MAX_HISTORY_MSGS = 12            # persistent chat messages kept (system prompt always kept on top)
MAX_INPUT_TOKENS = 28000         # trim ephemeral tool results above this estimate (~4.7k headroom)
MAX_RESULTS_IN_CONTEXT = 6       # cap results per tool call fed to the LLM
ABSTRACT_CHARS_IN_CONTEXT = 280  # truncate each abstract fed to the LLM


_CONVERSATIONAL_MARKERS = (
    "can you hear", "can you understand", "are you there", "are you listening",
    "are you working", "are you a bot", "are you a human", "are you real",
    "is the transcription", "is transcription", "is this working", "is it working",
    "is this on", "is the mic", "is the audio", "is my mic", "do you hear",
    "how are you", "how's it going", "how is it going", "what's up", "whats up",
    "who are you", "what are you", "what can you do", "say something",
    "good morning", "good afternoon", "good evening", "nice to meet",
    "just testing", "testing testing", "check check", "hows the weather",
)
_GREETING_WORDS = frozenset(
    "hi hello hey yo hiya heya thanks thank thankyou ty ok okay cool nice bye "
    "goodbye sup great awesome perfect gotcha test testing hmm yeah yep".split()
)


def _is_conversational_query(query: str) -> bool:
    """Best-effort check for a purely conversational / meta / system-check turn
    (greeting, small talk, "can you hear me?", "is the transcription working?").

    Such turns must NOT be force-retrieved -- forcing a tool call turns "can you
    hear me?" into an irrelevant paper dump. Tuned to catch conversational turns
    broadly (a false positive just means tool_choice="auto" instead of forced, and
    the model still retrieves for anything genuinely research-y via the system
    prompt); a substantive research question should never match here."""
    q = (query or "").strip().lower()
    if not q:
        return False
    words = q.rstrip("?.!").split()
    if len(words) > 9:
        return False  # long enough to be a real question; don't second-guess it
    if any(m in q for m in _CONVERSATIONAL_MARKERS):
        return True
    # A short utterance made up entirely of greeting/ack words (e.g. "hey there",
    # "ok thanks", "cool cool").
    stripped = [w.strip(",.!?") for w in words]
    if 1 <= len(stripped) <= 4 and all(
        w in _GREETING_WORDS or w in ("there", "you", "so", "a", "the") for w in stripped
    ):
        return True
    return False


def _compact_tool_result(result: Any) -> Any:
    """Shrink a RAG tool result before it goes into the LLM prompt: cap the
    number of papers and truncate abstracts. Keeps the fields the model needs to
    ground an answer (title/authors/year/venue/citations) without dumping full
    abstracts for dozens of papers."""
    if not isinstance(result, dict):
        return result
    out = {k: v for k, v in result.items() if k != "results"}
    results = result.get("results", []) or []
    compact: list[Any] = []
    for r in results[:MAX_RESULTS_IN_CONTEXT]:
        if not isinstance(r, dict):
            compact.append(r)
            continue
        item: dict[str, Any] = {}
        for k in ("title", "session_name", "authors", "year", "venue", "citation_count"):
            if r.get(k) is not None:
                item[k] = r[k]
        ab = r.get("abstract")
        if ab:
            item["abstract"] = (
                ab[:ABSTRACT_CHARS_IN_CONTEXT] + ("..." if len(ab) > ABSTRACT_CHARS_IN_CONTEXT else "")
            )
        compact.append(item or r)
    out["results"] = compact
    if len(results) > MAX_RESULTS_IN_CONTEXT:
        out["results_truncated"] = f"{len(results)} found, showing top {MAX_RESULTS_IN_CONTEXT}"
    return out


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough (~4 chars/token) size of an OpenAI message list, plus the tool
    schema that's always sent. Only needs to be good enough to stay under the
    context limit with margin."""
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
        for tc in m.get("tool_calls") or []:
            total += len(json.dumps(tc))
    return total // 4 + 1300  # +~1.3k for the tool-definitions schema


_LEAKED_CONTROL_TOKEN_RE = re.compile(
    r"^\s*(SEARCH_COMPLETE|DONE|COMPLETE)\s*[:\-\n]*\s*", re.IGNORECASE
)
_TOOL_CALL_XML_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


# Emoji / pictographic / symbol ranges that must never reach the TTS -- they get
# read out literally ("smiling face") or mispronounced. Defensive backstop in
# case the model ignores the "no emoji" system-prompt instruction.
_EMOJI_RE = re.compile(
    "["
    "\U0001f000-\U0001faff"  # emoticons, pictographs, transport, symbols, supplemental
    "\U00002600-\U000027bf"  # misc symbols + dingbats
    "\U00002190-\U000021ff"  # arrows
    "\U00002b00-\U00002bff"  # misc symbols and arrows
    "\U0001f1e6-\U0001f1ff"  # regional indicator (flags)
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "]+",
    flags=re.UNICODE,
)
# zero-width space/joiner, directional marks, word joiner, BOM
_ZERO_WIDTH_RE = re.compile("[​-‏‪-‮⁠﻿]")


def _sanitize_for_tts(text: str) -> str:
    """Make text safe to speak: drop emojis/symbols/zero-width chars and turn
    em/en-dashes into commas.

    The delayed-streams TTS mishandles a standalone punctuation token, and any
    emoji/symbol is read out literally, so strip them here as a backstop
    regardless of what the model produced.
    """
    text = _EMOJI_RE.sub("", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = re.sub(r"\s*[—–]\s*", ", ", text)
    # collapse any whitespace runs left by removals
    return re.sub(r"[ \t]{2,}", " ", text)


def _strip_control_artifacts(text: str) -> str:
    """Strip stray control/status tokens the LLM sometimes leaks as a prefix
    (e.g. "SEARCH_COMPLETE\\n...") plus any <tool_call>...</tool_call> XML the
    tool-parser occasionally emits into the answer even with tool_choice=none."""
    text = _TOOL_CALL_XML_RE.sub("", text)
    return _LEAKED_CONTROL_TOKEN_RE.sub("", text).strip()


def _summarize_papers(results: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    """Compact, human-readable summary of retrieved items for the trace JSON --
    just enough to eyeball grounding (which papers came back), not the full
    payloads. Handles the varying result shapes across the RAG tools."""
    out: list[dict[str, Any]] = []
    for r in results[:limit]:
        if not isinstance(r, dict):
            continue
        out.append(
            {
                "title": r.get("title") or r.get("session_name") or "",
                "authors": r.get("authors", ""),
                "year": r.get("year"),
            }
        )
    return out


def _short_authors(authors: Any) -> str:
    if isinstance(authors, list):
        if not authors:
            return ""
        return authors[0] + (" et al." if len(authors) > 1 else "")
    return str(authors or "")


def _papers_for_ui(results: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    """Compact paper info for the live SAGA UI cards."""
    out: list[dict[str, Any]] = []
    for r in results[:limit]:
        if not isinstance(r, dict):
            continue
        out.append(
            {
                "title": r.get("title") or r.get("session_name") or "",
                "authors": _short_authors(r.get("authors", "")),
                "year": r.get("year"),
                "citations": r.get("citation_count"),
            }
        )
    return out


def _tool_query_str(args: dict[str, Any]) -> str:
    if args.get("author_name") and args.get("topic"):
        return f"{args['author_name']} · {args['topic']}"
    return (
        args.get("query")
        or args.get("topic")
        or args.get("author_name")
        or args.get("name")
        or ""
    )


def _tool_calls_to_openai_messages(
    tool_calls: list[dict[str, Any]], tool_results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    },
                }
                for tc in tool_calls
            ],
        }
    ]
    for tr in tool_results:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tr["id"],
                "content": json.dumps(_compact_tool_result(tr.get("result", {}))),
            }
        )
    return messages


class UnmuteHandlerAgenticRAG(UnmuteHandler):
    # Stricter STT-VAD barge-in than the base 0.4: the agentic turn is a longer,
    # information-dense answer the listener usually wants to hear out, so require
    # the VAD to be MORE confident the user is actually talking before cutting it
    # off -- fewer false interruptions from backchannels ("mm", "I see"), echo, or
    # STT hallucinations. Lower = stricter. (The STT-word interrupt path is
    # additionally gated by _should_interrupt_on_stt_message's grace period.)
    VAD_INTERRUPT_PAUSE_THRESHOLD: float = 0.2

    # Require a longer/more-confident pause before treating the user as finished,
    # so a short mid-sentence pause (thinking, drawing breath) is NOT mistaken for
    # end-of-turn and doesn't make the bot cut in prematurely. Higher than the base
    # 0.6. The speculative opener fires from pre-VAD anticipation, so this added
    # end-of-turn patience costs little perceived latency.
    VAD_PAUSE_THRESHOLD: float = 0.8

    def __init__(
        self,
        rag_url: str = SPEECH_RAG_SERVER,
        rag_api_key: str | None = SPEECH_RAG_API_KEY,
        rag_top_k: int = 3,
        max_turns: int = MAX_LOOP_TURNS,
    ) -> None:
        super().__init__()
        self.chatbot.set_instructions(AgenticRagInstructions())

        self._rag_client = AgenticRagClient(
            rag_url=rag_url, api_key=rag_api_key, top_k=rag_top_k
        )
        self._agentic_llm_client = get_openai_client(server_url=AGENTIC_LLM_SERVER)
        self._max_turns = max_turns

        self._agentic_traces: list[dict[str, Any]] = []
        self._current_turn_started_at: float | None = None

        # ── Speculative-note (anticipation) state ──
        self._spec_note: SpeculativeNote | None = None
        # Most recent fully-buffered note that got superseded; committed at VAD if
        # the current speculation isn't ready in time (generic filler is reusable).
        self._fallback_note: SpeculativeNote | None = None
        self._spec_lock = asyncio.Lock()
        self._last_anticipation_time: float = 0.0
        # Set at VAD when a buffered note is committed; consumed post-flush by
        # _generate_response to coordinate the placeholder + finalized transcript.
        self._early_committed_note: SpeculativeNote | None = None
        self._note_flush_event: asyncio.Event | None = None

    def get_agentic_traces(self) -> list[dict[str, Any]]:
        return [dict(t) for t in self._agentic_traces]

    async def _should_interrupt_on_stt_message(self, data: Any) -> bool:
        if self.chatbot.conversation_state() != "bot_speaking":
            return False
        if self._current_turn_started_at is not None:
            elapsed = self.audio_received_sec() - self._current_turn_started_at
            if elapsed < AGENTIC_INTERRUPT_GRACE_SEC:
                logger.info(
                    "[AgenticRAG] Ignoring STT-based interruption %.2fs into turn "
                    "(grace period %.2fs): %r",
                    elapsed,
                    AGENTIC_INTERRUPT_GRACE_SEC,
                    data.text,
                )
                return False
        return True

    async def start_up(self) -> None:
        await super().start_up()  # STT
        try:
            ok = await self._rag_client.health()
            logger.info("[AgenticRAG] RAG server health: %s", ok)
        except Exception as exc:
            logger.warning("[AgenticRAG] RAG server health check failed: %r", exc)
        try:
            await self.start_up_endpointer()
        except Exception as exc:
            logger.warning(
                "[AgenticRAG] Endpointer failed to start (%r). "
                "Continuing WITHOUT speculative fillers.",
                exc,
            )

    # ------------------------------------------------------------------
    # Anticipation: endpointer wiring (additive; composed onto UnmuteHandler)
    # ------------------------------------------------------------------

    async def start_up_endpointer(self) -> None:
        async def _init() -> Endpointer:
            ep = Endpointer()
            await ep.start_up()
            return ep

        async def _run(ep: Endpointer) -> None:
            await self._endpointer_loop(ep)

        async def _close(ep: Endpointer) -> None:
            await ep.shutdown()

        quest = await self.quest_manager.add(Quest("endpointer", _init, _run, _close))
        await quest.get()  # wait for the connection before returning
        logger.info("[AgenticRAG] Endpointer started.")

    def _prediction_audio_time_sec(self, frame_count: int | None) -> float:
        # Anticipator outputs one probability per 960-sample frame (40ms @ 24kHz).
        if frame_count is None:
            return self.audio_received_sec()
        return float(frame_count) * (ANTICIPATOR_FRAME_SAMPLES / SAMPLE_RATE)

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        """Forward audio to the endpointer (for anticipation) in addition to the
        normal STT/VAD processing done by the parent."""
        ep_quest = self.quest_manager.quests.get("endpointer")
        if ep_quest is not None:
            ep: Endpointer | None = ep_quest.get_nowait()
            if ep is not None:
                array = frame[1][0]  # mono
                try:
                    for i in range(0, len(array), ANTICIPATOR_FRAME_SAMPLES):
                        ep_chunk = array[i : i + ANTICIPATOR_FRAME_SAMPLES]
                        if len(ep_chunk) < ANTICIPATOR_FRAME_SAMPLES:
                            ep_chunk = np.pad(
                                ep_chunk,
                                (0, ANTICIPATOR_FRAME_SAMPLES - len(ep_chunk)),
                            )
                        await ep.send_audio(ep_chunk)
                except Exception as exc:
                    logger.debug("[AgenticRAG] endpointer send_audio failed: %r", exc)
        await super().receive(frame)

    async def _endpointer_loop(self, ep: Endpointer) -> None:
        """Consume anticipator predictions; fire a speculative note when the user
        is about to finish. Long-lived, with reconnect on disconnect."""
        reconnect_delay = ANTICIPATOR_RECONNECT_BASE_SEC
        while True:
            try:
                async for msg in ep:
                    prob = msg.user_end_probability
                    now = time.perf_counter()
                    state = self.chatbot.conversation_state()
                    cooldown_elapsed = now - self._last_anticipation_time
                    triggered = (
                        prob >= ANTICIPATE_THRESHOLD
                        and cooldown_elapsed >= ANTICIPATE_COOLDOWN_SEC
                        and state == "user_speaking"
                        # Only speculate during a genuine, ongoing user utterance:
                        # not during the post-VAD STT flush (stt_end_of_flush_time
                        # set), and not in the brief window after a commit before
                        # the assistant placeholder flips state to bot_speaking
                        # (_early_committed_note set). Otherwise a note could be
                        # started that later gets stale-committed on a wrong turn.
                        and self.stt_end_of_flush_time is None
                        and self._early_committed_note is None
                    )
                    if not triggered:
                        continue
                    self._last_anticipation_time = now
                    transcript = self.chatbot.last_message("user") or ""
                    logger.info(
                        "[AgenticRAG] Anticipation fired (prob=%.2f) -> speculating "
                        "note for %r",
                        prob, transcript,
                    )
                    await self._start_note_speculation(transcript)
                logger.warning(
                    "[AgenticRAG] Endpointer stream ended; reconnect in %.2fs.",
                    reconnect_delay,
                )
            except asyncio.CancelledError:
                raise
            except websockets.exceptions.ConnectionClosed as exc:
                logger.warning(
                    "[AgenticRAG] Endpointer closed (%s); reconnect in %.2fs.",
                    repr(exc), reconnect_delay,
                )
            except Exception as exc:
                logger.warning(
                    "[AgenticRAG] Endpointer loop error (%r); reconnect in %.2fs.",
                    exc, reconnect_delay,
                )
            await asyncio.sleep(reconnect_delay)
            try:
                await ep.start_up()
                reconnect_delay = ANTICIPATOR_RECONNECT_BASE_SEC
                logger.info("[AgenticRAG] Endpointer reconnected.")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[AgenticRAG] Endpointer reconnect failed: %r", exc)
                reconnect_delay = min(ANTICIPATOR_RECONNECT_MAX_SEC, reconnect_delay * 2)

    # ------------------------------------------------------------------
    # Speculative note: generate off the partial transcript, buffer its audio
    # ------------------------------------------------------------------

    @staticmethod
    def _note_committable(note: SpeculativeNote | None) -> bool:
        """A note is committable once it has buffered audio and isn't discarded.

        We do NOT require note.task.done(): the delayed-streams TTS keeps the
        receive stream open briefly after EOS, so a fully-synthesized note (all
        its audio already in audio_chunks) can still show task-not-done for a
        moment. _drain_note_audio polls until the task actually finishes, reading
        any trailing chunks, so committing an in-flight note is safe -- and the
        common case at VAD is exactly this (audio buffered, task about to return).
        """
        return (
            note is not None
            and not note.discarded
            and bool(note.audio_chunks)
        )

    @staticmethod
    def _teardown_note(note: SpeculativeNote | None) -> None:
        """Cancel a note's tasks and shut its TTS (audio already captured in
        note.audio_chunks stays valid). Safe to call on a done note."""
        if note is None:
            return
        note.discarded = True
        if note.tts_receive_task is not None and not note.tts_receive_task.done():
            note.tts_receive_task.cancel()
        if note.task is not None and not note.task.done():
            note.task.cancel()
        if note.tts_connection is not None:
            conn = note.tts_connection
            note.tts_connection = None
            asyncio.create_task(conn.shutdown())

    async def _start_note_speculation(self, transcript: str) -> None:
        """Re-speculate as the transcript GROWS, keeping the freshest note.

        Each new (more complete) transcript starts a new note; the superseded one
        is promoted to `_fallback_note` if it already has buffered audio (via
        _cancel_spec_note). At VAD we commit the freshest note that has audio,
        falling back to the last ready one -- so the spoken filler reflects as much
        of the question as had time to synthesize (a note off "...multi-speaker
        far-field ASR..." instead of the near-empty first hypothesis "Got it.").
        Skips re-speculating on an unchanged transcript (no point re-synthesizing
        identical text). `_spec_note`/`_fallback_note` are cleared every VAD.
        """
        async with self._spec_lock:
            cur = self._spec_note
            if (
                cur is not None
                and not cur.discarded
                and transcript.strip() == cur.trigger_transcript.strip()
            ):
                return  # identical transcript -> keep the in-flight speculation
            # Supersede: _cancel_spec_note keeps the outgoing note as the fallback
            # if it's already buffered audio, else discards it.
            await self._cancel_spec_note(reason="superseded")
            note = SpeculativeNote(trigger_transcript=transcript)
            self._spec_note = note
        note.task = asyncio.create_task(
            self._note_speculation_task(note), name="agentic_note_speculation"
        )
        # Live UI: we're anticipating (pre-VAD) off the growing ASR hypothesis.
        self._emit_agentic("speculating", transcript=transcript)

    async def _cancel_spec_note(self, reason: str = "cancelled") -> None:
        """Discard the current speculative note (must hold _spec_lock). If it was
        superseded but is already fully buffered, preserve it as a fallback so a
        later speculation that isn't ready in time doesn't cost us the filler."""
        note = self._spec_note
        if note is None or note.committed or note.discarded:
            return
        self._spec_note = None
        if reason == "superseded" and self._note_committable(note):
            # Keep the most recent fully-buffered note as the fallback.
            old_fallback = self._fallback_note
            self._fallback_note = note
            if old_fallback is not None and old_fallback is not note:
                self._teardown_note(old_fallback)
            logger.info("[AgenticRAG] Speculative note kept as fallback (%s).", reason)
            return
        self._teardown_note(note)
        logger.info("[AgenticRAG] Speculative note discarded (%s).", reason)

    async def _note_speculation_task(self, note: SpeculativeNote) -> None:
        """Generate the filler note (streamed) and buffer its TTS audio."""
        _t0 = time.perf_counter()
        try:
            tts = await find_instance(
                "tts",
                partial(
                    TextToSpeech,
                    recorder=None,  # don't record speculative audio
                    get_time=self.audio_received_sec,
                    voice=self.tts_voice,
                ),
            )
            note.tts_connection = tts
            note.tts_receive_task = asyncio.create_task(
                self._spec_note_tts_receive(tts, note), name="agentic_note_tts_recv"
            )

            stream = await self._agentic_llm_client.chat.completions.create(
                model=AGENTIC_LLM_MODEL,
                messages=[
                    {"role": "system", "content": SPECULATIVE_NOTE_PROMPT},
                    {"role": "user", "content": _speculative_note_user_msg(note.trigger_transcript)},
                ],
                max_tokens=SPEC_NOTE_MAX_TOKENS,
                temperature=0.7,
                stream=True,
                timeout=NOTE_TIMEOUT_SEC,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            send_buf = ""
            async with stream:
                async for chunk in stream:
                    if note.discarded:
                        break
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content
                    if not delta:
                        continue
                    note.text_tokens.append(delta)
                    send_buf += delta
                    # Stream whole words to the TTS so audio buffers as we go.
                    sp = max(send_buf.rfind(" "), send_buf.rfind("\n"))
                    if sp >= 0:
                        to_send = _sanitize_for_tts(send_buf[: sp + 1])
                        send_buf = send_buf[sp + 1 :]
                        if to_send.strip():
                            await tts.send(to_send)
            _t_llm_done = time.perf_counter()
            if not note.discarded:
                tail = _sanitize_for_tts(send_buf)
                if tail.strip():
                    await tts.send(tail)
                await tts.send(TTSClientEosMessage())  # flush the tail audio
                # Drain remaining audio; the receive task fills note.audio_chunks.
                if note.tts_receive_task is not None:
                    await note.tts_receive_task
            logger.info(
                "[AgenticRAG] SPEC NOTE ready: text=%r chunks=%d llm=%.2fs synth_total=%.2fs discarded=%s",
                note.text, len(note.audio_chunks),
                _t_llm_done - _t0, time.perf_counter() - _t0, note.discarded,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[AgenticRAG] speculative note generation failed: %r", exc)
        finally:
            # If not committed, close the spec TTS (committed path owns cleanup).
            if not note.committed and note.tts_connection is not None:
                conn = note.tts_connection
                note.tts_connection = None
                try:
                    await conn.shutdown()
                except Exception:
                    pass

    async def _spec_note_tts_receive(
        self, tts: TextToSpeech, note: SpeculativeNote
    ) -> None:
        """Collect speculative TTS audio into note.audio_chunks."""
        try:
            async for message in tts:
                if note.discarded:
                    break
                if isinstance(message, TTSAudioMessage):
                    note.audio_chunks.append(np.array(message.pcm, dtype=np.float32))
                # TTSTextMessage ignored -- text already accumulated from the LLM.
        except Exception as exc:
            if not note.discarded:
                logger.debug("[AgenticRAG] spec note TTS receive error: %r", exc)

    # ------------------------------------------------------------------
    # Commit at VAD (the sub-600ms filler moment) + entry point
    # ------------------------------------------------------------------

    async def _on_vad_pause(self) -> None:
        """Fired the instant VAD detects turn-end, BEFORE the STT flush. If a
        speculative note is buffered, commit it here so its audio starts playing
        immediately -- we wait for nothing else. The grounded loop runs behind it
        (it still needs the finalized transcript, which arrives after the flush).

        If no usable note is buffered, this no-ops and the handler falls back to
        the normal post-flush path in _generate_response -- byte-for-byte the
        current behavior.
        """
        async with self._spec_lock:
            sn, fn = self._spec_note, self._fallback_note
            logger.info(
                "[AgenticRAG] VAD spec state: cur=%s fallback=%s",
                (None if sn is None else
                 f"chunks={len(sn.audio_chunks)},done={sn.task.done() if sn.task else None},disc={sn.discarded}"),
                (None if fn is None else
                 f"chunks={len(fn.audio_chunks)},done={fn.task.done() if fn.task else None},disc={fn.discarded}"),
            )
            # Prefer the current speculation if it's fully buffered; otherwise the
            # fallback (an earlier, generic, fully-buffered note). Tear down
            # whichever we don't commit so nothing lingers into the next turn.
            note: SpeculativeNote | None = None
            if self._note_committable(self._spec_note):
                note = self._spec_note
            elif self._note_committable(self._fallback_note):
                note = self._fallback_note
            if note is not self._spec_note:
                self._teardown_note(self._spec_note)
            if note is not self._fallback_note:
                self._teardown_note(self._fallback_note)
            self._spec_note = None
            self._fallback_note = None
            if note is None:
                return  # graceful fallback: nothing ready -> normal path
            note.committed = True

        self._early_committed_note = note
        self._note_flush_event = asyncio.Event()
        # The turn effectively starts now (VAD), not at _generate_response -- used
        # by the interrupt grace period.
        self._current_turn_started_at = self.audio_received_sec()
        # Anticipates the assistant placeholder _generate_response() adds after the
        # flush. The flush only appends tokens to the existing user message (same
        # role -> no new list entry), so len(chat_history) is unchanged until then.
        generating_message_i = len(self.chatbot.chat_history) + 1
        flush_event = self._note_flush_event
        logger.info(
            "[AgenticRAG] VAD: committing speculative note (%d chunks, gmi=%d) | "
            "ASR hypothesis at speculation=%r -> speculation answer=%r",
            len(note.audio_chunks), generating_message_i,
            note.trigger_transcript, note.text,
        )
        quest = Quest.from_run_step(
            "llm",
            lambda: self._committed_note_then_loop_task(
                note, generating_message_i, flush_event
            ),
        )
        await self.quest_manager.add(quest)

    async def _committed_note_then_loop_task(
        self,
        note: SpeculativeNote,
        generating_message_i: int,
        flush_event: asyncio.Event,
    ) -> None:
        """Play the committed note audio immediately, then run the agentic loop
        behind it (skipping the loop's own opener note)."""
        await self.output_queue.put(
            ora.ResponseCreated(
                response=ora.Response(
                    status="in_progress",
                    voice=self.tts_voice or "missing",
                    chat_history=self.chatbot.chat_history,
                )
            )
        )

        trace: dict[str, Any] = {
            "user_transcript": "",
            "transcript_final_sec": round(self.audio_received_sec(), 2),
            "speculative": True,
            # ASR hypothesis the filler was speculated from (pre-VAD, partial) and
            # the resulting spoken filler ("speculation answer").
            "speculative_transcript": note.trigger_transcript,
            "speculative_note": note.text.strip(),
            "notes": [],
            "answer": "",
            "loop": [],
            "totals": {},
            "interrupted": False,
        }
        t_turn_start = time.perf_counter()
        tts_quest = None
        drain_task: asyncio.Task | None = None
        # Set once the opener's commit-time audio is fully enqueued; the loop waits
        # on THIS (not on drain_task/note.task) before its first main-TTS send, so
        # it doesn't stall on the spec TTS stream closing (~2s of dead air).
        note_enqueued = asyncio.Event()

        try:
            # Start playing the buffered note audio immediately (this is the
            # sub-600ms filler) and bring up the MAIN tts for the answer in
            # parallel. Overlaps with the STT flush wait below.
            drain_task = asyncio.create_task(
                self._drain_note_audio(note, generating_message_i, note_enqueued)
            )
            tts_quest = await self.start_up_tts(generating_message_i)
            tts = await tts_quest.get()

            # Grounded work needs the finalized transcript -> wait for the flush.
            await flush_event.wait()

            if self._interrupted(generating_message_i):
                self._emit_agentic("interrupted")
                return

            note_text = note.text.strip()
            trace["user_transcript"] = self.chatbot.last_message("user") or ""
            if note_text:
                # The note played on the spec TTS (not the main _tts_loop), so its
                # text isn't auto-synced to chat_history -- record it explicitly so
                # the answer continuation + history form one continuous assistant
                # turn.
                trace["notes"].append(note_text)
                await self.add_chat_message_delta(
                    note_text + " ", "assistant",
                    generating_message_i=generating_message_i,
                )
                self._emit_agentic("turn_start", query=trace["user_transcript"])
                # turn=0 + speculative flag: the UI renders this as the anticipated
                # opener. Emitted NOW (right after flush) -- NOT gated on the note
                # audio finishing -- so the "anticipating" pill flips to
                # "anticipated" immediately.
                self._emit_agentic(
                    "note", turn=0, text=note_text,
                    speculative=True, hypothesis=note.trigger_transcript,
                )

            # Run the grounded loop CONCURRENTLY with the opener audio draining
            # (don't await the drain first -- that blocked decompose+retrieval
            # until the opener's TTS fully rendered, ~2-3s of dead air). The loop
            # gets `note_gate=drain_task`: it does decompose+retrieval freely
            # (no audio) and only awaits the gate right before its FIRST TTS send,
            # so the tool-note audio still can't jump ahead of the opener audio,
            # but the slow grounded work now overlaps the opener playback.
            # Keep the loop's own first note (skip_first_note=False): the opener
            # covers VAD->decompose, the loop note covers decompose->answer.
            await self._run_agentic_loop(
                trace,
                generating_message_i,
                tts,
                preplayed_notes=[note_text] if note_text else [],
                skip_first_note=False,
                note_gate=note_enqueued,
            )
            # drain_task keeps running in the background to flush any trailing
            # opener chunks; the `finally` below cancels it if it's still going.
        except asyncio.CancelledError:
            logger.info(
                "[AgenticRAG] committed-note task CANCELLED gmi=%d", generating_message_i
            )
            self._emit_agentic("interrupted")
            raise
        finally:
            if drain_task is not None and not drain_task.done():
                drain_task.cancel()
            # Belt-and-suspenders: make sure the spec TTS connection is closed.
            if note.tts_connection is not None:
                conn = note.tts_connection
                note.tts_connection = None
                try:
                    await conn.shutdown()
                except Exception:
                    pass
            trace["totals"]["turn_latency_ms"] = round(
                (time.perf_counter() - t_turn_start) * 1000.0
            )
            trace["interrupted"] = len(self.chatbot.chat_history) > generating_message_i
            self._agentic_traces.append(trace)

        if not trace["interrupted"] and tts_quest is not None:
            tts = await tts_quest.get()
            logger.info("[AgenticRAG] Sending TTS EOS (committed-note path).")
            await tts.send(TTSClientEosMessage())

    async def _drain_note_audio(
        self,
        note: SpeculativeNote,
        generating_message_i: int,
        enqueued_event: asyncio.Event | None = None,
    ) -> None:
        """Enqueue the buffered speculative-note audio onto the output queue in
        order, ALL AT ONCE (not paced). Dumping keeps the opener a single, strictly-
        ordered block ahead of the answer in output_queue: the main-TTS answer audio
        (paced by its own RealtimeQueue) then always lands strictly AFTER the opener,
        never interleaved with it. Interleaving is what caused the audible clicks --
        two independently real-time-paced streams merging frame-by-frame at the seam,
        underrunning the frontend worklet. The cost of dumping (a deep client buffer,
        which slows barge-in) is handled the RIGHT way: the frontend flushes its audio
        worklet on `unmute.interrupted_by_vad` (see Saga.tsx), so a barge-in is snappy
        regardless of buffer depth -- a cleaner split than pacing here.

        `enqueued_event` (if given) is set once the commit-time backlog is queued --
        the only thing the loop must wait for before feeding the main TTS. It must NOT
        wait for note.task to finish (the spec TTS stream stays open ~2s after EOS);
        trailing chunks drain here in the background, still ahead of the main TTS."""
        idx = 0
        signalled = False
        try:
            while True:
                if self._interrupted(generating_message_i):
                    return
                while idx < len(note.audio_chunks):
                    await self.output_queue.put((SAMPLE_RATE, note.audio_chunks[idx]))
                    idx += 1
                # Release the loop's gate once the commit-time backlog is queued.
                if not signalled and enqueued_event is not None:
                    enqueued_event.set()
                    signalled = True
                if note.task is None or note.task.done():
                    while idx < len(note.audio_chunks):
                        await self.output_queue.put((SAMPLE_RATE, note.audio_chunks[idx]))
                        idx += 1
                    return
                await asyncio.sleep(0.01)
        finally:
            # Never leave the loop blocked on the gate, even if we return/cancel early.
            if enqueued_event is not None and not enqueued_event.is_set():
                enqueued_event.set()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def _generate_response(self) -> None:
        # Early-commit path: a speculative note was committed at VAD and its
        # audio+loop task is already running. STT flush is now done, so add the
        # assistant placeholder and release the loop to read the final transcript.
        early_note = self._early_committed_note
        if early_note is not None:
            self._early_committed_note = None
            flush_event = self._note_flush_event
            self._note_flush_event = None
            if "llm" in self.quest_manager.quests:
                await self.add_chat_message_delta("", "assistant")
                if flush_event is not None:
                    flush_event.set()
                return
            # The committed-note quest was interrupted before the flush completed;
            # fall through to normal generation.
            logger.info(
                "[AgenticRAG] committed-note quest gone before flush; normal gen."
            )

        logger.info(
            "[AgenticRAG] DEBUG _generate_response() called, chat_history_len=%d, last_user=%r",
            len(self.chatbot.chat_history),
            self.chatbot.last_message("user"),
        )
        self._current_turn_started_at = self.audio_received_sec()
        await self.add_chat_message_delta("", "assistant")
        quest = Quest.from_run_step("llm", self._agentic_generate_response_task)
        await self.quest_manager.add(quest)

    async def _agentic_generate_response_task(self) -> None:
        generating_message_i = len(self.chatbot.chat_history)
        logger.info(
            "[AgenticRAG] DEBUG task START generating_message_i=%d", generating_message_i
        )

        await self.output_queue.put(
            ora.ResponseCreated(
                response=ora.Response(
                    status="in_progress",
                    voice=self.tts_voice or "missing",
                    chat_history=self.chatbot.chat_history,
                )
            )
        )

        trace: dict[str, Any] = {
            "user_transcript": self.chatbot.last_message("user") or "",
            "transcript_final_sec": round(self.audio_received_sec(), 2),
            "notes": [],
            "answer": "",
            "loop": [],
            "totals": {},
            "interrupted": False,
        }
        t_turn_start = time.perf_counter()
        if trace["user_transcript"].strip():
            self._emit_agentic("turn_start", query=trace["user_transcript"])
        tts_quest = None

        try:
            # start_up_tts() must be INSIDE the try so that if it gets
            # interrupted the trace is still appended in `finally`.
            tts_quest = await self.start_up_tts(generating_message_i)
            tts = await tts_quest.get()
            await self._run_agentic_loop(trace, generating_message_i, tts)
        except asyncio.CancelledError:
            # Real interruption: barge-in cancels the "llm" quest (this task)
            # AND the "tts" quest (via interrupt_bot()). Nothing else to tear
            # down -- there's no child delivery task anymore.
            logger.info(
                "[AgenticRAG] DEBUG task CANCELLED generating_message_i=%d",
                generating_message_i,
            )
            self._emit_agentic("interrupted")
            raise
        finally:
            trace["totals"]["turn_latency_ms"] = round(
                (time.perf_counter() - t_turn_start) * 1000.0
            )
            trace["interrupted"] = len(self.chatbot.chat_history) > generating_message_i
            self._agentic_traces.append(trace)
            logger.info(
                "[AgenticRAG] DEBUG task END generating_message_i=%d interrupted=%s "
                "answer=%r",
                generating_message_i,
                trace["interrupted"],
                trace.get("answer"),
            )

        # Normal completion only (a cancellation re-raised above skips this):
        # flush the TTS so it emits the tail of the last text and finishes the
        # turn. Skip if we were passively interrupted mid-loop.
        if not trace["interrupted"] and tts_quest is not None:
            tts = await tts_quest.get()
            logger.info("[AgenticRAG] Sending TTS EOS.")
            await tts.send(TTSClientEosMessage())

    # ------------------------------------------------------------------
    # The agentic loop itself (mirrors reference pipeline.run_pipeline)
    # ------------------------------------------------------------------

    def _interrupted(self, generating_message_i: int) -> bool:
        return len(self.chatbot.chat_history) > generating_message_i

    def _emit_agentic(self, kind: str, **data: Any) -> None:
        """Push a live SAGA trace event to the client (best-effort; never raises
        into the loop). The websocket layer serializes any ora.ServerEvent."""
        try:
            self.output_queue.put_nowait(
                ora.UnmuteAgenticUpdate(kind=kind, data=data)
            )
        except Exception as exc:  # queue swapped mid-interrupt, etc.
            logger.debug("[AgenticRAG] _emit_agentic(%s) dropped: %r", kind, exc)

    async def _run_agentic_loop(
        self,
        trace: dict[str, Any],
        generating_message_i: int,
        tts,
        preplayed_notes: list[str] | None = None,
        skip_first_note: bool = False,
        note_gate: "asyncio.Event | None" = None,
    ) -> None:
        query = trace["user_transcript"]
        ephemeral_history: list[dict[str, Any]] = []
        # Seed with any note already spoken before the loop (e.g. a committed
        # speculative opener) so the continuation prompt knows what was said and
        # subsequent notes don't repeat it.
        spoken_notes: list[str] = list(preplayed_notes or [])
        direct_answer_text: str | None = None
        answer_streamed = False

        for turn_index in range(1, self._max_turns + 1):
            if self._interrupted(generating_message_i):
                return

            logger.info(
                "[AgenticRAG] DEBUG turn=%d decompose START gen_i=%d",
                turn_index, generating_message_i,
            )
            # On turns where we already have retrieved results, use the
            # continuation prompt and stream the answer straight to the TTS as
            # it generates -- this is one merged call (decide-more-tools OR
            # answer) instead of a separate decompose + answer, and it keeps
            # the TTS continuously fed so the prior note doesn't stall waiting
            # for continuation text. Turn 1 (no results yet) expects tool_calls,
            # so don't stream it (the spoken note covers that latency).
            if ephemeral_history:
                system_override = _continuation_prompt(spoken_notes)
                stream_tts = tts
            else:
                system_override = None
                stream_tts = None
            tool_calls, answer_text, decompose_timings, streamed = (
                await self._query_decompose(
                    ephemeral_history,
                    system_override=system_override,
                    tts=stream_tts,
                    generating_message_i=generating_message_i,
                    # Force retrieval on turn 1 of a real RESEARCH question (not
                    # the empty-transcript greeting, not later result-in-hand
                    # turns, and NOT a purely conversational/meta turn like "can
                    # you hear me?" -- forcing a tool call there turns it into an
                    # irrelevant paper dump; those answer directly via "auto").
                    force_tools=(
                        not ephemeral_history
                        and bool(query.strip())
                        and not _is_conversational_query(query)
                    ),
                )
            )
            logger.info(
                "[AgenticRAG] DEBUG turn=%d decompose END tool_calls=%s total=%.2fs streamed=%s",
                turn_index, [tc["name"] for tc in tool_calls],
                decompose_timings.get("total", -1), streamed,
            )
            turn_trace: dict[str, Any] = {
                "turn": turn_index,
                "latency_sec": round(decompose_timings.get("total", 0.0), 3),
                "tool_calls": [
                    {"name": tc["name"], "arguments": tc["arguments"]} for tc in tool_calls
                ],
                "results": [],
            }
            trace["loop"].append(turn_trace)

            if not tool_calls:
                direct_answer_text = answer_text
                answer_streamed = streamed
                break

            # Live UI: the parallel tool calls this turn is issuing.
            self._emit_agentic(
                "tool_calls",
                turn=turn_index,
                calls=[
                    {
                        "id": tc.get("id"),
                        "tool": tc["name"],
                        "query": _tool_query_str(tc["arguments"]),
                        "freshness": tc["arguments"].get("freshness"),
                        "impact": tc["arguments"].get("paper_impact"),
                    }
                    for tc in tool_calls
                ],
            )

            # Structural filler cap (no playback gating -- see MAX_LOOP_NOTES): at
            # most ONE bridge note, on turn 1. Later search turns retrieve silently
            # so we never stack "let me also check... let me also check...". Turn 1
            # also skips its note if a speculative opener already covered it
            # (skip_first_note).
            generate_note = (
                len(trace["notes"]) < MAX_LOOP_NOTES
                and not (skip_first_note and turn_index == 1)
            )
            # When a speculative opener already played (spoken_notes non-empty on
            # turn 1), keep this first tool note very short -- the opener already
            # bought the time, so a long note here only delays the answer.
            brief_note = turn_index == 1 and bool(spoken_notes)
            note_text, tool_results = await self._retrieve_and_note(
                query, tool_calls, spoken_notes,
                generate_note=generate_note, brief_note=brief_note,
            )
            # Live UI: what came back (papers, tethered to their search).
            self._emit_agentic(
                "tool_results",
                turn=turn_index,
                results=[
                    {
                        "id": tr.get("id"),
                        "tool": tr.get("name"),
                        "count": len(tr.get("result", {}).get("results", [])),
                        "error": tr.get("result", {}).get("error"),
                        "papers": _papers_for_ui(tr.get("result", {}).get("results", [])),
                    }
                    for tr in tool_results
                ],
            )
            turn_trace["results"] = [
                {
                    "tool": tr.get("name"),
                    "count": len(tr.get("result", {}).get("results", [])),
                    "error": tr.get("result", {}).get("error"),
                    "papers": _summarize_papers(tr.get("result", {}).get("results", [])),
                }
                for tr in tool_results
            ]

            # Gate the loop's FIRST audio output on the opener's commit-time audio
            # being ENQUEUED (not on the spec TTS stream closing), so tool-note/
            # answer audio can't jump ahead of the opener in the queue while also
            # not eating the ~2s of dead air that waiting for note.task would.
            # decompose+retrieval above already ran concurrently with the drain, so
            # this usually blocks little or nothing. Idempotent: only turn 1 (the
            # first place the loop can emit audio -- turn 1 never streams) blocks;
            # note_gate is cleared after. Turn 1 always reaches here before any send.
            if note_gate is not None:
                await note_gate.wait()
                note_gate = None

            if note_text:
                spoken_notes.append(note_text)
                trace["notes"].append(note_text)
                self._emit_agentic("note", turn=turn_index, text=note_text)
                # Continuous feeding: send the note the moment it's ready.
                if self._interrupted(generating_message_i):
                    return
                await tts.send(_sanitize_for_tts(note_text))

            ephemeral_history.append(
                {
                    "tool_calls": tool_calls,
                    "tool_results": tool_results,
                    "reasoning": decompose_timings.get("reasoning", ""),
                }
            )
        else:
            # Exhausted max_turns without the LLM stopping on its own: force an
            # answer from whatever was retrieved.
            direct_answer_text = None

        if self._interrupted(generating_message_i):
            return

        # ---- Answer ----  (no truncation: speak the model's full reply)
        if ephemeral_history and direct_answer_text and direct_answer_text.strip():
            # The merged decompose call already produced (and, if streamed=True,
            # already spoke) the continuation answer. Send it only if it wasn't
            # streamed live.
            final_text = direct_answer_text
            if not answer_streamed and not self._interrupted(generating_message_i):
                await tts.send(_sanitize_for_tts(final_text))
        elif ephemeral_history:
            # Forced final answer (max_turns exhausted) or the merged call came
            # back empty: fall back to a dedicated accumulated answer call.
            logger.info("[AgenticRAG] DEBUG answer START (fallback/forced)")
            answer_t0 = time.perf_counter()
            final_text = await self._generate_answer(ephemeral_history, spoken_notes)
            logger.info(
                "[AgenticRAG] DEBUG answer END len=%d total=%.2fs",
                len(final_text), time.perf_counter() - answer_t0,
            )
            if final_text.strip() and not self._interrupted(generating_message_i):
                await tts.send(_sanitize_for_tts(final_text))
        else:
            # No retrieval happened: greeting (empty user transcript) or a direct
            # factoid the model chose to answer from its own knowledge.
            final_text = _strip_control_artifacts(direct_answer_text or "")
            if final_text.strip() and not self._interrupted(generating_message_i):
                await tts.send(_sanitize_for_tts(final_text))

        trace["answer"] = final_text
        if final_text.strip():
            self._emit_agentic("answer", text=final_text)
        self._emit_agentic("turn_end")
        trace["totals"] = {
            "loop_turns": len(trace["loop"]),
            "n_tool_calls": sum(len(t["tool_calls"]) for t in trace["loop"]),
            "n_notes": len(spoken_notes),
            "answer_chars": len(final_text),
            "answer_streamed": answer_streamed,
        }

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _base_messages(
        self,
        ephemeral_history: list[dict[str, Any]],
        system_override: str | None = None,
    ) -> list[dict[str, Any]]:
        history = list(self.chatbot.preprocessed_messages())
        if (
            system_override is not None
            and history
            and history[0].get("role") == "system"
        ):
            history = [{"role": "system", "content": system_override}, *history[1:]]
        # Cap persistent history: always keep the system prompt, then only the
        # most recent messages (recent turns are what matter for coreference).
        if len(history) > MAX_HISTORY_MSGS + 1:
            history = [history[0]] + history[-MAX_HISTORY_MSGS:]

        # Add this loop's tool-call/result turns, dropping the OLDEST ones if the
        # assembled prompt would still blow the context budget (a broad search
        # can dump a lot). Keeps the most recent results, which the answer needs.
        turns = list(ephemeral_history)
        while True:
            messages = list(history)
            for turn in turns:
                messages.extend(
                    _tool_calls_to_openai_messages(turn["tool_calls"], turn["tool_results"])
                )
            if _estimate_tokens(messages) <= MAX_INPUT_TOKENS or len(turns) <= 1:
                if _estimate_tokens(messages) > MAX_INPUT_TOKENS:
                    logger.warning(
                        "[AgenticRAG] prompt still ~%d tokens after trimming to the last "
                        "tool-result turn -- proceeding (may be truncated by the server)",
                        _estimate_tokens(messages),
                    )
                return messages
            dropped = turns.pop(0)
            logger.info(
                "[AgenticRAG] context budget: dropped oldest in-loop tool results "
                "(%s) to fit under %d tokens",
                [tc.get("name") for tc in dropped.get("tool_calls", [])],
                MAX_INPUT_TOKENS,
            )

    async def _flush_words_to_tts(self, buf: str, tts) -> tuple[str, str]:
        """Send all *whole* words in `buf` to the TTS, keeping any trailing
        partial word (no whitespace after it yet) buffered for later. Returns
        (remaining_buffer, text_actually_sent). Splitting on the last whitespace
        keeps word boundaries intact so the TTS never mispronounces a split
        word (same reason the base handler rechunks to words)."""
        sp = max(buf.rfind(" "), buf.rfind("\n"))
        if sp < 0:
            return buf, ""
        to_send = _sanitize_for_tts(buf[: sp + 1])
        rest = buf[sp + 1 :]
        if to_send.strip():
            await tts.send(to_send)
            return rest, to_send
        return rest, ""

    async def _query_decompose(
        self,
        ephemeral_history: list[dict[str, Any]],
        system_override: str | None = None,
        tts=None,
        generating_message_i: int | None = None,
        force_tools: bool = False,
    ) -> tuple[list[dict[str, Any]], str, dict[str, Any], bool]:
        """One agentic-turn call (mirrors reference pipeline call_agentic_turn):
        returns either tool_calls (search more) OR the answer text.

        When `tts` is given, answer *content* is streamed to the TTS word-by-word
        as it generates (not accumulated first) -- this keeps the delayed-streams
        TTS continuously fed so it doesn't stall the tail of the previous note
        waiting for continuation text. Content is only streamed once we're
        confident this is an answer (no tool_calls seen yet + a small commit
        threshold), so a search turn's tool_calls are never mistaken for speech.
        The whole answer is streamed -- no truncation. The 4th return value says
        whether the answer was already sent to TTS.
        """
        from unmute.agentic_rag_tools import TOOLS

        # Prompt-level forcing (see FORCE_RETRIEVAL_DIRECTIVE): on the turn-1
        # tool-decision call for a real research question, append a directive that
        # tells the model to search NOW instead of answering from context. This
        # replaces the old tool_choice="required", which stalled the stream for
        # many seconds on this Qwen build. tool_choice stays "auto".
        if force_tools and system_override is None:
            system_override = SYSTEM_PROMPT + FORCE_RETRIEVAL_DIRECTIVE
        messages = self._base_messages(ephemeral_history, system_override=system_override)
        t0 = time.perf_counter()
        est_in = _estimate_tokens(messages)
        # ALWAYS "auto" -- never "required". `tool_choice="required"` makes vLLM run
        # GUIDED decoding to force a tool call, and on this Qwen build that stalls
        # hard: the stream is accepted (HTTP 200) but no token is emitted for many
        # seconds, so the whole turn dead-airs for up to DECOMPOSE_TIMEOUT_SEC after
        # the speculative opener before the first tool window appears (observed live:
        # 5-15s gap; debug log item 5). The reference pipeline runs "auto" and grounds
        # fine because SYSTEM_PROMPT already mandates retrieval for any research
        # question ("ALWAYS retrieve before answering ANY question about research...").
        # So grounding is enforced by the prompt, not by a decoding constraint that
        # blocks the turn. `force_tools` is retained in the signature (callers still
        # pass it) but no longer maps to "required".
        tool_choice = "auto"
        stream = await self._agentic_llm_client.chat.completions.create(
            model=AGENTIC_LLM_MODEL,
            messages=messages,  # type: ignore[arg-type]
            tools=TOOLS,  # type: ignore[arg-type]
            tool_choice=tool_choice,
            stream=True,
            stream_options={"include_usage": True},  # -> final chunk carries usage
            timeout=DECOMPOSE_TIMEOUT_SEC,  # bound a wedged stream (see constant)
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        tc_chunks: dict[int, dict[str, str]] = {}
        text_chunks: list[str] = []
        saw_tool_call = False
        committed = False  # committed to speaking this turn's content as the answer
        send_buf = ""
        streamed = False
        usage = None  # vLLM prompt/completion token counts (authoritative)
        COMMIT_CHARS = 24  # brief reasoning preambles won't trip the commit

        try:
            async with stream:
                async for chunk in stream:
                    if getattr(chunk, "usage", None):
                        usage = chunk.usage  # arrives in a final choices-less chunk
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.tool_calls:
                        saw_tool_call = True
                        for tcd in delta.tool_calls:
                            idx = tcd.index
                            slot = tc_chunks.setdefault(
                                idx, {"id": "", "name": "", "arguments": ""}
                            )
                            if tcd.id:
                                slot["id"] += tcd.id
                            if tcd.function:
                                if tcd.function.name:
                                    slot["name"] += tcd.function.name
                                if tcd.function.arguments:
                                    slot["arguments"] += tcd.function.arguments
                    if delta.content:
                        text_chunks.append(delta.content)
                        if (
                            tts is not None
                            and not saw_tool_call
                            and not (
                                generating_message_i is not None
                                and self._interrupted(generating_message_i)
                            )
                        ):
                            send_buf += delta.content
                            if (
                                not committed
                                and len(send_buf.strip()) >= COMMIT_CHARS
                                and "<" not in send_buf
                            ):
                                committed = True
                            if committed:
                                send_buf, sent = await self._flush_words_to_tts(send_buf, tts)
                                if sent:
                                    streamed = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A wedged/timed-out vLLM stream must not hang the turn. Proceed with
            # whatever tool_calls/text accumulated before the stall; if nothing
            # accumulated, the caller falls through to the answer path (which has
            # its own retry) rather than getting stuck. See DECOMPOSE_TIMEOUT_SEC.
            logger.warning(
                "[AgenticRAG] decompose stream aborted after %.1fs (%r); "
                "proceeding with %d partial tool-call(s), %d text chunk(s)",
                time.perf_counter() - t0, exc, len(tc_chunks), len(text_chunks),
            )
        total = time.perf_counter() - t0
        if usage is not None:
            logger.info(
                "[AgenticRAG] LLM tokens: prompt=%d completion=%d total=%d "
                "(ctx=32768, my_estimate_was=%d)",
                usage.prompt_tokens, usage.completion_tokens, usage.total_tokens, est_in,
            )
        else:
            logger.info("[AgenticRAG] LLM tokens: (no usage) my_estimate=%d", est_in)

        calls: list[dict[str, Any]] = []
        for idx in sorted(tc_chunks):
            tc = tc_chunks[idx]
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            calls.append(
                {"name": tc["name"], "arguments": args, "id": tc["id"] or f"call_{idx}"}
            )

        answer_text = _strip_control_artifacts("".join(text_chunks).strip())
        if calls:
            # Search turn: any streamed content shouldn't have happened (we gate
            # streaming on not-yet-seen tool calls), but if a preamble slipped
            # through, it's already spoken -- nothing to undo. Report no answer.
            return calls, "", {"total": total}, False

        interrupted = generating_message_i is not None and self._interrupted(
            generating_message_i
        )
        if tts is not None:
            if committed:
                # Flush the trailing partial word held back by the word-splitter.
                if send_buf.strip() and not interrupted:
                    await tts.send(_sanitize_for_tts(send_buf))
                    streamed = True
                return [], answer_text, {"total": total}, streamed
            # Too short to have committed to word-streaming: send it whole.
            if answer_text.strip() and not interrupted:
                await tts.send(_sanitize_for_tts(answer_text))
                return [], answer_text, {"total": total}, True
            return [], answer_text, {"total": total}, False
        return [], answer_text, {"total": total}, streamed

    async def _generate_note(
        self,
        query: str,
        tool_calls: list[dict[str, Any]],
        prior_notes: list[str],
        brief: bool = False,
    ) -> str:
        user_msg = _spoken_note_user_msg(query, tool_calls, prior_notes, brief=brief)
        try:
            resp = await self._agentic_llm_client.chat.completions.create(
                model=AGENTIC_LLM_MODEL,
                messages=[
                    {"role": "system", "content": SPOKEN_NOTE_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=24 if brief else 60,
                temperature=0.7,
                timeout=NOTE_TIMEOUT_SEC,
                stream=False,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return (resp.choices[0].message.content or "").strip().strip('"')
        except Exception as exc:
            logger.warning("[AgenticRAG] note generation failed: %r", exc)
            return ""

    async def _retrieve_and_note(
        self,
        query: str,
        tool_calls: list[dict[str, Any]],
        prior_notes: list[str],
        generate_note: bool = True,
        brief_note: bool = False,
    ) -> tuple[str, list[dict[str, Any]]]:
        logger.info(
            "[AgenticRAG] DEBUG retrieve_and_note START tools=%d generate_note=%s brief=%s",
            len(tool_calls), generate_note, brief_note,
        )
        # generate_note=False when a speculative opener already played this turn
        # -- retrieve only, no redundant "let me check" note.
        note_task = (
            asyncio.create_task(
                self._generate_note(query, tool_calls, prior_notes, brief=brief_note)
            )
            if generate_note
            else None
        )
        retrieval_task = asyncio.gather(
            *[self._rag_client.execute_tool_call(tc) for tc in tool_calls]
        )
        tool_results = await retrieval_task
        # Log the query (args) and what came back (count / matched author / error /
        # top titles) so an empty or errored RAG call is visible in the backend
        # log immediately -- e.g. timeouts under node load, or a genuine miss --
        # instead of only surfacing as the model improvising "the database is shy".
        for tc, tr in zip(tool_calls, tool_results):
            res = tr.get("result", {}) if isinstance(tr, dict) else {}
            results = res.get("results", []) or []
            titles = [
                (r.get("title") or r.get("session_name") or "?")
                for r in results[:3]
                if isinstance(r, dict)
            ]
            logger.info(
                "[AgenticRAG] RAG %s(%s) -> %d results%s%s%s",
                tc.get("name"),
                tc.get("arguments"),
                len(results),
                f" matched_author={res.get('matched_author')!r}"
                if res.get("matched_author") is not None else "",
                f" ERROR={res.get('error')!r}" if res.get("error") else "",
                (" | " + " | ".join(titles)) if titles else "",
            )
        logger.info("[AgenticRAG] DEBUG retrieval done, awaiting note...")
        note_text = await note_task if note_task is not None else ""
        logger.info("[AgenticRAG] DEBUG retrieve_and_note END note=%r", note_text)
        return note_text, list(tool_results)

    async def _generate_answer(
        self, ephemeral_history: list[dict[str, Any]], spoken_notes: list[str]
    ) -> str:
        """Final grounded answer, generated as a NORMAL completion with a
        continuation system-prompt override (reference pipeline approach).

        Deliberately NOT continue_final_message with the note as an assistant
        prefix -- that made the model treat the assistant turn as already
        finished and emit nothing (empty answers). See CLAUDE.md / debug log.
        """
        system_override = _continuation_prompt(spoken_notes)
        messages = self._base_messages(ephemeral_history, system_override=system_override)

        # Occasionally comes back empty -- retry once at a higher temperature
        # before giving up, rather than leaving the turn with no answer at all.
        text = ""
        for attempt in range(2):
            chunks: list[str] = []
            try:
                stream = await self._agentic_llm_client.chat.completions.create(
                    model=AGENTIC_LLM_MODEL,
                    messages=messages,  # type: ignore[arg-type]
                    tool_choice="none",
                    stream=True,
                    temperature=0.7 if attempt == 0 else 1.0,
                    timeout=ANSWER_TIMEOUT_SEC,  # bound a wedged stream
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                async with stream:
                    async for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            chunks.append(chunk.choices[0].delta.content)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "[AgenticRAG] answer stream aborted (attempt %d/2): %r",
                    attempt + 1, exc,
                )
            text = _strip_control_artifacts("".join(chunks).strip())
            if text:
                return text
            logger.warning(
                "[AgenticRAG] answer came back empty (attempt %d/2), retrying",
                attempt + 1,
            )
        return text
