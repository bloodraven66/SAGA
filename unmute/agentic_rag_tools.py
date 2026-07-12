"""Tool definitions, prompts, and async RAG client for the agentic RAG handler.

Ported from /mnt/matylda4/udupa/exps/RAG/Speech_Research_RAG/rag/pipeline.py
(TOOLS, SYSTEM_PROMPT, SPOKEN_NOTE_PROMPT, execute_tool_call), adapted from sync
httpx/ThreadPoolExecutor to async urllib-in-thread (matching the pattern already
used by RagRetriever in unmute_handler_speculative_rag.py, so we don't add a new
HTTP client dependency).
"""

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any

from unmute.kyutai_constants import SPEECH_RAG_API_KEY, SPEECH_RAG_SERVER

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI tool-calling format, ready for VLLMStream)
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_metadata",
            "description": (
                "Search papers by title, author, venue, year, or session. "
                "Use for navigational queries: finding papers by a specific author, "
                "papers from a specific conference year, or papers in a named session."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "freshness": {"type": "number", "description": "0.0 = historical/foundational query, prefer older works; 1.0 = cutting-edge query, prefer recent works. Default 0.5."},
                    "paper_impact": {"type": "number", "description": "0.0 = broad coverage, any paper works; 1.0 = need landmark/seminal/highly-cited papers or key research directions. Default 0.5."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_abstract",
            "description": (
                "Search papers by topic, method, or research area using abstract content. "
                "Use for semantic queries: finding papers about a research topic, "
                "a specific method, or a technical approach."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "freshness": {"type": "number", "description": "0.0 = historical/foundational query, prefer older works; 1.0 = cutting-edge query, prefer recent works. Default 0.5."},
                    "paper_impact": {"type": "number", "description": "0.0 = broad coverage, any paper works; 1.0 = need landmark/seminal/highly-cited papers or key research directions. Default 0.5."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_session",
            "description": (
                "Search conference sessions — topic clusters at a specific venue and year. "
                "Use for conference-level queries: what sessions/topics appeared at a conference, "
                "how themes evolved across years, or what a workshop focuses on."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "freshness": {"type": "number", "description": "0.0 = historical/foundational query, prefer older works; 1.0 = cutting-edge query, prefer recent works. Default 0.5."},
                    "paper_impact": {"type": "number", "description": "0.0 = broad coverage, any paper works; 1.0 = need landmark/seminal/highly-cited papers or key research directions. Default 0.5."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_author",
            "description": (
                "Find publications by a specific researcher, matched by name. "
                "Use this when the query asks about a named researcher's work or body of publications. "
                "Handles typos, hyphens, and speech-recognition errors — fuzzy matching is built in, so call it ONCE with your best guess at the name, never call it multiple times with spelling variants."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Researcher's full name (first and last preferred). Approximate spellings accepted."},
                    "freshness": {"type": "number", "description": "0.0 = historical/foundational query, prefer older works; 1.0 = cutting-edge query, prefer recent works. Default 0.5."},
                    "paper_impact": {"type": "number", "description": "0.0 = broad coverage, any paper works; 1.0 = need landmark/seminal/highly-cited papers or key research directions. Default 0.5."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ranked_papers_by_author",
            "description": (
                "Return a researcher's papers ranked by citation count and recency, with citation counts visible. "
                "Use this when the query asks for the most important, most cited, or most influential works by a specific researcher, "
                "or when comparing the impact of their contributions. "
                "Handles name typos and speech-recognition errors — call ONCE with your best guess at the name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Researcher's full name. Approximate spellings accepted."},
                    "freshness": {"type": "number", "description": "0.0 = prefer older landmark works; 1.0 = prefer recent high-impact works. Default 0.5."},
                    "paper_impact": {"type": "number", "description": "How strongly to weight citation count in ranking. 1.0 = rank primarily by citations. Default 0.9 for this tool."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_author_topics",
            "description": (
                "Find a specific researcher's papers in a particular topic area. "
                "Use this instead of search_abstract when you want one author's work on a specific topic "
                "(e.g. 'Emmanuel Vincent's speech enhancement papers', 'what has Shinji Watanabe done on ASR'). "
                "Returns only that author's papers filtered to the topic — avoids mixing in unrelated papers. "
                "Handles name typos — call ONCE with your best guess."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "author_name": {"type": "string", "description": "Researcher's full name. Approximate spellings accepted."},
                    "topic": {"type": "string", "description": "Topic or research area (e.g. 'speech enhancement', 'self-supervised learning', 'speaker diarization')."},
                },
                "required": ["author_name", "topic"],
            },
        },
    },
]

TOOL_NAMES: frozenset[str] = frozenset(t["function"]["name"] for t in TOOLS)

SYSTEM_PROMPT = """You are a speech research expert embedded in a spoken dialogue system. Use the search tools to retrieve relevant papers before answering.

ALWAYS retrieve before answering ANY question about research — topics, methods, findings, papers, authors, venues, or trends — INCLUDING follow-up questions that build on what was discussed earlier. Even when earlier turns in this conversation already covered related material, issue FRESH tool calls for the new question instead of answering from memory or from prior context. Do not rely on your own background knowledge for factual claims; ground them in retrieved papers. Only skip retrieval for pure greetings or acknowledgments (e.g. "thanks", "got it", "hello").

CONVERSATIONAL / META TURNS: not every turn is a research question. If the user is just greeting you ("hi", "hello"), acknowledging ("thanks", "got it", "cool"), making small talk ("how are you"), or asking a meta/system question about the conversation itself ("can you hear me?", "is the transcription working?", "are you there?", "who are you?", "what can you do?"), then do NOT call any tool and do NOT mention papers, sessions, authors, venues, or research findings at all. Just reply briefly and naturally, like a normal conversation partner would, and stop. Only genuine questions about speech research get retrieval — never pad a conversational reply with unsolicited research content.

ASR ROBUSTNESS: the question reaches you through speech recognition and often contains transcription errors, especially in technical terms and researcher names. Silently correct the obvious mis-hearing to the nearest plausible SPEECH-RESEARCH term or name and search for THAT — never search the garbled literal string, and never ask the user to repeat or spell it out. Examples: "speed recognition" -> speech recognition; "confermer"/"conformist" -> conformer; "wave to veck"/"wave2vec" -> wav2vec; "you bert"/"hubert" -> HuBERT; "dye arization" -> diarization; "shinji watanabe" variants -> Shinji Watanabe. When a name or term is genuinely ambiguous, pick the single most likely speech-research interpretation and proceed — the author-search tools already do fuzzy matching, so give them your best corrected guess.

Tool selection:
- search_abstract: topic, method, or research area queries
- search_metadata: author, paper title, venue, or year queries
- search_session: conference theme or session-level queries
- search_by_author: broad author publication lookup; handles typos and hyphens — fuzzy matching built in, call ONCE only
- get_ranked_papers_by_author: use when the query asks for most important/cited/influential works by an author; returns papers ranked by citation count with citation numbers shown
- search_author_topics: use when you need one specific author's work in a particular topic area (e.g. "Emmanuel Vincent's work on speech enhancement"); more precise than search_abstract for author+topic queries

Query decomposition: Break the question into DISTINCT sub-queries that together cover it, and issue up to 5 tool calls in parallel — mixing tools to maximise coverage. Each call MUST use a DIFFERENT, specific query; never fire two calls with the same umbrella term (e.g. do NOT search "speech recognition" twice). Turn a bare topic into concrete named facets.

Broad / survey questions especially — "what's been happening", "trends", "overview", "landmark or seminal papers", "past N years", "state of the field" — must be decomposed into 3-5 concrete sub-directions rather than the umbrella term. For example, for speech recognition over the last decade: end-to-end CTC/attention models, transformer and conformer architectures, self-supervised pretraining (wav2vec, HuBERT), streaming and RNN-T, and multilingual/low-resource ASR — search those, not "speech recognition". Use search_abstract for such topics (NOT search_metadata, which is only for a KNOWN author, title, venue, or year), and add a search_session call for trend/theme-level coverage.

Per-tool scores (set on EVERY call — INDEPENDENT axes; read them off the question's wording):
- freshness: recency preference. Push toward 1.0 for "recent", "lately", "these days", "past N years", "current", "trends", "latest", "state of the art"; toward 0.0 for "foundational", "classic", "early", "origins", "seminal". Default 0.5.
- paper_impact: citation-impact preference. Push toward 1.0 for "landmark", "seminal", "influential", "important", "most-cited", "key", "biggest breakthroughs"; 0.0 when citation count is irrelevant. Default 0.5. (For a "landmark works" question, prefer get_ranked_papers_by_author when an author is named.)

Multi-turn search: You may search multiple times before answering. After seeing results, if key information is still missing, issue additional targeted searches. Stop searching when you have enough to answer confidently.

This is a live SPOKEN conversation, so talk like a sharp, friendly expert chatting with a colleague — warm, natural, a little playful. Contractions and a bit of personality are great ("oh nice, great question", "turns out", "the fun part is", "honestly"); stiff, robotic list-reading is not. The user already heard a quick spoken acknowledgment that you're looking things up, so don't reintroduce yourself or say "let me check" again — just dive straight into the answer.

LENGTH — keep it conversational and easy to follow by ear. For most questions one or two sentences is plenty: lead with the most relevant point and don't reel off a long list of papers or methods a listener can't absorb aloud. It's fine to give a bit more when the question genuinely calls for it or the user asks for detail ("tell me more", "the full picture", a deeper dive) — just match the length to the question and finish your thought naturally rather than trailing off.

Ground everything in the retrieved papers and name the actual findings or researchers rather than referring to the database, conferences, sessions, or venues. If the results don't answer the question, just say so plainly and briefly.

Your reply is READ ALOUD by a text-to-speech voice, so write ONLY spoken words and normal sentence punctuation. Absolutely NO emojis, no emoticons, no symbols or special characters, no markdown, asterisks, bullet points, or headers, and no control/status tokens (e.g. never output things like "SEARCH_COMPLETE"). Anything that isn't a spoken word will be mispronounced or read out literally, so leave it out entirely."""

# Appended to SYSTEM_PROMPT on the turn-1 tool-decision call for a real research
# question. We run tool_choice="auto" (never "required" -- guided decoding stalls
# hard on this Qwen build, dead-airing the turn for many seconds; see debug log).
# "auto" grounds fresh questions fine, but on FOLLOW-UPS the long conversation
# history biases the model to answer from context without searching -- this
# directive forces a fresh tool call at the PROMPT level instead, which does not
# trip the guided-decoding stall.
FORCE_RETRIEVAL_DIRECTIVE = """

FOR THIS TURN SPECIFICALLY: the user just asked a question about speech research, so you MUST begin by issuing one or more search tool calls RIGHT NOW and ground your answer only in what they return. Do NOT answer from earlier turns in this conversation or from your own memory -- even if earlier turns already covered related material, treat this as a fresh question that needs fresh retrieval. If the question refers back to something earlier ("that", "those", "similar", "what about X", "how about Y"), resolve the reference to the concrete new topic and search for THAT. Only skip the tool call if this is a pure greeting or acknowledgment with no research content at all."""

SPOKEN_NOTE_PROMPT = """You are a speech research assistant in a live spoken conversation. Given a user question and what is being searched, write ONE short conversational sentence (max 20 words) to say aloud BEFORE the search results come back. Signal you are looking it up — do NOT answer the question, give facts, or state conclusions.

If earlier acknowledgments were already said in this same turn (given below), do NOT repeat their phrasing — continue naturally, e.g. "let me also check..." instead of "let me check..." again.

This is read aloud by a TTS voice: no emojis, no symbols, no markdown, no quotes, no em-dashes or other special punctuation -- only spoken words with commas or periods.

Examples (first acknowledgment in a turn):
- "Oh, Hung-yi Lee? Let me see what topics he has worked on."
- "Shinji Watanabe on end-to-end ASR, let me check that."
- "Speech enhancement in noisy environments, give me a second to look that up."

Examples (continuing after an earlier acknowledgment):
- "Let me also check his more recent papers on that."
- "I'll dig a bit deeper into the session-level trends too."
"""


def _continuation_prompt(spoken_notes: list[str]) -> str:
    """System-prompt override for the final answer, ported from the reference
    pipeline (Speech_Research_RAG/rag/pipeline.py:_continuation_prompt).

    The reference generates the answer as a NORMAL completion (default
    add_generation_prompt) with this override telling the model it already said
    the note(s) aloud and to continue naturally -- NOT via
    continue_final_message with the note as a literal assistant prefix (which
    makes the model treat the assistant turn as already finished and emit
    nothing). Generalized here to accept multiple notes said this turn.
    """
    if not spoken_notes:
        return SYSTEM_PROMPT
    if len(spoken_notes) == 1:
        said = f'You already said aloud: "{spoken_notes[0]}".'
    else:
        joined = " ".join(f'"{n}"' for n in spoken_notes)
        said = f"You already said aloud, in order: {joined}."
    return (
        SYSTEM_PROMPT
        + f"\n\n{said} "
        "Now give the answer based on the retrieved papers, as a natural, "
        "warm continuation of what you just said. "
        "Do NOT re-introduce yourself, do NOT repeat the spoken note(s), "
        "do NOT say 'let me' or 'I found'. "
        "Keep it conversational and reasonably brief — a sentence or two is "
        "usually plenty, a bit more if the question calls for it."
    )


def _spoken_note_user_msg(
    query: str,
    tool_calls: list[dict[str, Any]],
    prior_notes: list[str],
    brief: bool = False,
) -> str:
    lines = [f"Question: {query}"]
    if prior_notes:
        lines.append("Already said aloud so far, in order:")
        for i, note in enumerate(prior_notes, 1):
            lines.append(f'  {i}. "{note}"')
    if brief:
        # A speculative opener already played, so this note only bridges the last
        # bit of retrieval before the answer. Keep it to a short half-sentence so
        # the grounded answer arrives sooner.
        lines.append(
            "You ALREADY acknowledged the question aloud, so this is just a tiny "
            "bridge before the answer. Write a VERY short continuation, under 8 "
            "words. Match whatever the question is actually about, and do NOT "
            'assume it is about a person (no "his"/"her" unless a specific '
            'researcher was named). Neutral examples: "let me dig a little '
            'deeper here." / "give me one sec on this." / "let me pull the key '
            'work." Do not restate the topic in full.'
        )
    lines.append("Searching:")
    for tc in tool_calls:
        name = tc["name"]
        args = tc["arguments"]
        if name in ("search_by_author", "get_ranked_papers_by_author"):
            lines.append(f"  - author lookup: {args.get('name', '')}")
        elif name == "search_author_topics":
            lines.append(f"  - author: {args.get('author_name', '')}, topic: {args.get('topic', '')}")
        elif name in ("search_abstract", "search_metadata", "search_session"):
            lines.append(f"  - {name.replace('search_', '')}: {args.get('query', '')}")
        else:
            lines.append(f"  - {name}")
    return "\n".join(lines)


# Speculative note: generated PRE-VAD from only the (possibly partial) user
# transcript -- no tool_calls exist yet. It's a filler said the instant the user
# stops, so it must be short, robust to a slightly-incomplete question, and never
# commit to a fact (retrieval hasn't happened). See
# docs/agentic_speculative_note_design.md.
SPECULATIVE_NOTE_PROMPT = """You are a speech research assistant in a live spoken conversation, and the user is finishing a question. Write ONE short, natural sentence (max 14 words) to say aloud IMMEDIATELY that (a) briefly NAMES the topic or area they're asking about, and (b) signals you're looking it up. Reflecting the topic back makes it feel like you actually listened.

You have NOT searched yet, so do NOT answer, do NOT state any fact, finding, specific paper name, number, or conclusion, and do NOT promise a particular result. Just mirror the topic and say you'll check.

If the transcript is too vague or cut off to tell what the topic is, stay generic instead of guessing.

If the transcript is clearly NOT a research question -- a greeting, small talk, or a meta/system question like "can you hear me?" or "is this working?" -- do NOT talk about searching, looking things up, or papers. Just give a short, warm conversational filler ("Yeah, one sec.", "Sure thing.", "Mhm, go on."). Never say you are checking a paper or searching for something that isn't a research topic.

The transcript comes from speech recognition and may contain typos, especially in technical terms and names. When you reflect the topic back, say the CORRECT speech-research term, not the garbled one (e.g. transcript "speed recognition" -> say "speech recognition"; "confermer" -> "conformer"). Never repeat an obviously mis-heard word.

This is read aloud by a TTS voice: no emojis, no symbols, no markdown, no quotes, no em-dashes -- only spoken words with commas or periods.

Examples (topic is clear -> name it):
- "Multi-speaker and far-field ASR, let me look into that."
- "Recent work on speech enhancement, give me a second."
- "Self-supervised speech models, let me pull that up."
- "Shinji Watanabe's work, let me check what he has done."
Examples (topic unclear or cut off -> stay generic):
- "Sure, let me look into that for you."
- "Good question, give me a second to check."
"""


def _speculative_note_user_msg(transcript: str) -> str:
    """User message for the pre-VAD speculative note: just the (partial)
    transcript, since no tool calls have been decided yet."""
    t = (transcript or "").strip()
    if not t:
        return "The user is starting to speak. Say a brief, warm filler while you wait."
    return (
        "The user is asking this (live transcript, may be slightly incomplete). "
        "Identify the topic/area and reflect it back briefly, then say you'll look "
        f"it up:\n{t}"
    )


# ---------------------------------------------------------------------------
# Async RAG tool executor
# ---------------------------------------------------------------------------


class AgenticRagError(Exception):
    pass


async def _post_json(
    url: str, payload: dict[str, Any], api_key: str | None, timeout_sec: float
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    def _call_once() -> dict[str, Any]:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8")
        decoded = json.loads(body)
        assert isinstance(decoded, dict)
        return decoded

    return await asyncio.to_thread(_call_once)


class AgenticRagClient:
    """Async client for the Speech_Research_RAG server's tool-call endpoints."""

    def __init__(
        self,
        rag_url: str = SPEECH_RAG_SERVER,
        api_key: str | None = SPEECH_RAG_API_KEY,
        top_k: int = 3,
        timeout_sec: float = 10.0,
    ):
        self.rag_url = rag_url.rstrip("/")
        self.api_key = api_key
        self.top_k = top_k
        self.timeout_sec = timeout_sec

    async def execute_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Call the RAG server for one tool call, return tool_call + result.

        Mirrors Speech_Research_RAG/rag/pipeline.py's execute_tool_call, ported to
        async. Never raises -- errors are captured in the result dict so a single
        failing tool call doesn't take down the whole agentic turn.
        """
        name = tool_call["name"]
        args = tool_call["arguments"]

        if name in ("search_by_author", "get_ranked_papers_by_author"):
            return await self._search_by_author(tool_call, name, args)
        if name == "search_author_topics":
            return await self._search_author_topics(tool_call, args)

        collection_map = {
            "search_metadata": "metadata",
            "search_abstract": "abstract",
            "search_session": "session",
        }
        collection = collection_map.get(name)
        if not collection:
            return {**tool_call, "result": {"error": f"Unknown tool: {name}"}}

        query = args.get("query", "")
        k = args.get("top_k", self.top_k)
        freshness = float(args.get("freshness", 0.5))
        paper_impact = float(args.get("paper_impact", 0.5))

        try:
            data = await _post_json(
                f"{self.rag_url}/search/{collection}",
                {"query": query, "top_k": k, "freshness": freshness, "paper_impact": paper_impact},
                self.api_key,
                self.timeout_sec,
            )
            results = [r["payload"] for r in data["results"]]
            return {**tool_call, "result": {"results": results}}
        except Exception as e:
            return {**tool_call, "result": {"error": repr(e), "results": []}}

    async def _search_by_author(
        self, tool_call: dict[str, Any], name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        author_name = (args.get("name") or args.get("query") or "").strip()
        default_impact = 0.9 if name == "get_ranked_papers_by_author" else 0.5
        freshness = float(args.get("freshness", 0.5))
        paper_impact = float(args.get("paper_impact", default_impact))
        try:
            data = await _post_json(
                f"{self.rag_url}/search/author",
                {
                    "name": author_name,
                    "top_k": self.top_k,
                    "freshness": freshness,
                    "paper_impact": paper_impact,
                },
                self.api_key,
                self.timeout_sec,
            )
            result: dict[str, Any] = {
                "matched_author": data.get("matched_author"),
                "paper_count": data.get("paper_count", 0),
                "results": data.get("results", []),
            }
            alts = data.get("alternatives", [])
            if alts:
                result["alternatives"] = [a["name"] for a in alts]
            if data.get("matched_author") is None:
                result["message"] = (
                    f"No author found matching '{author_name}'. "
                    "Try providing the full name or a different spelling."
                )
        except Exception as e:
            result = {"error": repr(e), "results": []}
        return {**tool_call, "result": result}

    async def _search_author_topics(
        self, tool_call: dict[str, Any], args: dict[str, Any]
    ) -> dict[str, Any]:
        author_name = args.get("author_name", "").strip()
        topic = args.get("topic", "").strip()
        query = f"{author_name} | {topic}"
        try:
            data = await _post_json(
                f"{self.rag_url}/search/author_topics",
                {"query": query, "top_k": min(self.top_k, 3)},
                self.api_key,
                self.timeout_sec,
            )
            seen_titles: set[str] = set()
            papers: list[dict[str, Any]] = []
            matched_author = None
            for r in data.get("results", []):
                p = r["payload"]
                if matched_author is None:
                    matched_author = p.get("author")
                for paper in p.get("papers", []):
                    t = (paper.get("title") or "").lower()
                    if t and t not in seen_titles:
                        seen_titles.add(t)
                        papers.append(
                            {
                                "title": paper.get("title", ""),
                                "authors": matched_author or author_name,
                                "year": paper.get("year"),
                                "citation_count": paper.get("citation_count"),
                            }
                        )
            result: dict[str, Any] = {
                "matched_author": matched_author,
                "topic": topic,
                "results": papers,
            }
            if not papers:
                result["message"] = (
                    f"No papers found for '{author_name}' on topic '{topic}'. "
                    "Try search_by_author for their full publication list."
                )
        except Exception as e:
            result = {"error": repr(e), "results": []}
        return {**tool_call, "result": result}

    async def health(self) -> bool:
        try:
            def _call_once() -> dict[str, Any]:
                req = urllib.request.Request(f"{self.rag_url}/health", method="GET")
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    return json.loads(resp.read().decode("utf-8"))

            data = await asyncio.to_thread(_call_once)
            return data.get("status") == "ok"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return False
