#!/usr/bin/env python3
"""
RAG pipeline — LLM query decomposition → RAG retrieval → grounded response.

Supports two LLM backends:
  --llm claude   : Anthropic Claude API
  --llm vllm     : vLLM OpenAI-compatible server (e.g. Gemma 3 4B)

Single-round tool use:
  1. LLM call 1 : query + tool definitions → tool calls
  2. Parallel   : execute tool calls against RAG server
  3. LLM call 2 : query + tool results → final response

Usage:
  python rag/pipeline.py --query "self supervised learning for speech"
  python rag/pipeline.py --query "..." --llm claude
  python rag/pipeline.py --query "..." --llm vllm --vllm-url http://localhost:8001/v1 --vllm-model google/gemma-3-4b-it
  python rag/pipeline.py --query "..." --rag-url http://sge-node:8000
"""

import argparse
import concurrent.futures
import json
import time
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.auth import load_api_key

# ---------------------------------------------------------------------------
# Tool definitions (collection-agnostic internal format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "search_metadata",
        "description": (
            "Search papers by title, author, venue, year, or session. "
            "Use for navigational queries: finding papers by a specific author, "
            "papers from a specific conference year, or papers in a named session."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query":        {"type": "string", "description": "Search query"},
                "freshness":    {"type": "number", "description": "0.0 = historical/foundational query, prefer older works; 1.0 = cutting-edge query, prefer recent works. Default 0.5."},
                "paper_impact": {"type": "number", "description": "0.0 = broad coverage, any paper works; 1.0 = need landmark/seminal/highly-cited papers or key research directions. Default 0.5."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_abstract",
        "description": (
            "Search papers by topic, method, or research area using abstract content. "
            "Use for semantic queries: finding papers about a research topic, "
            "a specific method, or a technical approach."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query":        {"type": "string", "description": "Search query"},
                "freshness":    {"type": "number", "description": "0.0 = historical/foundational query, prefer older works; 1.0 = cutting-edge query, prefer recent works. Default 0.5."},
                "paper_impact": {"type": "number", "description": "0.0 = broad coverage, any paper works; 1.0 = need landmark/seminal/highly-cited papers or key research directions. Default 0.5."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_session",
        "description": (
            "Search conference sessions — topic clusters at a specific venue and year. "
            "Use for conference-level queries: what sessions/topics appeared at a conference, "
            "how themes evolved across years, or what a workshop focuses on."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query":        {"type": "string", "description": "Search query"},
                "freshness":    {"type": "number", "description": "0.0 = historical/foundational query, prefer older works; 1.0 = cutting-edge query, prefer recent works. Default 0.5."},
                "paper_impact": {"type": "number", "description": "0.0 = broad coverage, any paper works; 1.0 = need landmark/seminal/highly-cited papers or key research directions. Default 0.5."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_by_author",
        "description": (
            "Find publications by a specific researcher, matched by name. "
            "Use this when the query asks about a named researcher's work or body of publications. "
            "Handles typos, hyphens, and speech-recognition errors — fuzzy matching is built in, so call it ONCE with your best guess at the name, never call it multiple times with spelling variants."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name":         {"type": "string", "description": "Researcher's full name (first and last preferred). Approximate spellings accepted."},
                "freshness":    {"type": "number", "description": "0.0 = historical/foundational query, prefer older works; 1.0 = cutting-edge query, prefer recent works. Default 0.5."},
                "paper_impact": {"type": "number", "description": "0.0 = broad coverage, any paper works; 1.0 = need landmark/seminal/highly-cited papers or key research directions. Default 0.5."},
            },
            "required": ["name"],
        },
    },
    {
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
                "name":         {"type": "string", "description": "Researcher's full name. Approximate spellings accepted."},
                "freshness":    {"type": "number", "description": "0.0 = prefer older landmark works; 1.0 = prefer recent high-impact works. Default 0.5."},
                "paper_impact": {"type": "number", "description": "How strongly to weight citation count in ranking. 1.0 = rank primarily by citations. Default 0.9 for this tool."},
            },
            "required": ["name"],
        },
    },
    {
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
                "topic":       {"type": "string", "description": "Topic or research area (e.g. 'speech enhancement', 'self-supervised learning', 'speaker diarization')."},
            },
            "required": ["author_name", "topic"],
        },
    },
]



SYSTEM_PROMPT = """You are a speech research expert. Use the search tools to retrieve relevant papers before answering.

Tool selection:
- search_abstract: topic, method, or research area queries
- search_metadata: author, paper title, venue, or year queries
- search_session: conference theme or session-level queries
- search_by_author: broad author publication lookup; handles typos and hyphens — fuzzy matching built in, call ONCE only
- get_ranked_papers_by_author: use when the query asks for most important/cited/influential works by an author; returns papers ranked by citation count with citation numbers shown
- search_author_topics: use when you need one specific author's work in a particular topic area (e.g. "Emmanuel Vincent's work on speech enhancement"); more precise than search_abstract for author+topic queries

Query decomposition: Before calling tools, decompose the question into distinct sub-queries that together cover the full question. Issue up to 5 tool calls in parallel — mix tools to maximise coverage. Examples:
- A broad or evolution question → search_abstract calls per sub-topic/era + search_session for theme-level coverage
- A question spanning multiple methods → one search_abstract call per method
- A question mixing topic and author → search_abstract for topic + search_metadata for author
- A simple factoid → a single targeted call is fine

Per-tool scores (set these on every tool call — they are INDEPENDENT axes):
- freshness: controls recency preference only. 0.0 = want older/foundational works ("early HMM", "origins of CTC"). 1.0 = want recent works ("latest in 2024", "what has X published recently"). Default 0.5. Does NOT affect citation ranking.
- paper_impact: controls citation-count preference only. 1.0 = user explicitly wants high-impact/seminal/most-cited papers ("most important", "key contributions", "influential works"). 0.0 = citation count irrelevant, return any matching papers. Default 0.5. Does NOT affect recency ranking.
  Examples: "latest papers by X" → freshness=0.9, paper_impact=0.5. "most cited works by X" → freshness=0.5, paper_impact=1.0. "recent influential SSL papers" → freshness=0.8, paper_impact=0.8.

Before issuing tool calls, briefly state your reasoning: why you chose these tools, and what freshness/paper_impact values make sense for this query.

Multi-turn search: You may search multiple times before answering. After seeing results, if key information is still missing, issue additional targeted searches. Stop searching when you have enough to answer confidently — do not search more than 4 times total.

Common patterns:
- Author + topic ("what has X worked on in Y"): use search_author_topics(author_name, topic) directly — it returns that author's papers pre-filtered to the topic. Do NOT use search_abstract for author+topic queries.
- Named paper: search_abstract first; if the target paper is not in the results, follow up with search_metadata using the exact title.
- Topic synthesis: search_abstract for the main topic; if key sub-areas are missing, search again with a more specific sub-query.
- Temporal evolution: issue separate searches per era across turns (e.g. "topic 2015-2018", "topic 2019-2022", "topic 2023-present") to build a chronological picture.

Answer as a domain expert grounded in the retrieved papers. Do not reference the database, conferences, sessions, or venues — cite findings and authors instead. If a question cannot be answered from the retrieved results, say so. Adapt response length to question complexity: 1-2 sentences for simple factoids, up to 5 sentences for broad or multi-part technical questions. Use plain prose only — no bullet points, bold, italics, headers, or emoji."""


# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

def _continuation_prompt(spoken_note: str) -> str:
    """System prompt override that tells the LLM to continue from the spoken note."""
    return (
        SYSTEM_PROMPT
        + f'\n\nYou already said aloud: "{spoken_note}". '
        "Now give the answer based on the retrieved papers. "
        "Do NOT re-introduce yourself, do NOT repeat the spoken note, do NOT say 'let me' or 'I found'. "
        "Jump straight into the answer as a natural continuation."
    )


def _spoken_note_user_msg(query: str, tool_calls: list[dict]) -> str:
    """Build a structured user message for spoken note generation."""
    lines = [f"Question: {query}", "Searching:"]
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


SPOKEN_NOTE_PROMPT = """You are a speech research assistant. Given a user question and what is being searched, write ONE short conversational sentence (max 20 words) to say aloud BEFORE the search results come back. Signal you are looking it up — do NOT answer the question, give facts, or state conclusions. For author queries: acknowledge the person. For topic queries: acknowledge the topic and say you are looking it up. No markdown, no quotes.

Examples:
- "Oh, Hung-yi Lee? Let me see what topics he has worked on."
- "Shinji Watanabe on end-to-end ASR — let me check that."
- "Karen Livescu — let me pull up her papers."
- "Speech enhancement in noisy environments — give me a second to look that up."
- "Conformer versus Transformer — interesting question, let me look that up."
- "Self-supervised learning for speech — let me dig into that.\""""


class LLMClient(ABC):
    @abstractmethod
    def call_with_tools(self, query: str) -> tuple[list[dict], dict]:
        """First call: returns (tool_calls, timings).
        timings keys: total, prefill, decode, prompt_tokens, completion_tokens."""

    @abstractmethod
    def call_with_results(self, query: str, tool_results: list[dict], system_override: str | None = None) -> tuple[str, dict]:
        """Second call: returns (response_text, timings)."""

    @abstractmethod
    def call_agentic_turn(self, query: str, history: list[dict],
                          system_override: str | None = None) -> tuple[list[dict], str, dict]:
        """Agentic loop turn. history is the accumulated message list from prior turns.
        Returns (tool_calls, response_text, timings).
        If tool_calls is non-empty, the LLM wants to search more.
        If tool_calls is empty, response_text is the final answer."""

    @abstractmethod
    def generate_spoken_note(self, query: str, tool_calls: list[dict]) -> str:
        """Fast call to generate a short conversational spoken note for TTS.
        Fired in parallel with retrieval — must be cheap (max_tokens=60)."""


class ClaudeClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model  = model
        self.tools  = [
            {
                "name":         t["name"],
                "description":  t["description"],
                "input_schema": t["parameters"],
            }
            for t in TOOLS
        ]

    def _history_to_messages(self, query: str, history: list[dict]) -> list[dict]:
        """Convert neutral history to Claude message format."""
        messages = [{"role": "user", "content": query}]
        for turn in history:
            # assistant message: tool calls + optional reasoning text
            assistant_content = []
            if turn.get("reasoning"):
                assistant_content.append({"type": "text", "text": turn["reasoning"]})
            for tc in turn["tool_calls"]:
                assistant_content.append({
                    "type": "tool_use", "id": tc["id"],
                    "name": tc["name"], "input": tc["arguments"],
                })
            messages.append({"role": "assistant", "content": assistant_content})
            # user message: tool results
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tr["id"],
                 "content": json.dumps(tr["result"])}
                for tr in turn["tool_results"]
            ]})
        return messages

    def call_with_tools(self, query: str) -> tuple[list[dict], dict]:
        t0 = time.time()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=self.tools,
            messages=[{"role": "user", "content": query}],
        )
        total = time.time() - t0
        calls = []
        reasoning_parts = []
        for block in resp.content:
            if block.type == "tool_use":
                calls.append({"name": block.name, "arguments": block.input, "id": block.id})
            elif block.type == "text" and block.text.strip():
                reasoning_parts.append(block.text.strip())
        timings = {
            "total": total, "prefill": None, "decode": None,
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
            "reasoning": " ".join(reasoning_parts),
        }
        return calls, timings

    def call_agentic_turn(self, query: str, history: list[dict],
                          system_override: str | None = None) -> tuple[list[dict], str, dict]:
        t0 = time.time()
        messages = self._history_to_messages(query, history)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system_override or SYSTEM_PROMPT,
            tools=self.tools,
            messages=messages,
        )
        total = time.time() - t0
        calls, reasoning_parts, text_parts = [], [], []
        for block in resp.content:
            if block.type == "tool_use":
                calls.append({"name": block.name, "arguments": block.input, "id": block.id})
            elif block.type == "text" and block.text.strip():
                if calls:
                    reasoning_parts.append(block.text.strip())
                else:
                    text_parts.append(block.text.strip())
        timings = {
            "total": total, "prefill": None, "decode": None,
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
            "reasoning": " ".join(reasoning_parts),
        }
        return calls, " ".join(text_parts), timings

    def call_with_results(self, query: str, tool_results: list[dict], system_override: str | None = None) -> tuple[str, dict]:
        system = system_override or SYSTEM_PROMPT
        if tool_results:
            assistant_content = [
                {"type": "tool_use", "id": r["id"], "name": r["name"], "input": r["arguments"]}
                for r in tool_results
            ]
            user_results = [
                {"type": "tool_result", "tool_use_id": r["id"], "content": json.dumps(r["result"])}
                for r in tool_results
            ]
            messages = [
                {"role": "user",      "content": query},
                {"role": "assistant", "content": assistant_content},
                {"role": "user",      "content": user_results},
            ]
        else:
            messages = [{"role": "user", "content": query}]
        t0 = time.time()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            tools=self.tools if tool_results else [],
            messages=messages,
        )
        total = time.time() - t0
        timings = {
            "total": total, "prefill": None, "decode": None,
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
        }
        return resp.content[0].text, timings

    def generate_spoken_note(self, query: str, tool_calls: list[dict]) -> str:
        user_msg = _spoken_note_user_msg(query, tool_calls)
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=60,
                system=SPOKEN_NOTE_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            return resp.content[0].text.strip().strip('"')
        except Exception:
            return ""


class VLLMClient(LLMClient):
    def __init__(self, base_url: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="vllm")
        self.model  = model
        self.tools  = [
            {
                "type": "function",
                "function": {
                    "name":        t["name"],
                    "description": t["description"],
                    "parameters":  t["parameters"],
                },
            }
            for t in TOOLS
        ]

    def call_with_tools(self, query: str) -> tuple[list[dict], dict]:
        t0 = time.time()
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": query},
            ],
            tools=self.tools,
            tool_choice="auto",
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        ttft = None
        # accumulate streamed tool call deltas
        tc_chunks: dict[int, dict] = {}  # index → accumulated call
        text_chunks: list[str] = []
        usage = None
        for chunk in stream:
            if ttft is None and chunk.choices and chunk.choices[0].delta.content is not None:
                ttft = time.time() - t0
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    text_chunks.append(delta.content)
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tc_chunks:
                            tc_chunks[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc_delta.id:
                            tc_chunks[idx]["id"] += tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tc_chunks[idx]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                tc_chunks[idx]["arguments"] += tc_delta.function.arguments
            if hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage
        total = time.time() - t0
        if ttft is None:
            ttft = total
        calls = []
        for idx in sorted(tc_chunks):
            tc = tc_chunks[idx]
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            calls.append({"name": tc["name"], "arguments": args, "id": tc["id"]})
        import re as _re
        raw_text = "".join(text_chunks).strip()
        reasoning = _re.sub(r"<tool_call>.*?</tool_call>", "", raw_text, flags=_re.DOTALL).strip()
        timings = {
            "total":             total,
            "prefill":           ttft,
            "decode":            total - ttft,
            "prompt_tokens":     usage.prompt_tokens     if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "reasoning":         reasoning,
        }
        return calls, timings

    def _history_to_messages(self, query: str, history: list[dict],
                             system_override: str | None = None) -> list[dict]:
        """Convert neutral history to OpenAI message format."""
        messages = [
            {"role": "system", "content": system_override or SYSTEM_PROMPT},
            {"role": "user",   "content": query},
        ]
        for turn in history:
            messages.append({
                "role":       "assistant",
                "content":    turn.get("reasoning") or None,
                "tool_calls": [
                    {
                        "id":       tc["id"],
                        "type":     "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                    }
                    for tc in turn["tool_calls"]
                ],
            })
            for tr in turn["tool_results"]:
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tr["id"],
                    "content":      json.dumps(tr["result"]),
                })
        return messages

    def call_agentic_turn(self, query: str, history: list[dict],
                          system_override: str | None = None) -> tuple[list[dict], str, dict]:
        messages = self._history_to_messages(query, history, system_override=system_override)
        t0 = time.time()
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice="auto",
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        ttft = None
        tc_chunks: dict[int, dict] = {}
        text_chunks: list[str] = []
        usage = None
        for chunk in stream:
            if ttft is None and chunk.choices and chunk.choices[0].delta.content is not None:
                ttft = time.time() - t0
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    text_chunks.append(delta.content)
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tc_chunks:
                            tc_chunks[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc_delta.id:
                            tc_chunks[idx]["id"] += tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tc_chunks[idx]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                tc_chunks[idx]["arguments"] += tc_delta.function.arguments
            if hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage
        total = time.time() - t0
        if ttft is None:
            ttft = total
        calls = []
        for idx in sorted(tc_chunks):
            tc = tc_chunks[idx]
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            calls.append({"name": tc["name"], "arguments": args, "id": tc["id"]})
        import re as _re
        raw_text = "".join(text_chunks).strip()
        raw_text = _re.sub(r"<tool_call>.*?</tool_call>", "", raw_text, flags=_re.DOTALL).strip()
        timings = {
            "total":             total,
            "prefill":           ttft,
            "decode":            total - ttft,
            "prompt_tokens":     usage.prompt_tokens     if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "reasoning":         raw_text if calls else "",
        }
        return calls, (raw_text if not calls else ""), timings

    def call_with_results(self, query: str, tool_results: list[dict], system_override: str | None = None) -> tuple[str, dict]:
        system = system_override or SYSTEM_PROMPT
        if tool_results:
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": query},
                {
                    "role":       "assistant",
                    "content":    None,
                    "tool_calls": [
                        {
                            "id":       r["id"],
                            "type":     "function",
                            "function": {"name": r["name"], "arguments": json.dumps(r["arguments"])},
                        }
                        for r in tool_results
                    ],
                },
            ] + [
                {
                    "role":         "tool",
                    "tool_call_id": r["id"],
                    "content":      json.dumps(r["result"]),
                }
                for r in tool_results
            ]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": query},
            ]
        t0 = time.time()
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice="none",
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        ttft = None
        text_chunks = []
        usage = None
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                content = delta.content or ""
                if content:
                    if ttft is None:
                        ttft = time.time() - t0
                    text_chunks.append(content)
            if hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage
        total = time.time() - t0
        if ttft is None:
            ttft = total
        timings = {
            "total":             total,
            "prefill":           ttft,
            "decode":            total - ttft,
            "prompt_tokens":     usage.prompt_tokens     if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
        }
        text = "".join(text_chunks)
        # Qwen sometimes generates <tool_call>...</tool_call> XML even with tool_choice=none; strip it
        import re as _re
        text = _re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=_re.DOTALL).strip()
        return text, timings

    def generate_spoken_note(self, query: str, tool_calls: list[dict]) -> str:
        user_msg = _spoken_note_user_msg(query, tool_calls)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SPOKEN_NOTE_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=60,
                temperature=0.7,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return (resp.choices[0].message.content or "").strip().strip('"')
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# RAG tool executor
# ---------------------------------------------------------------------------

def execute_tool_call(tool_call: dict, rag_url: str, top_k: int, api_key: str) -> dict:
    """Call the RAG server for one tool call, return tool_call + result."""
    name = tool_call["name"]
    args = tool_call["arguments"]

    # Author fuzzy lookup — separate endpoint, different response shape
    if name in ("search_by_author", "get_ranked_papers_by_author"):
        # model sometimes uses 'query' instead of 'name' — accept both
        author_name = (args.get("name") or args.get("query") or "").strip()
        default_impact = 0.9 if name == "get_ranked_papers_by_author" else 0.5
        freshness    = float(args.get("freshness",    0.5))
        paper_impact = float(args.get("paper_impact", default_impact))
        try:
            resp = httpx.post(
                f"{rag_url}/search/author",
                json={"name": author_name, "top_k": top_k,
                      "freshness": freshness, "paper_impact": paper_impact},
                headers={"X-API-Key": api_key},
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            result: dict = {
                "matched_author": data.get("matched_author"),
                "paper_count":    data.get("paper_count", 0),
                "results":        data.get("results", []),
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
            print(f"[WARN] search_by_author failed: {e}")
            result = {"error": str(e), "results": []}
        return {**tool_call, "result": result}

    if name == "search_author_topics":
        author_name = args.get("author_name", "").strip()
        topic       = args.get("topic", "").strip()
        query       = f"{author_name} | {topic}"
        try:
            resp = httpx.post(
                f"{rag_url}/search/author_topics",
                json={"query": query, "top_k": min(top_k, 3)},
                headers={"X-API-Key": api_key},
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            # flatten papers from all returned chunks, deduplicate by title
            seen_titles: set[str] = set()
            papers = []
            matched_author = None
            for r in data.get("results", []):
                p = r["payload"]
                if matched_author is None:
                    matched_author = p.get("author")
                for paper in p.get("papers", []):
                    t = (paper.get("title") or "").lower()
                    if t and t not in seen_titles:
                        seen_titles.add(t)
                        papers.append({
                            "title":          paper.get("title", ""),
                            "authors":        matched_author or author_name,
                            "year":           paper.get("year"),
                            "citation_count": paper.get("citation_count"),
                        })
            result = {
                "matched_author": matched_author,
                "topic":          topic,
                "results":        papers,
            }
            if not papers:
                result["message"] = (
                    f"No papers found for '{author_name}' on topic '{topic}'. "
                    "Try search_by_author for their full publication list."
                )
        except Exception as e:
            print(f"[WARN] search_author_topics failed: {e}")
            result = {"error": str(e), "results": []}
        return {**tool_call, "result": result}

    collection_map = {
        "search_metadata": "metadata",
        "search_abstract":  "abstract",
        "search_session":   "session",
    }
    collection = collection_map.get(name)
    if not collection:
        return {**tool_call, "result": {"error": f"Unknown tool: {name}"}}

    query        = args.get("query", "")
    k            = args.get("top_k", top_k)
    freshness    = float(args.get("freshness",    0.5))
    paper_impact = float(args.get("paper_impact", 0.5))

    try:
        resp = httpx.post(
            f"{rag_url}/search/{collection}",
            json={"query": query, "top_k": k,
                  "freshness": freshness, "paper_impact": paper_impact},
            headers={"X-API-Key": api_key},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        # strip scores for LLM context — only payload content
        results = [r["payload"] for r in data["results"]]
        return {**tool_call, "result": {"results": results}}
    except Exception as e:
        print(f"[WARN] tool call {name} failed: {e}")
        return {**tool_call, "result": {"error": str(e), "results": []}}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    query: str,
    llm: LLMClient,
    rag_url: str,
    top_k: int,
    api_key: str,
    verbose: bool = False,
    return_details: bool = False,
    max_turns: int = 4,
) -> "str | dict":
    W = 65
    t_start = time.time()

    if verbose:
        print(f"\n{'='*W}")
        print(f"QUERY: {query}")
        print(f"{'='*W}")

    def _fmt_tokens(t: dict) -> str:
        p = t.get("prompt_tokens")
        c = t.get("completion_tokens")
        if p is None and c is None:
            return ""
        return f"  [{p}→{c} tok]"

    def _print_retrieval(tool_results: list[dict], elapsed: float, turn: int) -> None:
        print(f"\n[Turn {turn} — Retrieval  {elapsed:.2f}s]")
        for tr in tool_results:
            results  = tr["result"].get("results", [])
            args_str = ", ".join(f"{k}={v!r}" for k, v in tr["arguments"].items())
            print(f"\n  [{tr['name']}({args_str})] — {len(results)} results")
            for i, r in enumerate(results, 1):
                title    = r.get("title", r.get("session_name", ""))
                authors  = r.get("authors", "")
                year     = r.get("year", "")
                venue    = r.get("venue", "")
                abstract = r.get("abstract", "")
                cit      = r.get("citation_count")
                cit_str  = "N/A citations" if cit is None else f"{cit} citations"
                meta     = " | ".join(x for x in [venue, str(year), cit_str] if x)
                print(f"  {i}. {title}")
                if authors:
                    print(f"     {authors}")
                if meta:
                    print(f"     {meta}")
                if abstract:
                    snippet = abstract[:200].replace("\n", " ")
                    print(f"     {snippet}{'...' if len(abstract) > 200 else ''}")

    # Accumulated state across turns
    history: list[dict] = []          # neutral turn dicts passed to call_agentic_turn
    all_tool_calls: list[dict]   = []
    all_tool_results: list[dict] = []
    first_reasoning: str  = ""
    spoken_note: str      = ""
    response: str         = ""

    for turn_num in range(1, max_turns + 2):  # +1 extra slot for final answer after max search turns
        t_turn = time.time()

        # On final forced turn: LLM must answer; skip tool-call phase
        is_forced_answer = (turn_num > max_turns and all_tool_results)

        if is_forced_answer:
            if verbose:
                print(f"\n[Turn {turn_num}] max_turns={max_turns} reached — forcing final answer")
            system_override = _continuation_prompt(spoken_note) if spoken_note else None
            response, timings = llm.call_with_results(query, all_tool_results,
                                                      system_override=system_override)
            t_end = time.time()
            if verbose:
                print(f"\n[Answer  total={t_end-t_turn:.2f}s{_fmt_tokens(timings)}]")
                print(f"{'-'*W}")
            break

        # On the final answer turn (after retrieval), inject continuation prompt
        is_answer_turn = bool(history) and spoken_note
        turn_system = _continuation_prompt(spoken_note) if is_answer_turn else None
        tool_calls, response_text, timings = llm.call_agentic_turn(query, history,
                                                                    system_override=turn_system)

        if turn_num == 1:
            first_reasoning = timings.get("reasoning", "")

        if not tool_calls:
            response = response_text
            t_end = time.time()
            if verbose:
                pre = timings.get("prefill") or 0.0
                dec = timings.get("decode")  or 0.0
                print(f"\n[Turn {turn_num} — Answer  total={t_end-t_turn:.2f}s"
                      f"  prefill={pre:.2f}s  decode={dec:.2f}s{_fmt_tokens(timings)}]")
                print(f"{'-'*W}")
            break

        if verbose:
            pre = timings.get("prefill") or 0.0
            dec = timings.get("decode")  or 0.0
            print(f"\n[Turn {turn_num} — Tool selection  total={time.time()-t_turn:.2f}s"
                  f"  prefill={pre:.2f}s  decode={dec:.2f}s{_fmt_tokens(timings)}]")
            if timings.get("reasoning"):
                print(f"  Reasoning: {timings['reasoning'][:200]}")
            for tc in tool_calls:
                args_str = ", ".join(f"{k}={v!r}" for k, v in tc["arguments"].items())
                print(f"  ▸ {tc['name']}({args_str})")

        # Execute tool calls + spoken note generation in parallel
        t_rag = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tool_calls) + 1) as pool:
            note_future = (
                pool.submit(llm.generate_spoken_note, query, tool_calls)
                if turn_num == 1 else None
            )
            retrieval_futures = [
                pool.submit(execute_tool_call, tc, rag_url, top_k, api_key)
                for tc in tool_calls
            ]
            turn_results = [f.result() for f in retrieval_futures]
            if note_future is not None:
                spoken_note = note_future.result()
        t_rag_end = time.time()

        if verbose:
            if turn_num == 1 and spoken_note:
                print(f"\n  [Spoken note] \"{spoken_note}\"")
            _print_retrieval(turn_results, t_rag_end - t_rag, turn_num)

        # Append to history for next turn
        history.append({
            "reasoning":   timings.get("reasoning", ""),
            "tool_calls":  tool_calls,
            "tool_results": turn_results,
        })
        for tc in tool_calls:
            all_tool_calls.append({**tc, "turn": turn_num})
        for tr in turn_results:
            all_tool_results.append({**tr, "turn": turn_num})

    if verbose:
        print(f"\n[Latency]  total={time.time()-t_start:.2f}s  |  turns={len(history)}"
              f"  |  tool_calls={len(all_tool_calls)}")

    if return_details:
        return {
            "response":    response,
            "reasoning":   first_reasoning,
            "spoken_note": spoken_note,
            "turns":       len(history),
            "tool_calls": [
                {
                    "name":         tc["name"],
                    "arguments":    tc["arguments"],
                    "freshness":    tc["arguments"].get("freshness",    0.5),
                    "paper_impact": tc["arguments"].get("paper_impact", 0.5),
                    "turn":         tc.get("turn", 1),
                }
                for tc in all_tool_calls
            ],
            "tool_results": [
                {
                    "name":           tr["name"],
                    "result_count":   len(tr["result"].get("results", [])),
                    "matched_author": tr["result"].get("matched_author"),
                    "turn":           tr.get("turn", 1),
                    "papers": [
                        {
                            "title":          r.get("title", r.get("session_name", "")),
                            "authors":        r.get("authors", ""),
                            "year":           r.get("year", ""),
                            "venue":          r.get("venue", ""),
                            "citation_count": r.get("citation_count"),
                        }
                        for r in tr["result"].get("results", [])
                    ],
                }
                for tr in all_tool_results
            ],
        }
    return response


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

NO_RAG_SYSTEM_PROMPT = """You are a speech research expert. Answer the question from your own knowledge. Adapt response length to question complexity: 1-2 sentences for simple factoids, up to 5 sentences for broad or multi-part technical questions. Use plain prose only — no bullet points, bold, italics, headers, or emoji."""


def run_no_rag(query: str, llm: LLMClient, verbose: bool = False) -> str:
    W = 65
    t0 = time.time()
    response, timings = llm.call_with_results(query, [], system_override=NO_RAG_SYSTEM_PROMPT)
    t1 = time.time()
    if verbose:
        pre = timings.get("prefill") or 0.0
        dec = timings.get("decode")  or 0.0
        p   = timings.get("prompt_tokens")
        c   = timings.get("completion_tokens")
        tok = f"  [{p}→{c} tok]" if p is not None else ""
        print(f"\n[No-RAG] total={t1-t0:.2f}s  prefill={pre:.2f}s  decode={dec:.2f}s{tok}")
        print(f"{'-'*W}")
    return response


def main():
    parser = argparse.ArgumentParser(description="RAG pipeline.")
    parser.add_argument("--query",       required=True)
    parser.add_argument("--llm",         default="vllm", choices=["claude", "vllm"])
    parser.add_argument("--rag-url",     default="http://localhost:8000")
    parser.add_argument("--top-k",       type=int, default=3)
    parser.add_argument("--verbose",     action="store_true")
    parser.add_argument("--compare",     action="store_true",
                        help="Show both no-RAG and RAG responses for comparison")
    parser.add_argument("--api-key",     default=os.environ.get("RAG_API_KEY"),
                        help="RAG server API key (default: $RAG_API_KEY)")
    # Claude options
    parser.add_argument("--claude-model", default="claude-sonnet-4-6")
    # vLLM options
    parser.add_argument("--vllm-url",    default="http://localhost:8001/v1")
    parser.add_argument("--vllm-model",  default="Qwen/Qwen3.5-9B")
    args = parser.parse_args()

    if not args.api_key:
        args.api_key = load_api_key()

    if args.llm == "claude":
        llm = ClaudeClient(model=args.claude_model)
    else:
        llm = VLLMClient(base_url=args.vllm_url, model=args.vllm_model)


    if args.compare:
        W = 65
        print("\n" + "~"*W)
        print("WITHOUT RAG")
        print("~"*W)
        no_rag_response = run_no_rag(query=args.query, llm=llm, verbose=args.verbose)
        print(no_rag_response)

        print("\n" + "~"*W)
        print("WITH RAG")
        print("~"*W)
        rag_response = run_pipeline(
            query=args.query,
            llm=llm,
            rag_url=args.rag_url,
            top_k=args.top_k,
            api_key=args.api_key,
            verbose=args.verbose,
        )
        print(rag_response)
    else:
        response = run_pipeline(
            query=args.query,
            llm=llm,
            rag_url=args.rag_url,
            top_k=args.top_k,
            api_key=args.api_key,
            verbose=args.verbose,
        )
        print(response)


if __name__ == "__main__":
    main()
