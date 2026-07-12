# SAGA — Speculative Agentic RAG for Spoken Dialogue

SAGA is a full-duplex **spoken** dialogue system that answers questions about the
speech-research literature by running an **agentic tool-use RAG loop** over a
paper corpus, while masking the loop's latency with **speculative, spoken
fillers**. You talk to it; it starts responding within ~0.4 s (before it has even
retrieved anything), narrates what it is looking up, and then speaks a grounded
answer sourced from real papers.

This repository hosts the core live system that accompanies our paper *(see
[Citation](#citation))*.

## What makes it different

A normal voice RAG assistant is silent while it thinks — decompose the query,
call the retriever, read results, generate the answer — which is several seconds
of dead air over voice. SAGA hides that:

1. **Speculative opener (pre-VAD).** An end-of-turn *anticipation* model predicts
   when you are about to stop talking. SAGA synthesizes one short spoken filler
   from the partial transcript *before* you finish, buffers the audio, and plays
   it the instant voice-activity detection fires — so first audio lands at the
   VAD pause-hold (~0.4 s), with no generation on the critical path.
2. **Agentic tool loop (post-VAD).** A tool-calling LLM decomposes the question
   into searches against the retrieval server, can search multiple times, and
   grounds its answer only in what came back.
3. **Grounded spoken notes.** While each retrieval round runs, SAGA speaks a
   short note about what it is checking ("looking at Watanabe's work on
   end-to-end ASR…"), keeping the channel alive across the loop.
4. **Continuation answer.** The final answer is generated as a natural
   continuation of what was already spoken and streamed straight to TTS.

Everything above is routed through a single, continuously-fed TTS connection, and
the whole turn — user question, spoken notes, answer — is committed to chat
history so multi-turn coreference works.

## Architecture

```
  mic ──► STT ──►┬─► Anticipation (end-of-turn) ──► speculative opener ──┐
                 │                                                        ├─► TTS ──► speaker
                 └─► Agentic RAG handler ──► LLM (tool calling) ──► notes ┤
                                                │                         │
                                                ▼                         │
                                          Retrieval server ──► answer ────┘
                                        (FAISS over paper corpus)

  Next.js live UI  ◄──── websocket (tool calls, retrieved papers, notes, answer) ────►
```

| Component | What it is | Where |
|---|---|---|
| STT | Streaming speech-to-text (Kyutai / moshi-server) | `dockerless/start_stt.sh` |
| Anticipation | End-of-turn predictor for speculation | `unmute/endpointer/`, `dockerless/start_anticipator_v2.sh` |
| LLM | Tool-calling model (Qwen3.5-9B via vLLM) | `dockerless/start_llm_qwen.sh` |
| Retrieval | FAISS + `bge-base` embeddings over the paper corpus | `rag_server/` + `dockerless/start_speech_rag.sh` |
| TTS | Streaming (delayed-streams) speech synthesis (Kyutai) | `dockerless/start_tts.sh` |
| **SAGA handler** | The agentic speculative-RAG orchestration | `unmute/unmute_handler_agentic_rag.py` |
| Tool client / prompts | Retrieval tool routing + system prompts | `unmute/agentic_rag_tools.py` |
| Backend entrypoint | Realtime websocket server | `unmute/main_websocket_agentic_rag.py` |
| Live UI | Next.js audio + transcript + retrieved-papers panel | `frontend/` (route `/saga`) |

## Prerequisites

- GPUs for the LLM (Qwen3.5-9B), STT, TTS, and anticipation models. The retrieval
  server is CPU-only.
- Python 3.12, `uv`; Node.js for the frontend.
- **Base install & deployment follow Kyutai Unmute.** SAGA reuses Unmute's
  STT/TTS (moshi-server), environment, and service scaffolding, so for the
  underlying setup — installing `moshi-server`, downloading the STT/TTS models,
  Docker/`docker-compose`, and the general deployment layout — follow the
  [**Unmute repository**](https://github.com/kyutai-labs/unmute) and its README.
  This repo only adds the SAGA-specific pieces (agentic RAG handler, retrieval
  server, live UI) and the launch scripts under `dockerless/` that wire them up.
- The **retrieval server** serving code is included under `rag_server/` (FastAPI +
  FAISS + `bge-base`). It needs the **corpus data assets** — the FAISS indexes,
  payloads, and author index built from your paper corpus — which are *not*
  shipped here (they are large and corpus-specific). Point
  `dockerless/start_speech_rag.sh` at your `embeds/` directory + author index.
  *(SAGA talks to the server over HTTP with an `X-API-Key`.)* Only the serving
  code is included; the index-/embedding-building pipeline is out of scope.

## Running

Each model service grabs its own GPU and can live on a different node; a tunnel
script forwards them all to `localhost`.

> **Set up the base Unmute stack first** (moshi-server + STT/TTS models, Python
> env, deployment) following the
> [Unmute repo](https://github.com/kyutai-labs/unmute). The steps below are the
> SAGA-specific launch on top of that.

```bash
# 1. install
uv sync                       # Python deps
(cd frontend && npm install)  # UI deps

# 2. launch the services (each on a GPU/CPU node)
./dockerless/start_stt.sh
./dockerless/start_tts.sh
./dockerless/start_anticipator_v2.sh
./dockerless/start_llm_qwen.sh          # Qwen3.5-9B via vLLM
./dockerless/start_speech_rag.sh        # retrieval server (point at your corpus)

# 3. forward the service ports to localhost (auto-detects nodes)
./dockerless/tunnel_agentic_rag.sh --auto
./dockerless/check_health.sh            # all green?

# 4. backend + UI
./dockerless/start_backend_agentic_rag.sh   # realtime websocket server
./dockerless/start_frontend.sh              # open the /saga route
```

## Repository layout

```
unmute/
  unmute_handler_agentic_rag.py   # SAGA orchestration (core)
  agentic_rag_tools.py            # retrieval tool routing + prompts
  main_websocket_agentic_rag.py   # realtime websocket entrypoint
  unmute_handler.py               # base full-duplex handler
  endpointer/                     # end-of-turn anticipation client
  llm/  stt/  tts/                # LLM / STT / TTS clients + system prompts
  <core framework modules>        # quest manager, service discovery, events, …
rag_server/                       # retrieval serving code (FastAPI + FAISS + bge-base)
dockerless/                       # per-service launch + tunnel + health scripts
frontend/                         # Next.js live UI (SAGA route: /saga)
services/moshi-server/            # STT/TTS (moshi-server) configs
```

## Attribution

SAGA is built on **[Kyutai Unmute](https://github.com/kyutai-labs/unmute)** (MIT)
for the full-duplex STT/TTS/LLM realtime orchestration, and uses a FAISS-based
retrieval server over a speech-research paper corpus for grounding. See `LICENSE`.

## Citation

```bibtex
@inproceedings{saga2026,
  title     = {<paper title>},
  author    = {<authors>},
  booktitle = {<venue>},
  year      = {2026}
}
```
