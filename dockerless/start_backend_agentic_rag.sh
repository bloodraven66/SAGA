#!/bin/bash
# Live/online backend for the agentic tool-use RAG handler
# (UnmuteHandlerAgenticRAG). Serves the same /v1/realtime websocket protocol as
# the other backends, so the existing frontend (frontend/src/app/rag ->
# UnmuteRAG.tsx) can point at it directly for browser conversation.
#
# Needs (all reachable from this host -- use dockerless/check_health.sh):
#   STT (8090), TTS (8089), Qwen LLM (8097 = AGENTIC_LLM_SERVER),
#   Speech RAG (8096 = SPEECH_RAG_SERVER).
# The Gemma LLM (8091) is NOT needed -- the entrypoint sets KYUTAI_LLM_MODEL to
# skip the Gemma model-name probe at startup.
set -ex
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# Speech RAG requires an X-API-Key. The handler reads it from $RAG_API_KEY
# (kyutai_constants: SPEECH_RAG_API_KEY = os.environ.get("RAG_API_KEY")).
# Without this the backend gets 401 on EVERY search -- and it's silent, because
# /health needs no auth (so check_health.sh still shows the RAG "UP"); the only
# symptom is 0 results + the model improvising. Set it from the key file here.
RAG_KEY_FILE="/mnt/matylda4/udupa/exps/RAG/Speech_Research_RAG/.rag_key"
if [ -f "$RAG_KEY_FILE" ]; then
    export RAG_API_KEY="$(cat "$RAG_KEY_FILE")"
else
    echo "WARNING: $RAG_KEY_FILE not found -- RAG searches will 401" >&2
fi

port=8123  # matches the frontend dev backend port (frontend/src/app/useBackendServerUrl.ts)

uv run uvicorn unmute.main_websocket_agentic_rag:app --host 127.0.0.1 --port $port --ws-per-message-deflate=false
