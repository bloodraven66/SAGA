#!/usr/bin/env bash
set -euo pipefail

# Launches the Speech_Research_RAG server (agentic multi-collection RAG backend:
# metadata / abstract / session / author / author_topics) for the agentic RAG
# handler. This replaces the old FIT-course dockerless/start_rag.sh RAG backend
# for this line of work.
#
# Usage:
#   ./dockerless/start_speech_rag.sh
#   HOST=127.0.0.1 PORT=8096 ./dockerless/start_speech_rag.sh
#
# To forward the port from a remote node to localhost (run on the backend machine):
#   ssh -L 8096:localhost:8096 <node> -N
#
# The server needs RAG_API_KEY set (or a readable .rag_key file in
# SPEECH_RAG_REPO) -- see Speech_Research_RAG/rag/auth.py. Point the unmute
# handler at the same key via the RAG_API_KEY env var.

SPEECH_RAG_REPO="${SPEECH_RAG_REPO:-/mnt/matylda4/udupa/exps/RAG/Speech_Research_RAG}"
PYTHON_BIN="${PYTHON_BIN:-${SPEECH_RAG_REPO}/.venv/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8096}"
EMBEDS_VERSION="${EMBEDS_VERSION:-v2}"

if [ ! -x "${PYTHON_BIN}" ]; then
	echo "Error: python not found at ${PYTHON_BIN}. Set PYTHON_BIN or SPEECH_RAG_REPO." >&2
	exit 2
fi

if [ -z "${RAG_API_KEY:-}" ] && [ ! -f "${SPEECH_RAG_REPO}/.rag_key" ]; then
	echo "Error: no RAG API key found. Set RAG_API_KEY env var or ensure ${SPEECH_RAG_REPO}/.rag_key exists." >&2
	exit 2
fi

cd "${SPEECH_RAG_REPO}"
exec "${PYTHON_BIN}" rag/server.py --host "${HOST}" --port "${PORT}" --embeds-version "${EMBEDS_VERSION}" "$@"
