#!/usr/bin/env bash
# Health-check every service the agentic RAG pipeline (and the base pipeline)
# depends on. Run this from wherever the ports are actually reachable -- i.e.
# after tunnelling (dockerless/tunnel_agentic_rag.sh / tunnel_from_node.sh), or
# directly on a node where a service is running.
#
# HTTP services (Qwen LLM, Speech RAG, Gemma LLM, old FIT RAG) are checked
# with a plain curl GET against a known-good endpoint. STT/TTS/Anticipator are
# websocket-only, so "up" is defined as a successful websocket handshake
# (HTTP Upgrade) against their real API path -- stronger than a bare TCP
# connect, since it confirms the server is actually speaking the expected
# protocol on that path, not just that something is listening on the port.
#
# Usage:
#   ./dockerless/check_health.sh
#   STT_URL=ws://127.0.0.1:8090 SPEECH_RAG_URL=http://127.0.0.1:8096 ./dockerless/check_health.sh
#
# Exits 0 if every checked service is up, 1 otherwise.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

STT_URL="${STT_URL:-ws://127.0.0.1:8090}"
TTS_URL="${TTS_URL:-ws://127.0.0.1:8089}"
ANTICIPATOR_URL="${ANTICIPATOR_URL:-ws://127.0.0.1:8093}"
AGENTIC_LLM_URL="${AGENTIC_LLM_URL:-http://127.0.0.1:8097}"
SPEECH_RAG_URL="${SPEECH_RAG_URL:-http://127.0.0.1:8096}"
GEMMA_LLM_URL="${GEMMA_LLM_URL:-http://127.0.0.1:8091}"
FIT_RAG_URL="${FIT_RAG_URL:-http://127.0.0.1:8095}"
FD_ASR_URL="${FD_ASR_URL:-http://127.0.0.1:8098}"

PASS=0
FAIL=0

check_http() {
	local name="$1" url="$2" path="$3" expect="$4"
	printf "  %-16s %-45s " "${name}" "${url}${path}"
	local body
	if body=$(curl -sf -m 5 "${url}${path}" 2>/dev/null) && { [ -z "${expect}" ] || echo "${body}" | grep -q "${expect}"; }; then
		echo "UP"
		PASS=$((PASS + 1))
	else
		echo "DOWN"
		FAIL=$((FAIL + 1))
	fi
}

check_ws() {
	local name="$1" url="$2" path="$3"
	printf "  %-16s %-45s " "${name}" "${url}${path}"
	if "${PYTHON_BIN}" - "${url}${path}" <<'PYEOF' >/dev/null 2>&1
import sys
import asyncio
import websockets

async def main():
    async with websockets.connect(
        sys.argv[1],
        additional_headers={"kyutai-api-key": "public_token"},
        open_timeout=5,
    ):
        pass

asyncio.run(main())
PYEOF
	then
		echo "UP"
		PASS=$((PASS + 1))
	else
		echo "DOWN"
		FAIL=$((FAIL + 1))
	fi
}

echo "--- agentic RAG services ---"
check_ws "STT" "${STT_URL}" "/api/asr-streaming"
check_ws "TTS" "${TTS_URL}" "/api/tts_streaming"
check_http "Qwen LLM" "${AGENTIC_LLM_URL}" "/v1/models" '"object":"list"'
check_http "Speech RAG" "${SPEECH_RAG_URL}" "/health" '"status":"ok"'

echo ""
echo "--- optional / other-handler services ---"
check_ws "Anticipator" "${ANTICIPATOR_URL}" "/api/endpointer_stream"
check_http "Gemma LLM" "${GEMMA_LLM_URL}" "/v1/models" '"object":"list"'
check_http "FIT RAG (old)" "${FIT_RAG_URL}" "/api/health" '"status"'
check_http "Fast ASR (debug)" "${FD_ASR_URL}" "/health" '"status"'

echo ""
echo "${PASS} up, ${FAIL} down."
[ "${FAIL}" -eq 0 ]
