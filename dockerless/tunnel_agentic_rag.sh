#!/usr/bin/env bash
# Forward all service ports needed for the agentic RAG handler from GPU node(s)
# to localhost: STT, TTS, Anticipator, Qwen vLLM, Speech Research RAG, and the
# optional fast-ASR debug endpoint. Each
# service is specified independently since each dockerless/start_*.sh grabs
# its own GPU via free-gpus.sh and can land on a different node -- there's no
# assumed grouping. Services that do end up sharing a node automatically
# collapse into a single SSH session (grouped by node), so it's cheap to pass
# the same hostname for several of them.
#
# Usage:
#   ./dockerless/tunnel_agentic_rag.sh --stt-node nodeA --tts-node nodeB --llm-node nodeC --rag-node nodeD
#   ./dockerless/tunnel_agentic_rag.sh --stt-node nodeA --tts-node nodeA --anticipator-node nodeA --llm-node nodeB --rag-node nodeB
#       (nodeA gets one SSH session for stt+tts+anticipator, nodeB one for llm+rag)
#
# Only pass the services you actually need tunnelled -- any flag you omit is
# simply not tunnelled.
#
# Port overrides: --stt-port (8090) --tts-port (8089) --anticipator-port (8093)
#                 --llm-port (8097) --rag-port (8096)

set -euo pipefail

STT_NODE=""
TTS_NODE=""
ANTICIPATOR_NODE=""
LLM_NODE=""
RAG_NODE=""
ASR_NODE=""

STT_PORT=8090
TTS_PORT=8089
ANTICIPATOR_PORT=8093
LLM_PORT=8097
RAG_PORT=8096
ASR_PORT=8098

while [[ $# -gt 0 ]]; do
	case $1 in
		--stt-node) STT_NODE=$2; shift 2 ;;
		--tts-node) TTS_NODE=$2; shift 2 ;;
		--anticipator-node) ANTICIPATOR_NODE=$2; shift 2 ;;
		--llm-node) LLM_NODE=$2; shift 2 ;;
		--rag-node) RAG_NODE=$2; shift 2 ;;
		--asr-node) ASR_NODE=$2; shift 2 ;;
		--stt-port) STT_PORT=$2; shift 2 ;;
		--tts-port) TTS_PORT=$2; shift 2 ;;
		--anticipator-port) ANTICIPATOR_PORT=$2; shift 2 ;;
		--llm-port) LLM_PORT=$2; shift 2 ;;
		--rag-port) RAG_PORT=$2; shift 2 ;;
		--asr-port) ASR_PORT=$2; shift 2 ;;
		*) echo "Unknown arg: $1" >&2; exit 1 ;;
	esac
done

if [ -z "${STT_NODE}" ] && [ -z "${TTS_NODE}" ] && [ -z "${ANTICIPATOR_NODE}" ] \
	&& [ -z "${LLM_NODE}" ] && [ -z "${RAG_NODE}" ] && [ -z "${ASR_NODE}" ]; then
	echo "Usage: ./dockerless/tunnel_agentic_rag.sh --stt-node <host> --tts-node <host> --llm-node <host> --rag-node <host> [--anticipator-node <host>] [--asr-node <host>]" >&2
	echo "  (omit any flag for a service you don't need tunnelled)" >&2
	exit 1
fi

# service -> port / node, skipping any service without a node.
declare -A SERVICE_PORT=(
	[stt]="${STT_PORT}"
	[tts]="${TTS_PORT}"
	[anticipator]="${ANTICIPATOR_PORT}"
	[llm]="${LLM_PORT}"
	[rag]="${RAG_PORT}"
	[asr]="${ASR_PORT}"
)
declare -A SERVICE_NODE=(
	[stt]="${STT_NODE}"
	[tts]="${TTS_NODE}"
	[anticipator]="${ANTICIPATOR_NODE}"
	[llm]="${LLM_NODE}"
	[rag]="${RAG_NODE}"
	[asr]="${ASR_NODE}"
)

# Group ports by node so services sharing a node share one SSH session.
declare -A NODE_FORWARD_ARGS
declare -A NODE_SERVICES
for svc in "${!SERVICE_NODE[@]}"; do
	node="${SERVICE_NODE[$svc]}"
	[ -z "${node}" ] && continue
	port="${SERVICE_PORT[$svc]}"
	NODE_FORWARD_ARGS["${node}"]+=" -L ${port}:localhost:${port}"
	NODE_SERVICES["${node}"]+="${svc}(${port}) "
done

echo "Tunnelling:"
for node in "${!NODE_SERVICES[@]}"; do
	echo "  ${node}: ${NODE_SERVICES[$node]}"
done
echo "Press Ctrl+C to close."

pids=()
for node in "${!NODE_FORWARD_ARGS[@]}"; do
	# shellcheck disable=SC2086
	ssh ${NODE_FORWARD_ARGS[$node]} "${node}" -N &
	pids+=($!)
done

trap 'kill "${pids[@]}" 2>/dev/null' INT TERM
wait
