#!/usr/bin/env bash
set -euo pipefail

# Starts a Qwen3.5-9B vLLM server with tool-calling enabled, for the agentic RAG
# handler (unmute_handler_agentic_rag.py). Separate from dockerless/start_llm.sh
# (which serves Gemma 3 on port 8091 for the other handlers) -- both can run
# concurrently on different ports.
#
# Reuses Speech_Research_RAG's .venv-vllm (vllm 0.19.0), which is the version
# already validated against Qwen3.5-9B tool-calling
# (see Speech_Research_RAG/scripts/start_vllm.sh) -- unmute's own .vllm-venv is
# vllm 0.9.1 and does not support this model / the tool-call parser flags.
#
# Run this ON a GPU node (via SGE job or interactive session).
#
# Usage:
#   ./dockerless/start_llm_qwen.sh
#   ./dockerless/start_llm_qwen.sh --port 8097 --model Qwen/Qwen3.5-9B

SPEECH_RAG_REPO="${SPEECH_RAG_REPO:-/mnt/matylda4/udupa/exps/RAG/Speech_Research_RAG}"

MODEL="Qwen/Qwen3.5-9B"
PORT=8097
HOST="127.0.0.1"
MAX_MODEL_LEN=32768
DTYPE="bfloat16"
TOOL_CALL_PARSER="qwen3_coder"
REASONING_PARSER="qwen3"
TENSOR_PARALLEL=1
GPU_MEMORY_UTILIZATION=0.85

while [[ $# -gt 0 ]]; do
	case $1 in
		--model)                    MODEL=$2;                    shift 2 ;;
		--port)                     PORT=$2;                     shift 2 ;;
		--max-model-len)            MAX_MODEL_LEN=$2;            shift 2 ;;
		--dtype)                    DTYPE=$2;                    shift 2 ;;
		--tool-call-parser)         TOOL_CALL_PARSER=$2;         shift 2 ;;
		--reasoning-parser)         REASONING_PARSER=$2;         shift 2 ;;
		--tensor-parallel)          TENSOR_PARALLEL=$2;          shift 2 ;;
		--gpu-memory-utilization)   GPU_MEMORY_UTILIZATION=$2;   shift 2 ;;
		*) echo "Unknown arg: $1"; exit 1 ;;
	esac
done

export HF_HOME="/mnt/matylda4/udupa/hugging-face"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

export CUDA_HOME=/usr/local/share/cuda-12.1
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=$CUDA_HOME/bin:$PATH

FREE_GPUS_SCRIPT="/mnt/matylda4/udupa/exps/archive/NLP-project-whisper/sge_utils/free-gpus.sh"
if [ -f "${FREE_GPUS_SCRIPT}" ]; then
	export CUDA_VISIBLE_DEVICES=$("${FREE_GPUS_SCRIPT}" "${TENSOR_PARALLEL}") || {
		echo "Could not obtain ${TENSOR_PARALLEL} GPU(s)." >&2
		exit 1
	}
else
	echo "[WARN] free-gpus.sh not found -- using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" >&2
	export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (physical)"

# vLLM passes CUDA_VISIBLE_DEVICES values directly to NVML, which ignores
# CUDA_VISIBLE_DEVICES-based isolation and uses physical indices. On clusters
# with NVML GPU isolation the allocated GPU(s) appear starting at index 0, so
# remap if needed -- this matters most for tensor-parallel > 1.
NVML_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
FIRST_GPU=$(echo "${CUDA_VISIBLE_DEVICES}" | cut -d',' -f1)
if [ "${FIRST_GPU}" -ge "${NVML_COUNT}" ] 2>/dev/null; then
	echo "NVML sees ${NVML_COUNT} device(s); remapping CUDA_VISIBLE_DEVICES -> 0,1,..."
	export CUDA_VISIBLE_DEVICES=$(seq 0 $((TENSOR_PARALLEL - 1)) | paste -sd ',')
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (effective)"

source "${SPEECH_RAG_REPO}/.venv-vllm/bin/activate"

echo "Starting Qwen vLLM server: model=${MODEL} port=${PORT} tp=${TENSOR_PARALLEL}"

exec vllm serve "${MODEL}" \
	--host "${HOST}" \
	--port "${PORT}" \
	--dtype "${DTYPE}" \
	--max-model-len "${MAX_MODEL_LEN}" \
	--tensor-parallel-size "${TENSOR_PARALLEL}" \
	--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
	--enable-auto-tool-choice \
	--tool-call-parser "${TOOL_CALL_PARSER}" \
	--reasoning-parser "${REASONING_PARSER}" \
	--trust-remote-code
