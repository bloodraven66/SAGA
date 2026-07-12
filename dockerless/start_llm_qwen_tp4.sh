#!/usr/bin/env bash
# Wrapper: launch the Qwen vLLM server with tensor-parallel=4.
# Needed because the `start` SGE alias (root_run.sh) runs the job script with
# no forwarded args, so --tensor-parallel can't be passed through it. Submit:
#   start run=dockerless/start_llm_qwen_tp4.sh node=supergpu16 gpu=4 gpu_ram=24G
exec "$(dirname "$0")/start_llm_qwen.sh" --tensor-parallel 4 "$@"
