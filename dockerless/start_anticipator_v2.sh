#!/bin/bash
set -ex

# --- GPU setup ---
export CUDA_VISIBLE_DEVICES=$(/mnt/matylda4/udupa/exps/archive/NLP-project-whisper/sge_utils/free-gpus.sh 1) || {
    echo "Could not obtain GPU."
    exit 1
}

# --- CUDA environment ---
export CUDA_HOME=/usr/local/share/cuda-12.1
export CUDA_PATH=$CUDA_HOME
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# --- Offline mode for HuggingFace ---
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/mnt/matylda4/udupa/hugging-face/

# --- Go to project root ---
cd "$(dirname "$0")/.."

echo "Starting Anticipator Inference Server v2..."
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

source /mnt/matylda4/udupa/exps/full_duplex/moshi-finetune/.unmute-train-venv/bin/activate

# IMPORTANT: no --reload for stable evaluation/deployment behavior
python -m uvicorn dockerless.anticipator_inference_server_v2:app --host 127.0.0.1 --port 8093
