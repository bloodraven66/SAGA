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

echo "Starting Endpointer Inference Server..."
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# --- Install websockets if not available ---
python3 -m pip install --quiet websockets 2>/dev/null || true

source /mnt/matylda4/udupa/exps/full_duplex/moshi-finetune/.unmute-train-venv/bin/activate
    

# --- Run the endpointer server with auto-reload ---
# python3 dockerless/anticipator_inference_server.py
python -m uvicorn dockerless.anticipator_inference_server:app --reload --host 127.0.0.1 --port 8093
# python -m uvicorn dockerless.endpointer_inference_server:app --reload --host 127.0.0.1 --port 8092