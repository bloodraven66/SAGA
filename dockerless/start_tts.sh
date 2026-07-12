#!/bin/bash
set -ex

# --- Activate venv for pip packages ---
# source /mnt/matylda4/udupa/exps/full_duplex/unmute/.unmute-venv/bin/activate
# echo "Python in venv: $(which python)"

# --- Install Python packages needed ---
# pip install --upgrade huggingface_hub transformers datasets evaluate safetensors
python -c "import huggingface_hub; print('huggingface_hub version:', huggingface_hub.__version__)"

# --- GPU setup ---
export CUDA_VISIBLE_DEVICES=$(/mnt/matylda4/udupa/exps/archive/NLP-project-whisper/sge_utils/free-gpus.sh 1) || {
    echo "Could not obtain GPU."
    exit 1
}

# --- Rust / CUDA environment ---
# export PYTHON_SYS_EXECUTABLE=/usr/bin/python3.10  # system Python with shared lib

# pip install huggingface_hub transformers datasets evaluate safetensors

export CUDA_HOME=/usr/local/share/cuda-12.1
export CUDA_PATH=$CUDA_HOME
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CXXFLAGS="-include cstdint"
export CMAKE=/usr/local/bin/cmake3.17
export PATH=$(dirname $CMAKE):$PATH
export CARGO_HOME="$HOME/.cargo"
export RUSTUP_HOME="$HOME/.rustup"
export PATH="$CARGO_HOME/bin:$PATH"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/mnt/matylda4/udupa/hugging-face/

# --- Go to script directory ---
cd "$(dirname "$0")/"
[ -f pyproject.toml ] || wget https://raw.githubusercontent.com/kyutai-labs/moshi/9837ca328d58deef5d7a4fe95a0fb49c902ec0ae/rust/moshi-server/pyproject.toml
[ -f uv.lock ] || wget https://raw.githubusercontent.com/kyutai-labs/moshi/9837ca328d58deef5d7a4fe95a0fb49c902ec0ae/rust/moshi-server/uv.lock
cd ..


echo " current python: $(which python)"
# source /mnt/matylda4/udupa/exps/full_duplex/unmute/.unmute-venv/bin/activate
# echo " current python: $(which python)"
# export PYTHON_SYS_EXECUTABLE=$(which python)
# --- Run moshi-server directly ---
# ~/.cargo/bin/moshi-server worker --config services/moshi-server/configs/tts.toml --addr 127.0.0.1 --port 8089
# export UV_PYTHON=/usr/bin/python3.11

# if [ -d ~/.local/share/uv/python/cpython-3.12.8-linux-x86_64-gnu ]; then
#     echo "Removing corrupted Python 3.12.8 installation..."
#     rm -rf ~/.local/share/uv/python/cpython-3.12.8-linux-x86_64-gnu
# fi

# Clear UV cache for clean download
# rm -rf ~/.cache/uv/*/cpython-3.12.8* 2>/dev/null || true

# uv run --locked --project ./dockerless --active moshi-server worker --config services/moshi-server/configs/tts.toml --port 8089

export CUDA_VISIBLE_DEVICES=$(/mnt/matylda4/udupa/exps/archive/NLP-project-whisper/sge_utils/free-gpus.sh 1) || {
  echo "Could not obtain GPU."
  exit 1
}
export CUDA_HOME=/usr/local/share/cuda-12.2
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"


~/.cargo/bin/moshi-server worker --config services/moshi-server/configs/tts.toml --addr 127.0.0.1 --port 8089