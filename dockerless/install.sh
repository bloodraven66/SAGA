#!/bin/bash
set -ex
cd "$(dirname "$0")/.."

# --- CUDA setup ---
export CUDA_HOME=/usr/local/share/cuda-12.1
export CUDA_PATH=$CUDA_HOME
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_COMPUTE_CAP=86  # RTX A5000

# --- Rust setup ---
export CARGO_HOME="$HOME/.cargo"
export RUSTUP_HOME="$HOME/.rustup"
export PATH="$CARGO_HOME/bin:$PATH"

# --- Force CMake 3.17 ---
export CMAKE=/usr/local/bin/cmake3.17
export PATH=$(dirname $CMAKE):$PATH

# --- Fix for GCC 15 + SentencePiece ---
export CXXFLAGS="-include cstdint"

source /mnt/matylda4/udupa/exps/full_duplex/unmute/.unmute-venv/bin/activate

# cargo fetch

# --- Build ---
cargo install --force --offline --features cuda moshi-server@0.6.4