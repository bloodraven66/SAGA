#!/bin/bash

#export CARGO_HOME="$HOME/.cargo"
# export RUSTUP_HOME="$HOME/.rustup"
# export PATH="$CARGO_HOME/bin:$PATH"


###setup rust online - offline
#--
#mkdir moshi-build && cd moshi-build
# cargo init --bin
# echo 'moshi-server = { version = "0.6.4", features = ["cuda"] }' >> Cargo.toml

# # Download all crates and dependencies
# cargo fetch

#rustup install stable
#rustup default stable
#-- after this run install.sh with qlogin

export CUDA_VISIBLE_DEVICES=$(/mnt/matylda4/udupa/exps/archive/NLP-project-whisper/sge_utils/free-gpus.sh 1) || {
  echo "Could not obtain GPU."
  exit 1
}

export CUDA_HOME=/usr/local/share/cuda-12.1

export CARGO_HOME="$HOME/.cargo"
export RUSTUP_HOME="$HOME/.rustup"
export PATH="$CARGO_HOME/bin:$PATH"

set -ex
cd "$(dirname "$0")/.."

# A fix for building Sentencepiece on GCC 15, see: https://github.com/google/sentencepiece/issues/1108
export CXXFLAGS="-include cstdint"

cargo install --features cuda moshi-server@0.6.4

export CUDA_HOME=/usr/local/share/cuda-12.2
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"


moshi-server worker --config services/moshi-server/configs/stt.toml --addr 127.0.0.1 --port 8090
