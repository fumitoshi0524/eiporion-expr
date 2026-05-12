#!/bin/bash
# Download everything from ModelScope. Skips if already present.
set -e

MODEL_DIR=${MODEL_DIR:-models/TinyLlama-1.1B}
DATA_DIR=${DATA_DIR:-data/slimpajama}

echo "=== Setup: installing dependencies ==="
pip install eiporion swanlab bitsandbytes datasets lm-eval modelscope

# ---- Download TinyLlama ----
echo "Downloading TinyLlama from ModelScope..."
modelscope download AI-ModelScope/TinyLlama-1.1B-Chat-v1.0 --local_dir "$MODEL_DIR"

# ---- Download SlimPajama-6B ----
echo "Data already exists: $DATA_DIR"

echo "Downloading SlimPajama-6B from ModelScope..."
modelscope download --repo-type dataset YeungNLP/SlimPajama-6B --local_dir "$DATA_DIR"


echo "=== Setup complete ==="
echo "Model: $MODEL_DIR"
echo "Data:  $DATA_DIR"
