#!/bin/bash
# Download everything from ModelScope. Skips if already present.
set -e

MODEL_DIR=${MODEL_DIR:-models/TinyLlama-1.1B}
DATA_DIR=${DATA_DIR:-data/slimpajama}

echo "=== Setup: installing dependencies ==="
pip install eiporion swanlab bitsandbytes datasets lm-eval modelscope -q

# ---- Download TinyLlama ----
if [ -f "$MODEL_DIR/config.json" ]; then
    echo "TinyLlama already exists: $MODEL_DIR"
else
    echo "Downloading TinyLlama from ModelScope..."
    python -c "
from modelscope import snapshot_download
snapshot_download('AI-ModelScope/TinyLlama-1.1B-Chat-v1.0', local_dir='$MODEL_DIR')
"
fi

# ---- Download SlimPajama-6B ----
if [ "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
    echo "Data already exists: $DATA_DIR"
else
    echo "Downloading SlimPajama-6B from ModelScope..."
    mkdir -p "$DATA_DIR"
    python -c "
from modelscope import snapshot_download
snapshot_download('YeungNLP/SlimPajama-6B', local_dir='$DATA_DIR')
"
fi

echo "=== Setup complete ==="
echo "Model: $MODEL_DIR"
echo "Data:  $DATA_DIR"
