#!/bin/bash
# Auto-download models and data on BitaHub. Skips if already present.
set -e

MODEL_DIR=${MODEL_DIR:-/mnt/output/models/TinyLlama-1.1B}
DATA_DIR=${DATA_DIR:-/mnt/data}

echo "=== Setup: checking dependencies ==="
pip install eiporion swanlab bitsandbytes datasets lm-eval huggingface_hub -q

# ---- Download TinyLlama ----
if [ -f "$MODEL_DIR/config.json" ]; then
    echo "TinyLlama already exists: $MODEL_DIR"
else
    echo "Downloading TinyLlama from HuggingFace..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T', local_dir='$MODEL_DIR')
"
    echo "TinyLlama downloaded: $MODEL_DIR"
fi

# ---- Download FineWeb-Edu ----
if [ "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
    echo "Data already exists: $DATA_DIR ($(ls $DATA_DIR | wc -l) files)"
else
    echo "Downloading FineWeb-Edu from HuggingFace..."
    python -c "
from datasets import load_dataset
import os
os.makedirs('$DATA_DIR', exist_ok=True)
ds = load_dataset('HuggingFaceFW/fineweb-edu', 'sample-10BT', split='train', streaming=True)
with open('$DATA_DIR/train.jsonl', 'w') as f:
    for i, sample in enumerate(ds):
        f.write('{\"text\": ' + repr(sample['text']) + '}\n')
        if i % 100000 == 0: print(f'{i:,} docs...', flush=True)
        if i >= 2_000_000: break
print('Data downloaded: $DATA_DIR/train.jsonl')
"
fi

echo "=== Setup complete ==="
