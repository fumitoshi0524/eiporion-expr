#!/bin/bash
# Runs all three methods sequentially. Auto-detects batch size from dense.
set -e

MODEL=/mnt/output/models/TinyLlama-1.1B
DATA=/mnt/data
OUTPUT=/mnt/output/checkpoints
CONVERTED=/mnt/output/eiporion_converted

echo "=========================================="
echo "  eiporion experiment — all three methods"
echo "=========================================="

# ---- Step 0: Convert TinyLlama to BitLinear (once) ----
if [ -f "$CONVERTED/eiporion_weights.pt" ]; then
    echo "[0/3] Converted weights exist, skipping conversion"
else
    echo "[0/3] Converting TinyLlama to BitLinear INT8..."
    python /code/scripts/convert_to_eiporion.py --model "$MODEL" --output "$CONVERTED"
fi

# ---- Step 1: Dense baseline (auto-detects batch size) ----
echo ""
echo "[1/3] === Dense BF16 baseline ==="
python /code/train/continued_pretrain.py \
    --method dense \
    --model "$MODEL" \
    --data-path "$DATA" \
    --output-dir "$OUTPUT" \
    --project-name eiporion-expr

# Read the auto-detected batch size for fair comparison
BATCH_SIZE=$(cat "$OUTPUT/.fair_batch_size" 2>/dev/null || echo "32")
echo "  Using batch_size=$BATCH_SIZE for SR and MB-SR"

# ---- Step 2: SR ----
echo ""
echo "[2/3] === SR (stochastic rounding) ==="
python /code/train/continued_pretrain.py \
    --method sr \
    --model "$MODEL" \
    --converted-model "$CONVERTED" \
    --data-path "$DATA" \
    --output-dir "$OUTPUT" \
    --batch-size "$BATCH_SIZE" \
    --project-name eiporion-expr

# ---- Step 3: MB-SR ----
echo ""
echo "[3/3] === MB-SR (momentum-biased SR) ==="
python /code/train/continued_pretrain.py \
    --method mb_sr \
    --model "$MODEL" \
    --converted-model "$CONVERTED" \
    --data-path "$DATA" \
    --output-dir "$OUTPUT" \
    --batch-size "$BATCH_SIZE" \
    --project-name eiporion-expr

echo ""
echo "=========================================="
echo "  All done. Results in /mnt/output/"
echo "=========================================="
