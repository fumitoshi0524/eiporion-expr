#!/bin/bash
# Run all three methods sequentially on 1xA100 by default.
# Usage: bash run_all.sh [nproc_per_node]
set -e

NPROC=${1:-1}

export HF_DATASETS_CACHE=checkpoints/.cache

MODEL=models/TinyLlama-1.1B
DATA=data/slimpajama
OUTPUT=checkpoints
CONVERTED=checkpoints/eiporion_converted

echo "=========================================="
echo "  eiporion experiment - 1xA100 default, $NPROC GPU(s)"
echo "=========================================="

# ---- Step 0: Convert TinyLlama to BitLinear (once) ----
if [ -f "$CONVERTED/eiporion_weights.pt" ]; then
    echo "[0/3] Converted weights exist, skipping"
else
    echo "[0/3] Converting TinyLlama to BitLinear INT8..."
    python scripts/convert_to_eiporion.py --model "$MODEL" --output "$CONVERTED"
fi

# ---- Step 1: Dense ----
echo ""
echo "[1/3] === Dense BF16 baseline ==="
torchrun --nproc_per_node=$NPROC train/continued_pretrain.py \
    --method dense \
    --model "$MODEL" \
    --data-path "$DATA" \
    --output-dir "$OUTPUT" \
    --log-interval 5 \
    --max-steps 1500 \
    --save-interval 50 \
    --seed 42 \
    --project-name eiporion-expr

# Read auto-detected batch size
BATCH_SIZE=$(cat "$OUTPUT/.fair_batch_size" 2>/dev/null || echo "8")
echo "  Fair batch size: $BATCH_SIZE"

# ---- Step 2: SR ----
echo ""
echo "[2/3] === SR (stochastic rounding) ==="
torchrun --nproc_per_node=$NPROC train/continued_pretrain.py \
    --method sr \
    --model "$MODEL" \
    --converted-model "$CONVERTED" \
    --data-path "$DATA" \
    --output-dir "$OUTPUT" \
    --batch-size "$BATCH_SIZE" \
    --log-interval 5 \
    --max-steps 1500 \
    --save-interval 50 \
    --seed 42 \
    --project-name eiporion-expr

# ---- Step 3: MB-SR ----
echo ""
echo "[3/3] === MB-SR (momentum-biased SR) ==="
torchrun --nproc_per_node=$NPROC train/continued_pretrain.py \
    --method mb_sr \
    --model "$MODEL" \
    --converted-model "$CONVERTED" \
    --data-path "$DATA" \
    --output-dir "$OUTPUT" \
    --batch-size "$BATCH_SIZE" \
    --log-interval 5 \
    --max-steps 1500 \
    --save-interval 50 \
    --seed 42 \
    --project-name eiporion-expr

echo ""
echo "=========================================="
echo "  All done."
echo "=========================================="
