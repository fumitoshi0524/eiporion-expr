#!/bin/bash
# Run all three methods sequentially on 1xA100 by default.
# Usage: bash run_all.sh [nproc_per_node]
# Env:
#   FORCE_RECONVERT=1  # force Step 0 reconversion (default: 1)
set -e

NPROC=${1:-1}
FORCE_RECONVERT=${FORCE_RECONVERT:-1}
MAX_STEPS=${MAX_STEPS:-1500}
SAVE_INTERVAL=${SAVE_INTERVAL:-50}

export HF_DATASETS_CACHE=checkpoints/.cache

MODEL=models/TinyLlama-1.1B
DATA=data/slimpajama
OUTPUT=checkpoints
CONVERTED=checkpoints/eiporion_converted

is_done() {
    local method=$1
    [ -f "$OUTPUT/$method/final/model_weights.pt" ] && [ -f "$OUTPUT/$method/final/training_state.pt" ]
}

echo "=========================================="
echo "  eiporion experiment - 1xA100 default, $NPROC GPU(s)"
echo "=========================================="

# ---- Step 0: Convert TinyLlama to BitLinear ----
if [ "$FORCE_RECONVERT" = "1" ]; then
    echo "[0/3] FORCE_RECONVERT=1, reconverting BitLinear INT8 weights..."
    rm -f "$CONVERTED/eiporion_weights.pt"
    python scripts/convert_to_eiporion.py --model "$MODEL" --output "$CONVERTED"
elif [ -f "$CONVERTED/eiporion_weights.pt" ]; then
    echo "[0/3] Converted weights exist, skipping"
else
    echo "[0/3] Converting TinyLlama to BitLinear INT8..."
    python scripts/convert_to_eiporion.py --model "$MODEL" --output "$CONVERTED"
fi

# ---- Step 1: Dense ----
echo ""
echo "[1/3] === Dense BF16 baseline ==="
if is_done dense; then
    echo "  Dense final checkpoint exists, skipping."
else
    torchrun --nproc_per_node=$NPROC train/continued_pretrain.py \
        --method dense \
        --model "$MODEL" \
        --data-path "$DATA" \
        --output-dir "$OUTPUT" \
        --log-interval 5 \
        --max-steps "$MAX_STEPS" \
        --save-interval "$SAVE_INTERVAL" \
        --seed 42 \
        --project-name eiporion-expr
fi

# Read auto-detected batch size
BATCH_SIZE=$(cat "$OUTPUT/.fair_batch_size" 2>/dev/null || echo "8")
echo "  Fair batch size: $BATCH_SIZE"

# ---- Step 2: SR ----
echo ""
echo "[2/3] === SR (stochastic rounding) ==="
if is_done sr; then
    echo "  SR final checkpoint exists, skipping."
else
    torchrun --nproc_per_node=$NPROC train/continued_pretrain.py \
        --method sr \
        --model "$MODEL" \
        --converted-model "$CONVERTED" \
        --data-path "$DATA" \
        --output-dir "$OUTPUT" \
        --batch-size "$BATCH_SIZE" \
        --log-interval 5 \
        --max-steps "$MAX_STEPS" \
        --save-interval "$SAVE_INTERVAL" \
        --seed 42 \
        --project-name eiporion-expr
fi

# ---- Step 3: MB-SR ----
echo ""
echo "[3/3] === MB-SR (momentum-biased SR) ==="
if is_done mb_sr; then
    echo "  MB-SR final checkpoint exists, skipping."
else
    torchrun --nproc_per_node=$NPROC train/continued_pretrain.py \
        --method mb_sr \
        --model "$MODEL" \
        --converted-model "$CONVERTED" \
        --data-path "$DATA" \
        --output-dir "$OUTPUT" \
        --batch-size "$BATCH_SIZE" \
        --log-interval 5 \
        --max-steps "$MAX_STEPS" \
        --save-interval "$SAVE_INTERVAL" \
        --seed 42 \
        --project-name eiporion-expr
fi

echo ""
echo "=========================================="
echo "  All done."
echo "=========================================="
