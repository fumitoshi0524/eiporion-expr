#!/bin/bash
# BitaHub distributed training launch script
# Parses platform env vars (VC_*) and launches PyTorch DDP training.
#
# Set METHOD env var in bitahub.yaml to choose dense|sr|mb_sr.
# Additional CLI args can be passed after the script name.
#
set -e

# ---- Parse BitaHub environment variables ----
# VC_WORKER_HOSTS: comma-separated list of worker addresses
# VC_WORKER_NUM:   number of worker nodes
# VC_TASK_INDEX:   index of this node within the role (0-based)

MASTER_ADDR=$(echo ${VC_WORKER_HOSTS} | awk -F ',' '{print $1}')
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${VC_WORKER_NUM:-1}
NODE_RANK=${VC_TASK_INDEX:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}

echo "=== BitaHub DDP Config ==="
echo "MASTER_ADDR:    $MASTER_ADDR"
echo "MASTER_PORT:    $MASTER_PORT"
echo "NNODES:         $NNODES"
echo "NODE_RANK:      $NODE_RANK"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "========================="

# ---- Default paths (can be overridden by args) ----
DATA_PATH=${DATA_PATH:-/mnt/data}
MODEL_PATH=${MODEL_PATH:-/mnt/output/models/TinyLlama-1.1B}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/output/checkpoints}
CONVERTED_MODEL=${CONVERTED_MODEL:-/mnt/output/eiporion_converted}

# ---- Launch with torchrun ----
cd /code

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train/continued_pretrain.py \
    --method "${METHOD:-dense}" \
    --model "$MODEL_PATH" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --converted-model "$CONVERTED_MODEL" \
    "$@"
