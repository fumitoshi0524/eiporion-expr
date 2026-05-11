"""Generate memory and throughput comparison from training JSONL logs.

Usage:
    python analysis/efficiency.py \
        --log-files checkpoints/dense/train_log.jsonl \
                     checkpoints/sr/train_log.jsonl \
                     checkpoints/mb_sr/train_log.jsonl \
        --output results/efficiency.md
"""

import argparse
import json
import os
import numpy as np


def load_efficiency_stats(log_path):
    """Extract throughput and memory stats from a JSONL training log."""
    tokens_per_sec = []
    peak_mem_mb = []

    with open(log_path) as f:
        for line in f:
            entry = json.loads(line.strip() or "{}")
            if not entry:
                continue
            if "train/tokens_per_sec" in entry:
                tokens_per_sec.append(entry["train/tokens_per_sec"])
            if "system/peak_gpu_memory_mb" in entry:
                peak_mem_mb.append(entry["system/peak_gpu_memory_mb"])

    # Use last 20% of logged values (steady state, after warmup)
    n = len(tokens_per_sec)
    start = n * 4 // 5 if n > 10 else 0

    avg_tps = np.mean(tokens_per_sec[start:]) if tokens_per_sec else 0
    avg_mem = np.mean(peak_mem_mb[start:]) if peak_mem_mb else 0

    return {
        "avg_tokens_per_sec": round(avg_tps, 0),
        "avg_gpu_memory_mb": round(avg_mem, 0),
        "avg_gpu_memory_gb": round(avg_mem / 1024, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-files", nargs="+", required=True,
                        help="JSONL training log files")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Display names")
    parser.add_argument("--output", default="results/efficiency.md")
    args = parser.parse_args()

    if args.names is None:
        args.names = []
        for p in args.log_files:
            parts = os.path.normpath(p).split(os.sep)
            args.names.append(parts[-2] if len(parts) >= 2 else os.path.basename(p))

    stats = {}
    for name, log_file in zip(args.names, args.log_files):
        if os.path.exists(log_file):
            stats[name] = load_efficiency_stats(log_file)
            print(f"  {name}: {stats[name]['avg_tokens_per_sec']:.0f} tok/s, "
                  f"{stats[name]['avg_gpu_memory_gb']:.1f} GB")
        else:
            print(f"  {name}: log file not found ({log_file})")
            stats[name] = {"avg_tokens_per_sec": "N/A", "avg_gpu_memory_gb": "N/A"}

    # Build markdown table
    lines = ["# Training Efficiency Comparison\n"]
    lines.append("| Method | Throughput (tok/s) | GPU Memory (GB) | Memory vs Dense |")
    lines.append("|--------|-------------------|-----------------|-----------------|")

    dense_mem = None
    if "dense" in stats and isinstance(stats["dense"].get("avg_gpu_memory_gb"), (int, float)):
        dense_mem = stats["dense"]["avg_gpu_memory_gb"]

    for name in args.names:
        s = stats.get(name, {})
        tps = s.get("avg_tokens_per_sec", "N/A")
        mem = s.get("avg_gpu_memory_gb", "N/A")

        mem_vs = "-"
        if dense_mem and isinstance(mem, (int, float)):
            ratio = (1 - mem / dense_mem) * 100
            mem_vs = f"-{ratio:.0f}%"

        lines.append(f"| {name} | {tps} | {mem} | {mem_vs} |")

    lines.append("")
    lines.append("*Throughput and memory are steady-state averages (final 20% of logged steps).*\n")
    lines.append("*Memory reduction is relative to the dense BF16 baseline.*\n")

    output = "\n".join(lines)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(output)

    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
