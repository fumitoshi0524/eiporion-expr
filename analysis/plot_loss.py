"""Plot training loss curves from JSONL log files.

Usage:
    python analysis/plot_loss.py \
        --log-files checkpoints/dense/train_log.jsonl \
                     checkpoints/sr/train_log.jsonl \
                     checkpoints/mb_sr/train_log.jsonl \
        --output results/loss_comparison.png
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt


def load_jsonl_log(path):
    """Extract step and loss from a JSONL training log."""
    steps, train_loss, val_steps, val_loss = [], [], [], []
    tokens_per_sec = []
    peak_mem = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            step = entry["step"]

            if "train/loss" in entry:
                steps.append(step)
                train_loss.append(entry["train/loss"])

            if "train/tokens_per_sec" in entry:
                tokens_per_sec.append(entry["train/tokens_per_sec"])

            if "system/peak_gpu_memory_mb" in entry:
                peak_mem.append(entry["system/peak_gpu_memory_mb"])

            if "eval/val_loss" in entry:
                val_steps.append(step)
                val_loss.append(entry["eval/val_loss"])

    return steps, train_loss, val_steps, val_loss, tokens_per_sec, peak_mem


def smooth(values, window_frac=50):
    """Apply moving average smoothing."""
    if len(values) < 2:
        return values
    window = max(1, len(values) // window_frac)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-files", nargs="+", required=True,
                        help="JSONL log files from training runs")
    parser.add_argument("--output", default="results/loss_comparison.png")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Display names (default: inferred from path)")
    parser.add_argument("--token-budget", type=float, default=5e9,
                        help="Total token budget for x-axis label")
    args = parser.parse_args()

    if args.names is None:
        args.names = []
        for p in args.log_files:
            parts = os.path.normpath(p).split(os.sep)
            # e.g. checkpoints/dense/train_log.jsonl -> dense
            name = parts[-2] if len(parts) >= 2 else os.path.basename(p)
            args.names.append(name)

    colors = {"dense": "#1f77b4", "sr": "#ff7f0e", "mb_sr": "#2ca02c",
              "gptq": "#d62728", "awq": "#9467bd"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Subplot 1: Training loss (raw) ---
    ax = axes[0, 0]
    for log_file, name in zip(args.log_files, args.names):
        steps, train_loss, val_steps, val_loss, _, _ = load_jsonl_log(log_file)
        if not steps:
            print(f"  No data in {log_file}")
            continue
        color = colors.get(name, None)
        ax.plot(steps, train_loss, label=name, color=color, alpha=0.6, linewidth=0.5)
        # Validation loss as markers
        if val_steps:
            ax.plot(val_steps, val_loss, 'o', color=color, markersize=4, alpha=0.9)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss (raw, dots = validation)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Subplot 2: Training loss (smoothed) ---
    ax = axes[0, 1]
    for log_file, name in zip(args.log_files, args.names):
        steps, train_loss, val_steps, val_loss, _, _ = load_jsonl_log(log_file)
        if not steps or len(train_loss) < 2:
            continue
        color = colors.get(name, None)
        smoothed = smooth(train_loss)
        offset = len(steps) - len(smoothed)
        ax.plot(steps[offset:], smoothed, label=name, color=color, linewidth=1.2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss (smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Subplot 3: Throughput ---
    ax = axes[1, 0]
    for log_file, name in zip(args.log_files, args.names):
        steps, _, _, _, tok_per_sec, _ = load_jsonl_log(log_file)
        if not steps:
            continue
        color = colors.get(name, None)
        # tok_per_sec is same length as train_loss (logged at same intervals)
        ax.plot(steps[:len(tok_per_sec)], tok_per_sec, label=name, color=color,
                linewidth=0.8, alpha=0.8)
        avg_tps = np.mean(tok_per_sec[-10:]) if len(tok_per_sec) >= 10 else np.mean(tok_per_sec)
        ax.axhline(y=avg_tps, color=color, linestyle="--", linewidth=0.8, alpha=0.4)

    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Training Throughput")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Subplot 4: GPU Memory ---
    ax = axes[1, 1]
    for log_file, name in zip(args.log_files, args.names):
        steps, _, _, _, _, peak_mem = load_jsonl_log(log_file)
        if not steps:
            continue
        color = colors.get(name, None)
        ax.plot(steps[:len(peak_mem)], peak_mem, label=name, color=color, linewidth=0.8)
        avg_mem = np.mean(peak_mem)
        ax.axhline(y=avg_mem, color=color, linestyle="--", linewidth=0.8, alpha=0.4)
        # Annotate average
        ax.text(steps[-1] if steps else 0, avg_mem,
                f" {avg_mem:.0f}MB", fontsize=8, color=color, va="center")

    ax.set_xlabel("Step")
    ax.set_ylabel("GPU Memory (MB)")
    ax.set_title("Peak GPU Memory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
