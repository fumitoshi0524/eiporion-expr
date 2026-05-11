"""Run lm-evaluation-harness benchmarks on all checkpoints.

BitLinear checkpoints (sr/mb_sr) must be exported first:
    python scripts/export_for_eval.py --checkpoint checkpoints/sr/final --output checkpoints/sr/eval
    python scripts/export_for_eval.py --checkpoint checkpoints/mb_sr/final --output checkpoints/mb_sr/eval

Then run:
    python eval/run_benchmarks.py --checkpoints checkpoints/dense/final checkpoints/sr/eval checkpoints/mb_sr/eval checkpoints/gptq checkpoints/awq
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


BENCHMARKS = [
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "mmlu",
    "piqa",
    "winogrande",
    "boolq",
    "wikitext",
    "c4",
]

def run_lm_eval(checkpoint_path, output_dir, tasks=None):
    """Run lm-eval on a single checkpoint."""
    if tasks is None:
        tasks = BENCHMARKS

    os.makedirs(output_dir, exist_ok=True)

    # Skip already-evaluated checkpoints (lm-eval writes results_<timestamp>.json)
    existing_files = list(Path(output_dir).glob("results_*.json"))
    if existing_files:
        latest = max(existing_files, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            existing = json.load(f)
        if existing.get("results"):
            print(f"  Already evaluated, skipping: {checkpoint_path}")
            return existing

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={checkpoint_path},dtype=bfloat16,trust_remote_code=True",
        "--tasks", ",".join(tasks),
        "--batch_size", "auto",
        "--output_path", output_dir,
        "--log_samples",
    ]

    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Read and return results
    results_files = list(Path(output_dir).glob("results_*.json"))
    if results_files:
        latest = max(results_files, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            return json.load(f)

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to model checkpoints")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help=f"Tasks to run (default: {BENCHMARKS})")
    args = parser.parse_args()

    all_results = {}

    for ckpt in args.checkpoints:
        # Infer run name: checkpoints/dense/final → dense, checkpoints/gptq → gptq
        parent = os.path.dirname(ckpt)
        name = os.path.basename(parent)
        if name in ("checkpoints", "results", ""):
            name = os.path.basename(ckpt)
        print(f"\n{'='*50}")
        print(f"Evaluating: {name}")
        print(f"Checkpoint: {ckpt}")
        print(f"{'='*50}")

        try:
            ckpt_output_dir = os.path.join(args.output_dir, name)
            results = run_lm_eval(ckpt, ckpt_output_dir, args.tasks)
            all_results[name] = results
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[name] = {"error": str(e)}

    # Save summary
    summary_path = os.path.join(args.output_dir, "all_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
