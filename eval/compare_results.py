"""Generate comparison tables and plots from benchmark results.

Usage:
    python eval/compare_results.py --results results/all_results.json --output results/comparison
"""

import argparse
import json
import os


def load_results(results_path):
    """Load results from lm-eval json."""
    with open(results_path) as f:
        return json.load(f)


def _find_metric(task_data, preferred_keys):
    """Find a metric value by trying multiple possible keys."""
    for key in preferred_keys:
        val = task_data.get(key)
        if val is not None:
            return val
    # Log available keys for debugging
    print(f"    Warning: none of {preferred_keys} found. Available keys: {list(task_data.keys())[:5]}")
    return None


def extract_metrics(all_results):
    """Extract key metrics from lm-eval results into a comparison table."""
    # Keys vary across lm-eval versions: older use "acc,none", newer use "acc,none,filter=None"
    ACC_KEYS = ["acc,none", "acc_norm,none", "acc,none,filter=None", "acc_norm,none,filter=None"]
    PPL_KEYS = ["word_perplexity,none", "perplexity,none",
                "word_perplexity,none,filter=None", "perplexity,none,filter=None"]

    metrics = {}
    task_rename = {
        "hellaswag": "HellaSwag",
        "arc_easy": "ARC-E",
        "arc_challenge": "ARC-C",
        "mmlu": "MMLU",
        "piqa": "PIQA",
        "winogrande": "WinoGrande",
        "boolq": "BoolQ",
        "wikitext": "WikiText-2",
        "c4": "C4",
    }

    for model_name, result_data in all_results.items():
        if "error" in result_data:
            metrics[model_name] = {"error": result_data["error"]}
            continue

        model_metrics = {}
        results = result_data.get("results", result_data)

        for task_key, task_display in task_rename.items():
            task_data = results.get(task_key, {})
            if not task_data:
                continue

            if task_key in ("wikitext", "c4"):
                metric_val = _find_metric(task_data, PPL_KEYS)
                if metric_val is not None:
                    model_metrics[task_display] = round(metric_val, 2)
            else:
                metric_val = _find_metric(task_data, ACC_KEYS)
                if metric_val is not None:
                    model_metrics[task_display] = round(metric_val * 100, 2)

        metrics[model_name] = model_metrics

    return metrics


def build_markdown_table(metrics, output_path):
    """Generate markdown comparison table."""
    all_tasks = set()
    for model_metrics in metrics.values():
        all_tasks.update(model_metrics.keys())
    all_tasks = sorted(all_tasks)

    lines = ["# Benchmark Results\n"]
    lines.append("| Model | " + " | ".join(all_tasks) + " |")
    lines.append("|" + "|".join(["------"] * (len(all_tasks) + 1)) + "|")

    for model_name, model_metrics in metrics.items():
        row = [model_name]
        for task in all_tasks:
            val = model_metrics.get(task, "-")
            row.append(str(val))
        lines.append("| " + " | ".join(row) + " |")

    with open(os.path.join(output_path, "benchmark_table.md"), "w") as f:
        f.write("\n".join(lines))

    print(f"  Markdown table saved to: {output_path}/benchmark_table.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/all_results.json")
    parser.add_argument("--output", default="results/comparison")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    all_results = load_results(args.results)
    metrics = extract_metrics(all_results)

    print("\nExtracted metrics:")
    for model, model_metrics in metrics.items():
        print(f"\n  {model}:")
        for task, val in model_metrics.items():
            print(f"    {task}: {val}")

    build_markdown_table(metrics, args.output)

    # Also save as clean JSON
    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nComparison data saved to: {args.output}")


if __name__ == "__main__":
    main()
