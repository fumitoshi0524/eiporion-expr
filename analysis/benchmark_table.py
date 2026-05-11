"""Generate the final LaTeX comparison table for paper-ready output.

Usage:
    python analysis/benchmark_table.py --metrics-json results/comparison/metrics.json --output results/paper_table.tex
"""

import argparse
import json
import os


def format_value(val):
    """Format a metric value for LaTeX."""
    if val is None or val == "-":
        return "-"
    if isinstance(val, str):
        return val
    return f"{val:.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-json", default="results/comparison/metrics.json")
    parser.add_argument("--output", default="results/paper_table.tex")
    args = parser.parse_args()

    with open(args.metrics_json) as f:
        metrics = json.load(f)

    task_order = [
        ("HellaSwag", True), ("ARC-E", True), ("ARC-C", True),
        ("MMLU", True), ("PIQA", True), ("WinoGrande", True),
        ("BoolQ", True), ("WikiText-2", False), ("C4", False),
    ]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Benchmark comparison across training and quantization methods.}")
    lines.append(r"\label{tab:benchmarks}")

    col_spec = "l" + "c" * len(task_order)
    lines.append(r"\begin{tabular}{@{}" + col_spec + r"@{}}")
    lines.append(r"\toprule")

    header = " & ".join([r"\textbf{Method}"] + [t for t, _ in task_order])
    lines.append(f"  {header} \\\\")
    lines.append(r"  \midrule")

    for method, method_metrics in metrics.items():
        row = [method.replace("_", "\\_")]
        for task, _ in task_order:
            val = method_metrics.get(task, "-")
            formatted = format_value(val)
            row.append(formatted)
        lines.append("  " + " & ".join(row) + r" \\")

    lines.append(r"  \bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    output = "\n".join(lines)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(output)

    print(f"LaTeX table saved to: {args.output}")
    print()
    print(output)


if __name__ == "__main__":
    main()
