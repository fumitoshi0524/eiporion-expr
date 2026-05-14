"""Compute directional deviation between training trajectories from checkpoints.

Directional deviation is defined as:
    DD(A, B) = 1 - cos(DeltaA, DeltaB)

where DeltaA and DeltaB are parameter update vectors between two checkpoints
for methods A and B (for example step_1000 -> step_2000).

Usage:
    python analysis/directional_deviation.py --root checkpoints --start step_1000 --end step_2000
    python analysis/directional_deviation.py --root checkpoints --start step_1000 --end final --pairs dense:sr dense:mb_sr
"""

import argparse
import gc
import json
import math
import os
from typing import Dict, Iterable, List, Tuple

import torch


def _parse_pairs(items: Iterable[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid pair format '{item}'. Expected METHOD_A:METHOD_B")
        left, right = item.split(":", 1)
        left = left.strip()
        right = right.strip()
        if not left or not right:
            raise ValueError(f"Invalid pair format '{item}'. Expected METHOD_A:METHOD_B")
        pairs.append((left, right))
    return pairs


def _default_pairs(methods: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            out.append((methods[i], methods[j]))
    return out


def _checkpoint_weights_path(root: str, method: str, label: str) -> str:
    return os.path.join(root, method, label, "model_weights.pt")


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint weights not found: {path}")
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint format at {path}")
    return state


def _compute_pair_metrics(
    a_start: Dict[str, torch.Tensor],
    a_end: Dict[str, torch.Tensor],
    b_start: Dict[str, torch.Tensor],
    b_end: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    dot = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    used_keys = 0
    used_params = 0
    total_shared_keys = 0
    total_shared_params = 0

    shared_keys = set(a_start.keys()) & set(a_end.keys()) & set(b_start.keys()) & set(b_end.keys())
    for key in shared_keys:
        a0 = a_start[key]
        a1 = a_end[key]
        b0 = b_start[key]
        b1 = b_end[key]

        if not (torch.is_tensor(a0) and torch.is_tensor(a1) and torch.is_tensor(b0) and torch.is_tensor(b1)):
            continue
        if a0.shape != a1.shape or b0.shape != b1.shape or a0.shape != b0.shape:
            continue

        numel = a0.numel()
        total_shared_keys += 1
        total_shared_params += numel

        if not (torch.is_floating_point(a0) and torch.is_floating_point(a1)
                and torch.is_floating_point(b0) and torch.is_floating_point(b1)):
            continue

        da = (a1 - a0).double().reshape(-1)
        db = (b1 - b0).double().reshape(-1)

        dot += torch.dot(da, db).item()
        norm_a_sq += torch.dot(da, da).item()
        norm_b_sq += torch.dot(db, db).item()
        used_keys += 1
        used_params += numel

    if used_keys == 0:
        raise ValueError("No shared floating-point parameter tensors found for comparison.")

    norm_a = math.sqrt(norm_a_sq)
    norm_b = math.sqrt(norm_b_sq)
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("One update vector has zero norm; cosine similarity is undefined.")

    cosine = dot / (norm_a * norm_b)
    cosine = max(-1.0, min(1.0, cosine))
    deviation = 1.0 - cosine

    return {
        "cosine_similarity": cosine,
        "directional_deviation": deviation,
        "shared_keys_total": total_shared_keys,
        "shared_keys_used": used_keys,
        "shared_keys_used_ratio": used_keys / total_shared_keys if total_shared_keys else 0.0,
        "shared_params_total": total_shared_params,
        "shared_params_used": used_params,
        "shared_params_used_ratio": used_params / total_shared_params if total_shared_params else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="checkpoints", help="Checkpoint root directory")
    parser.add_argument("--start", default="step_1000",
                        help="Start checkpoint label (e.g., step_1000)")
    parser.add_argument("--end", default="step_2000",
                        help="End checkpoint label (e.g., step_2000 or final)")
    parser.add_argument("--methods", nargs="+", default=["dense", "sr", "mb_sr"],
                        help="Method names under root/")
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="Optional explicit method pairs: dense:sr dense:mb_sr")
    parser.add_argument("--output-json", default=None,
                        help="Optional output JSON path")
    args = parser.parse_args()

    pairs = _parse_pairs(args.pairs) if args.pairs else _default_pairs(args.methods)
    if not pairs:
        raise ValueError("No method pairs to compare.")

    results = []
    for left, right in pairs:
        left_start_path = _checkpoint_weights_path(args.root, left, args.start)
        left_end_path = _checkpoint_weights_path(args.root, left, args.end)
        right_start_path = _checkpoint_weights_path(args.root, right, args.start)
        right_end_path = _checkpoint_weights_path(args.root, right, args.end)

        print(f"\nComparing {left} vs {right} ({args.start} -> {args.end})")
        print(f"  {left} start: {left_start_path}")
        print(f"  {left} end:   {left_end_path}")
        print(f"  {right} start: {right_start_path}")
        print(f"  {right} end:   {right_end_path}")

        left_start = _load_state_dict(left_start_path)
        left_end = _load_state_dict(left_end_path)
        right_start = _load_state_dict(right_start_path)
        right_end = _load_state_dict(right_end_path)

        metrics = _compute_pair_metrics(left_start, left_end, right_start, right_end)
        metrics["pair"] = f"{left}:{right}"
        results.append(metrics)

        print(f"  cosine_similarity:     {metrics['cosine_similarity']:.6f}")
        print(f"  directional_deviation: {metrics['directional_deviation']:.6f}")
        print(
            f"  used keys/total keys:  {metrics['shared_keys_used']}/{metrics['shared_keys_total']} "
            f"({metrics['shared_keys_used_ratio'] * 100:.2f}%)"
        )
        print(
            f"  used params/total:     {metrics['shared_params_used']}/{metrics['shared_params_total']} "
            f"({metrics['shared_params_used_ratio'] * 100:.2f}%)"
        )

        del left_start, left_end, right_start, right_end
        gc.collect()

    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(
                {
                    "root": args.root,
                    "start": args.start,
                    "end": args.end,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\nSaved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
