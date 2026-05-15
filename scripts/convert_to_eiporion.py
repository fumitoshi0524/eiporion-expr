"""Convert TinyLlama-1.1B checkpoint to eiporion BitLinear format.

Usage:
    python scripts/convert_to_eiporion.py \
        --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
        --output checkpoints/eiporion_converted \
        --block-size 128
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM
from eiporion import BitLinear, quantize_fp_to_int8


def replace_linears_with_bitlinear(model, block_size=128, keep_lm_head=True):
    """Replace all nn.Linear with BitLinear, quantizing weights to INT8.

    Uses named_modules() to find Linear layers, then navigates to the parent
    module to perform the replacement safely.
    """
    # Collect replacements first (can't modify during iteration)
    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if keep_lm_head and "lm_head" in name:
            continue

        # Find parent module
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)

        attr_name = parts[-1]
        replacements.append((parent, attr_name, module))

    # Apply replacements
    replaced_count = 0
    for parent, attr_name, old_linear in replacements:
        in_features = old_linear.in_features
        out_features = old_linear.out_features
        has_bias = old_linear.bias is not None

        # Quantize FP32 weights to INT8
        weight_fp = old_linear.weight.data.float()
        weight_int8, row_scales = quantize_fp_to_int8(weight_fp)

        # Create BitLinear and load quantized weights
        bitlinear = BitLinear(in_features, out_features, bias=has_bias)
        bitlinear.int_weight.data.copy_(weight_int8)
        bitlinear.weight_scale.data.copy_(row_scales)
        if has_bias:
            bitlinear.bias.data.copy_(old_linear.bias.data)

        setattr(parent, attr_name, bitlinear)
        replaced_count += 1

    return replaced_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--output", default="checkpoints/eiporion_converted")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--convert-lm-head", action="store_true",
                        help="Also convert lm_head to BitLinear (not recommended)")
    args = parser.parse_args()

    keep_lm_head = not args.convert_lm_head

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    # Count before
    linear_before = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    print(f"  nn.Linear layers before conversion: {linear_before}")

    print("Converting nn.Linear -> BitLinear...")
    replaced = replace_linears_with_bitlinear(model, args.block_size, keep_lm_head)

    # Verify
    linear_after = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    bitlinear_count = sum(1 for m in model.modules() if isinstance(m, BitLinear))
    total_params = sum(p.numel() for p in model.parameters())

    print(f"  Replaced: {replaced} Linear -> BitLinear")
    print(f"  Remaining nn.Linear: {linear_after}")
    print(f"  BitLinear layers: {bitlinear_count}")
    print(f"  Total parameters: {total_params:,}")

    # Show which Linear layers remain (should only be lm_head)
    if linear_after > 0:
        print("  Remaining Linear layers:")
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                print(f"    {name}  [{m.in_features} -> {m.out_features}]")

    print(f"\nSaving converted weights to: {args.output}")
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "eiporion_weights.pt")
    torch.save(model.state_dict(), out_path)
    print(f"  Saved: {out_path}")

    # Also save tokenizer for convenience
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.save_pretrained(args.output)

    print("Done.")


if __name__ == "__main__":
    main()
