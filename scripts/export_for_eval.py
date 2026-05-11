"""Export a training checkpoint to a directory that lm-eval can load via HF API.

BitLinear checkpoints (sr/mb_sr) can't be loaded directly by AutoModelForCausalLM
because they contain BitLinear layers. This script bundles a custom model class
that teaches HuggingFace how to load them.

Usage:
    python scripts/export_for_eval.py --checkpoint checkpoints/sr/final --output checkpoints/sr/eval
    python scripts/export_for_eval.py --checkpoint checkpoints/mb_sr/final --output checkpoints/mb_sr/eval
"""

import argparse
import os
import shutil
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from eiporion import BitLinear, quantize_fp_to_int8


MODELING_STUB = '''"""
Custom model class for loading BitLinear-ized TinyLlama checkpoints.
Used with trust_remote_code=True in AutoModelForCausalLM.from_pretrained().
"""
import torch
from transformers import LlamaForCausalLM
from eiporion import BitLinear, quantize_fp_to_int8


def _replace_linears_with_bitlinear(module, block_size=128):
    """Recursively replace nn.Linear with BitLinear."""
    replacements = []
    for name, child in module.named_modules():
        if not isinstance(child, torch.nn.Linear):
            continue
        if "lm_head" in name:
            continue
        parent = module
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        replacements.append((parent, parts[-1], child))

    for parent, attr_name, old_linear in replacements:
        bitlinear = BitLinear(
            old_linear.in_features, old_linear.out_features,
            bias=old_linear.bias is not None,
        )
        setattr(parent, attr_name, bitlinear)

    return len(replacements)


class EiporionLlamaForCausalLM(LlamaForCausalLM):
    """LlamaForCausalLM with BitLinear layers for INT8 weight loading."""

    def __init__(self, config):
        super().__init__(config)
        _replace_linears_with_bitlinear(self, block_size=128)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    ckpt = args.checkpoint
    weights_path = os.path.join(ckpt, "model_weights.pt")
    if not os.path.exists(weights_path):
        # Standard checkpoint — just copy (dense, gptq, awq)
        print(f"Standard checkpoint, copying to: {args.output}")
        for f in os.listdir(ckpt):
            src = os.path.join(ckpt, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(args.output, f))
        return

    print(f"BitLinear checkpoint detected. Loading...")

    # Load config
    config = AutoConfig.from_pretrained(ckpt)
    if config is None:
        # Fallback: load from original model
        config = AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")

    # Build model: standard TinyLlama → BitLinear conversion → load weights
    print("  Building model from config...")
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

    # Convert Linear → BitLinear
    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if "lm_head" in name:
            continue
        parent = model
        for part in name.split(".")[:-1]:
            parent = getattr(parent, part)
        replacements.append((parent, name.split(".")[-1], module))

    for parent, attr_name, old_linear in replacements:
        bitlinear = BitLinear(
            old_linear.in_features, old_linear.out_features,
            bias=old_linear.bias is not None,
        )
        setattr(parent, attr_name, bitlinear)

    print(f"  Replaced {len(replacements)} Linear → BitLinear")

    # Load trained weights
    print(f"  Loading weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    # Save with custom model class
    print(f"  Saving to: {args.output}")

    # Write the custom modeling stub
    with open(os.path.join(args.output, "modeling_eiporion_load.py"), "w") as f:
        f.write(MODELING_STUB)

    # Update config to use custom class
    config.auto_map = {
        "AutoModelForCausalLM": "modeling_eiporion_load.EiporionLlamaForCausalLM"
    }
    config.save_pretrained(args.output)

    # Save model weights in safetensors format
    model.save_pretrained(args.output)

    # Copy tokenizer
    tokenizer = None
    for src_dir in [ckpt, "checkpoints/eiporion_converted",
                    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(src_dir)
            break
        except Exception:
            continue
    if tokenizer:
        tokenizer.save_pretrained(args.output)

    print("Done. Load with: AutoModelForCausalLM.from_pretrained(output, trust_remote_code=True)")


if __name__ == "__main__":
    main()
