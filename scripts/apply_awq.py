"""Apply AWQ post-training quantization to the dense checkpoint.

Usage:
    python scripts/apply_awq.py \
        --model checkpoints/dense/final \
        --output checkpoints/awq \
        --bits 8 \
        --calibration-data data/calibration.json
"""

import argparse
import torch
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to dense model checkpoint")
    parser.add_argument("--output", required=True, help="Output path for AWQ model")
    parser.add_argument("--bits", type=int, default=8,
                        choices=[4, 8], help="Quantization bit-width")
    parser.add_argument("--calibration-data", help="Path to calibration data")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        print("autoawq not installed. Install with: pip install autoawq")
        print("Skipping AWQ quantization.")
        return

    # Load calibration data
    if args.calibration_data:
        from datasets import load_dataset
        dataset = load_dataset("json", data_files=args.calibration_data, split="train")
    else:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding=False,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 0)

    calibration_data = []
    for i, sample in enumerate(tokenized):
        if i >= 128:
            break
        calibration_data.append(tokenizer.decode(sample["input_ids"]))

    print(f"Loading as AutoAWQForCausalLM...")
    model = AutoAWQForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )

    quant_config = {
        "zero_point": True,
        "q_group_size": args.group_size,
        "w_bit": args.bits,
        "version": "GEMM",
    }

    print(f"Running AWQ quantization with {len(calibration_data)} calibration samples...")
    model.quantize(
        tokenizer=tokenizer,
        quant_config=quant_config,
        calib_data=calibration_data,
    )

    print(f"Saving AWQ model to: {args.output}")
    model.save_quantized(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
