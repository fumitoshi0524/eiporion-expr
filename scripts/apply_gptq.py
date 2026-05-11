"""Apply GPTQ post-training quantization to the dense checkpoint.

Usage:
    python scripts/apply_gptq.py \
        --model checkpoints/dense/final \
        --output checkpoints/gptq \
        --bits 8 \
        --calibration-data data/calibration.json
"""

import argparse
import torch
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to dense model checkpoint")
    parser.add_argument("--output", required=True, help="Output path for GPTQ model")
    parser.add_argument("--bits", type=int, default=8,
                        choices=[4, 8], help="Quantization bit-width")
    parser.add_argument("--calibration-data", help="Path to calibration data (json/parquet)")
    parser.add_argument("--group-size", type=int, default=128,
                        help="Group size for quantization")
    parser.add_argument("--damp-percent", type=float, default=0.01)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except ImportError:
        print("auto-gptq not installed. Install with: pip install auto-gptq")
        print("Skipping GPTQ quantization.")
        return

    # Load calibration data
    if args.calibration_data:
        from datasets import load_dataset
        dataset = load_dataset("json", data_files=args.calibration_data, split="train")
    else:
        # Use wikitext as default calibration
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        # Filter empty strings
        dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

    # Tokenize calibration data
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding=False,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 0)

    quant_config = BaseQuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        damp_percent=args.damp_percent,
        desc_act=False,
        sym=True,
    )

    print(f"Loading as AutoGPTQForCausalLM...")
    model = AutoGPTQForCausalLM.from_pretrained(
        args.model,
        quant_config,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )

    # Prepare calibration examples
    calibration_examples = []
    for i, sample in enumerate(tokenized):
        if i >= 128:  # 128 samples is enough for calibration
            break
        calibration_examples.append(tokenizer.decode(sample["input_ids"]))

    print(f"Running GPTQ quantization with {len(calibration_examples)} calibration samples...")
    model.quantize(calibration_examples, batch_size=1)

    print(f"Saving GPTQ model to: {args.output}")
    model.save_quantized(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
