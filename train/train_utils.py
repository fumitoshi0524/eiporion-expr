"""Utilities for the continued pretraining experiment."""

import os
import torch
from datasets import load_dataset


def save_checkpoint(model, optimizer, scheduler, step, path, save_hf=False):
    """Save full training state.

    Always saves model_weights.pt + training_state.pt for resume.
    If save_hf=True (dense models), also saves in HF format for direct lm-eval loading.
    """
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model_weights.pt"))
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
    }, os.path.join(path, "training_state.pt"))
    if save_hf:
        model.save_pretrained(path)  # safetensors + config.json
    else:
        model.config.save_pretrained(path)  # config.json only, needs export for eval
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, scheduler, path):
    """Load full training state. Restores model weights, optimizer, scheduler.
    Returns step number.
    """
    # Restore model weights first
    weights_path = os.path.join(path, "model_weights.pt")
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Warning: missing keys in checkpoint: {missing}")
        if unexpected:
            print(f"  Warning: unexpected keys in checkpoint: {unexpected}")
    # Restore training state
    state = torch.load(os.path.join(path, "training_state.pt"), map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    step = state["step"]
    print(f"  Resumed from step {step}")
    return step


def load_pretrain_dataset(data_path, tokenizer, seq_length, split_size=None):
    """Load and tokenize a pretraining dataset. Returns a tokenized HuggingFace Dataset.

    data_path can be:
      - A local directory with .json/.jsonl/.parquet files
      - A HuggingFace dataset path like "HuggingFaceFW/fineweb-edu" with optional config
    """
    # Try HF dataset name first (e.g. "HuggingFaceFW/fineweb-edu:CC-MAIN-2024-10")
    if "/" in data_path and not os.path.isdir(data_path):
        parts = data_path.split(":", 1)
        ds_name = parts[0]
        ds_config = parts[1] if len(parts) > 1 else None
        kwargs = {"path": data_path, "split": "train", "streaming": True}
        if ds_config:
            kwargs = {"path": ds_name, "name": ds_config, "split": "train", "streaming": True}
        dataset = load_dataset(**kwargs)
    elif os.path.isdir(data_path):
        # Local directory — try json, jsonl, then parquet
        files = os.listdir(data_path)
        json_files = [f for f in files if f.endswith(('.json', '.jsonl'))]
        parquet_files = [f for f in files if f.endswith('.parquet')]
        if json_files:
            paths = [os.path.join(data_path, f) for f in json_files]
            dataset = load_dataset("json", data_files=paths, split="train")
        elif parquet_files:
            paths = [os.path.join(data_path, f) for f in parquet_files]
            dataset = load_dataset("parquet", data_files=paths, split="train")
        else:
            raise FileNotFoundError(f"No .json/.jsonl/.parquet files in {data_path}")
    else:
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    if split_size and len(dataset) > split_size:
        dataset = dataset.select(range(split_size))

    def tokenize_fn(examples):
        texts = examples["text"]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=seq_length + 1,
            padding=False,
            return_attention_mask=False,
        )
        result = {"input_ids": [], "labels": []}
        for ids in tokenized["input_ids"]:
            if len(ids) < seq_length + 1:
                ids = ids + [tokenizer.pad_token_id or 0] * (seq_length + 1 - len(ids))
            result["input_ids"].append(ids[:-1])
            result["labels"].append(ids[1:])
        return result

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    tokenized_dataset = tokenized_dataset.filter(
        lambda x: len(x["input_ids"]) == seq_length
    )

    return tokenized_dataset
