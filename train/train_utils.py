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


def load_slimpajama(data_path, tokenizer, seq_length, split_size=None):
    """Load and tokenize SlimPajama dataset. Returns a tokenized HuggingFace Dataset.

    Expects data in HuggingFace datasets format (json/parquet files).
    Caller is responsible for creating DataLoaders.
    """
    try:
        dataset = load_dataset("json", data_files=data_path, split="train")
    except Exception:
        try:
            dataset = load_dataset("parquet", data_files=f"{data_path}/*.parquet", split="train")
        except Exception:
            raise FileNotFoundError(
                f"Cannot load dataset from {data_path}. "
                "Place SlimPajama .json or .parquet files in this directory."
            )

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
