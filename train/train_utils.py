"""Utilities for the continued pretraining experiment."""

import os
import torch
from datasets import load_dataset


def save_checkpoint(model, optimizer, scheduler, step, path, save_hf=False):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model_weights.pt"))
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
    }, os.path.join(path, "training_state.pt"))
    if save_hf:
        model.save_pretrained(path)
    else:
        model.config.save_pretrained(path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, scheduler, path):
    weights_path = os.path.join(path, "model_weights.pt")
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    state = torch.load(os.path.join(path, "training_state.pt"), map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    step = state["step"]
    print(f"  Resumed from step {step}")
    return step


def load_pretrain_dataset(data_path, tokenizer, seq_length, split_size=None):
    if os.path.isdir(data_path):
        from pathlib import Path
        root = Path(data_path)
        jsonl_files = sorted(root.rglob("*.jsonl"))
        parquet_files = sorted(root.rglob("*.parquet"))
        json_files = sorted(root.rglob("*.json"))

        if jsonl_files:
            # Prefer JSONL shards when present. Some datasets also ship metadata
            # JSON files in the same directory, which break schema casting.
            paths = [str(p) for p in jsonl_files]
            dataset = load_dataset("json", data_files=paths, split="train")
        elif parquet_files:
            paths = [str(p) for p in parquet_files]
            dataset = load_dataset("parquet", data_files=paths, split="train")
        elif json_files:
            metadata_json_names = {"dataset_info.json", "dataset_infos.json"}
            data_json_files = [p for p in json_files if p.name.lower() not in metadata_json_names]
            if not data_json_files:
                raise FileNotFoundError(f"No data JSON files found in {data_path}")
            paths = [str(p) for p in data_json_files]
            dataset = load_dataset("json", data_files=paths, split="train")
        else:
            raise FileNotFoundError(f"No data files found in {data_path}")
    else:
        dataset = load_dataset(data_path, split="train")

    if split_size:
        try:
            if len(dataset) > split_size:
                dataset = dataset.select(range(split_size))
        except TypeError:
            dataset = dataset.take(split_size)

    text_column = "text" if "text" in dataset.column_names else None
    if text_column is None:
        for candidate in ("content", "default"):
            if candidate in dataset.column_names:
                text_column = candidate
                break
    if text_column is None:
        raise ValueError(
            f"Unable to find a text column. Available columns: {dataset.column_names}"
        )

    def tokenize_fn(examples):
        texts = examples[text_column]
        tokenized = tokenizer(
            texts, truncation=True, max_length=seq_length + 1,
            padding=False, return_attention_mask=False,
        )
        input_ids_list = []
        labels_list = []
        for ids in tokenized["input_ids"]:
            if len(ids) < seq_length + 1:
                ids = ids + [tokenizer.pad_token_id or 0] * (seq_length + 1 - len(ids))
            if len(ids) >= seq_length + 1:
                input_ids_list.append(ids[:seq_length])
                labels_list.append(ids[1:seq_length + 1])
        return {"input_ids": input_ids_list, "labels": labels_list}

    tokenized_dataset = dataset.map(
        tokenize_fn, batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    return tokenized_dataset
