"""Continued pretraining comparison: Dense BF16 vs SR vs MB-SR on TinyLlama-1.1B.

Usage:
    python train/continued_pretrain.py --method dense
    python train/continued_pretrain.py --method sr
    python train/continued_pretrain.py --method mb_sr
"""

import argparse
import json
import math
import os
import time
import torch
import swanlab
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

from eiporion import EiporionOptim, EiporionOptimSR, collect_bitlinear_modules, quantize_fp_to_int8, BitLinear
from train_utils import (
    load_slimpajama,
    save_checkpoint,
    load_checkpoint,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["dense", "sr", "mb_sr"])
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--converted-model", default="checkpoints/eiporion_converted")
    parser.add_argument("--data-path", default="data/slimpajama")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--target-batch-size", type=int, default=1_048_576)
    parser.add_argument("--total-tokens", type=int, default=5_000_000_000)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--sr-bias-scale", type=float, default=0.01)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--from-checkpoint", default=None)
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-samples", type=int, default=50,
                        help="Number of validation batches per eval")
    parser.add_argument("--project-name", default="eiporion-expr")
    return parser.parse_args()


def auto_batch_size(model, seq_length, device):
    """Find maximum batch size that fits in GPU memory.

    Probes increasing batch sizes until OOM, then backs off to 80% of the last
    successful batch. Much more accurate than linear extrapolation because model
    weights consume a large fixed memory cost.
    """
    print("Profiling memory to determine batch size...")
    model.to(device)
    total_mb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
    target_mb = total_mb * 0.80

    def try_batch(size):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        input_ids = torch.randint(0, 32000, (size, seq_length), device=device)
        labels = input_ids.clone()
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)
        outputs.loss.backward()
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        torch.cuda.empty_cache()
        return peak

    # Probe: 1, 2, 4, 8, 16, 32, 64, 128, 256
    feasible = 1
    for candidate in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        try:
            peak = try_batch(candidate)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
        if peak > target_mb:
            break
        feasible = candidate

    model.to("cpu")
    torch.cuda.empty_cache()

    print(f"  Max micro batch size: {feasible} (target <{target_mb:.0f}MB, total {total_mb:.0f}MB)")
    return feasible


def _replace_linears(model, block_size, keep_lm_head=True):
    """Replace nn.Linear with BitLinear in-place. Quantizes weights to INT8."""
    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if keep_lm_head and "lm_head" in name:
            continue
        parent = model
        for part in name.split(".")[:-1]:
            parent = getattr(parent, part)
        replacements.append((parent, name.split(".")[-1], module))

    for parent, attr_name, old_linear in replacements:
        weight_fp = old_linear.weight.data.float()
        weight_int8, row_scales = quantize_fp_to_int8(weight_fp, block_size)
        bitlinear = BitLinear(old_linear.in_features, old_linear.out_features,
                              bias=old_linear.bias is not None)
        bitlinear.weight.data.copy_(weight_int8)
        bitlinear.weight_scale.data.copy_(row_scales)
        if old_linear.bias is not None:
            bitlinear.bias.data.copy_(old_linear.bias.data)
        setattr(parent, attr_name, bitlinear)

    return len(replacements)


def build_model_and_optimizer(args, device):
    """Load model and create optimizer based on method."""
    if args.method == "dense":
        print("Loading dense TinyLlama model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
        )
        model = model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
        bit_modules = None
    else:
        # Load standard TinyLlama, convert Linear → BitLinear, load quantized weights
        print(f"Loading standard TinyLlama for conversion...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
        )

        # Convert nn.Linear → BitLinear (lm_head stays as fp32 Linear)
        _replace_linears(model, args.block_size, keep_lm_head=True)

        # Load pre-quantized INT8 weights
        weights_path = os.path.join(args.converted_model, "eiporion_weights.pt")
        print(f"Loading quantized weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Warning: missing keys: {len(missing)}")
        if unexpected:
            print(f"  Warning: unexpected keys: {len(unexpected)}")

        model = model.to(device)
        bit_modules = collect_bitlinear_modules(model)
        print(f"  Found {len(bit_modules)} BitLinear modules")

        if args.method == "sr":
            optimizer = EiporionOptimSR(
                model.parameters(),
                lr=args.lr,
                bit_modules=bit_modules,
                block_size=args.block_size,
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = EiporionOptim(
                model.parameters(),
                lr=args.lr,
                bit_modules=bit_modules,
                block_size=args.block_size,
                sr_bias_scale=args.sr_bias_scale,
                weight_decay=args.weight_decay,
            )

    return model, optimizer, bit_modules


@torch.no_grad()
def validate(model, val_loader, device, max_batches):
    """Compute validation loss and perplexity."""
    model.eval()
    total_loss = 0.0
    count = 0

    for batch in val_loader:
        if count >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = input_ids.clone()
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)
        total_loss += outputs.loss.item()
        count += 1

    model.train()
    avg_loss = total_loss / count if count > 0 else float("inf")
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    return avg_loss, ppl


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model, optimizer, bit_modules = build_model_and_optimizer(args, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Auto-detect batch size
    if args.batch_size is None:
        args.batch_size = auto_batch_size(model, args.seq_length, device)
    model = model.to(device)

    tokens_per_micro_batch = args.batch_size * args.seq_length
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = max(1, args.target_batch_size // tokens_per_micro_batch)
    effective_batch_tokens = args.batch_size * args.seq_length * args.gradient_accumulation_steps

    total_steps = args.total_tokens // effective_batch_tokens
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    # Dataset — load and split BEFORE creating DataLoaders
    print("Loading SlimPajama dataset...")
    tokenized_dataset = load_slimpajama(
        args.data_path, tokenizer, args.seq_length,
        split_size=args.total_tokens // args.seq_length,
    )
    # Split off validation
    train_size = int(0.999 * len(tokenized_dataset))
    val_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
    train_dataset = tokenized_dataset.select(range(train_size))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
    )

    # LR scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps,
    )

    # ---- SwanLab init ----
    method_output = os.path.join(args.output_dir, args.method)
    os.makedirs(method_output, exist_ok=True)

    config_dict = {
        "method": args.method,
        "model": args.model,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_tokens": effective_batch_tokens,
        "total_tokens": total_steps * effective_batch_tokens,
        "seq_length": args.seq_length,
        "block_size": args.block_size,
        "sr_bias_scale": args.sr_bias_scale if args.method == "mb_sr" else 0.0,
        "params_total": sum(p.numel() for p in model.parameters()),
    }

    swanlab.init(
        project=args.project_name,
        experiment_name=f"{args.method}",
        config=config_dict,
    )

    # Backup plain-text log
    log_path = os.path.join(method_output, "train_log.jsonl")
    log_file = open(log_path, "w")

    # Resume
    start_step = 0
    if args.from_checkpoint:
        start_step = load_checkpoint(model, optimizer, scheduler, args.from_checkpoint)

    torch.cuda.reset_peak_memory_stats()

    # ---- Print config ----
    print(f"\n{'='*60}")
    print(f"Method:        {args.method}")
    print(f"Micro batch:   {args.batch_size} x {args.seq_length} = {tokens_per_micro_batch:,} tokens")
    print(f"Grad accum:    {args.gradient_accumulation_steps} steps")
    print(f"Effective:     {effective_batch_tokens:,} tokens/step")
    print(f"Total steps:   {total_steps:,}")
    print(f"Total tokens:  {total_steps * effective_batch_tokens:,}")
    print(f"Params:        {config_dict['params_total']:,}")
    print(f"{'='*60}\n")

    model.train()
    global_step = start_step
    accumulation_loss = 0.0

    pbar = tqdm(total=total_steps - start_step, desc=f"[{args.method}]")
    data_iter = iter(train_loader)
    last_log_time = time.time()

    while global_step < total_steps:
        optimizer.zero_grad()

        for micro_step in range(args.gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / args.gradient_accumulation_steps

            loss.backward()
            accumulation_loss += loss.item()

        # Gradient clipping
        grad_norm = 0.0
        if args.method == "dense":
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()

        optimizer.step()
        scheduler.step()
        global_step += 1

        # ---- Logging ----
        if global_step % args.log_interval == 0:
            now = time.time()
            elapsed_since_log = now - last_log_time
            tokens_since_log = effective_batch_tokens * args.log_interval
            tokens_per_sec = tokens_since_log / elapsed_since_log if elapsed_since_log > 0 else 0
            peak_mem_bytes = torch.cuda.max_memory_allocated()
            peak_mem_mb = peak_mem_bytes / (1024 ** 2)
            lr_now = scheduler.get_last_lr()[0]

            # SwanLab logging — each metric in its own group
            log_data = {
                "train/loss": accumulation_loss,
                "train/perplexity": math.exp(accumulation_loss),
                "train/learning_rate": lr_now,
                "train/tokens_per_sec": tokens_per_sec,
                "system/peak_gpu_memory_mb": peak_mem_mb,
            }

            if args.method == "dense":
                log_data["train/grad_norm"] = grad_norm

            swanlab.log(log_data, step=global_step)

            # Backup JSONL log
            json.dump({"step": global_step, **log_data}, log_file)
            log_file.write("\n")
            log_file.flush()

            pbar.set_postfix({
                "loss": f"{accumulation_loss:.4f}",
                "lr": f"{lr_now:.2e}",
                "tok/s": f"{tokens_per_sec:.0f}",
                "mem": f"{peak_mem_mb:.0f}MB",
            })

            last_log_time = time.time()

        accumulation_loss = 0.0
        pbar.update(1)

        # ---- Validation ----
        if global_step % args.eval_interval == 0 and len(val_dataset) > 0:
            val_loss, val_ppl = validate(model, val_loader, device, args.val_samples)
            swanlab.log({"eval/val_loss": val_loss, "eval/val_perplexity": val_ppl}, step=global_step)
            json.dump({"step": global_step, "eval/val_loss": val_loss, "eval/val_perplexity": val_ppl}, log_file)
            log_file.write("\n")
            log_file.flush()
            print(f"\n  [Step {global_step}] val_loss = {val_loss:.4f}, val_ppl = {val_ppl:.2f}")

        # ---- Checkpoint ----
        if global_step % args.save_interval == 0:
            ckpt_dir = os.path.join(method_output, f"step_{global_step}")
            save_checkpoint(model, optimizer, scheduler, global_step, ckpt_dir,
                           save_hf=(args.method == "dense"))

    pbar.close()

    # Final save
    final_path = os.path.join(method_output, "final")
    save_checkpoint(model, optimizer, scheduler, global_step, final_path,
                   save_hf=(args.method == "dense"))

    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    swanlab.log({"system/final_peak_gpu_memory_mb": peak_mem_mb}, step=global_step)

    log_file.close()
    swanlab.finish()

    print(f"\nTraining complete. Final checkpoint: {final_path}")
    print(f"Logs: {log_path}")
    print(f"Peak GPU memory: {peak_mem_mb:.0f} MB")


if __name__ == "__main__":
    args = get_args()
    train(args)
