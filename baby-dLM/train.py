import os
# Reduce CUDA memory fragmentation (PyTorch 2.1+).  Must be set before
# any torch.cuda call.  See: https://pytorch.org/docs/stable/notes/cuda.html
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import sys
import time
import math
import pickle
import argparse
import importlib
import inspect
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("true", "1", "yes", "y", "t"):
        return True
    if v in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v}")


parser = argparse.ArgumentParser()

# Model selection
parser.add_argument(
    "--model", type=str, default="mdlm",
    choices=["mdlm", "ar", "bd3lm"],
    help="Which model variant to train",
)
parser.add_argument(
    "--data", type=str, default="climbmix",
    choices=["climbmix"],
    help="Data source: 'climbmix' (BPE, HuggingFace parquet shards)",
)

# Training
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--block_size", type=int, default=256)
parser.add_argument("--block_len", type=int, default=None,
                    help="Diffusion block length for block models (default: block_size)")
parser.add_argument("--grad_accum_steps", type=int, default=1,
                    help="Gradient accumulation steps.  Effective batch = "
                         "batch_size * grad_accum_steps.  Use 2 for effective "
                         "bs=256 at micro-batch=128.")
parser.add_argument("--max_iters", type=int, default=4000)
parser.add_argument("--eval_interval", type=int, default=300)
parser.add_argument("--train_log_interval", type=int, default=100,
                    help="How often to print train loss/grad norm progress lines")
parser.add_argument("--warmup_iters", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=3e-3)
parser.add_argument("--min_lr", type=float, default=3e-4)
parser.add_argument("--eval_iters", type=int, default=50)
parser.add_argument("--save_interval", type=int, default=200)
parser.add_argument(
    "--save_steps", type=str, default="",
    help="Comma-separated list of exact step numbers to save checkpoints at "
         "(e.g. '200,300,500,800'). Overrides --save_interval when non-empty.",
)
parser.add_argument(
    "--save_weights_only", type=str2bool, default=False,
    help="If true, ALL checkpoints (step-numbered and final) omit optimizer state to save disk.",
)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument(
    "--optimizer", type=str, default="adamw",
    choices=["adamw", "normuon"],
    help="Optimizer family: 'adamw' (single global LR) or 'normuon' (grouped MuonAdamW)",
)
parser.add_argument("--adam_mult", type=float, default=1.0,
                    help="NorMuon: multiplier for AdamW group LRs (embedding, unembedding, scalar)")
parser.add_argument("--matrix_mult", type=float, default=1.0,
                    help="NorMuon: multiplier for Muon matrix LR")
parser.add_argument("--normuon_weight_decay", type=float, default=0.2,
                    help="NorMuon: cautious weight decay for Muon groups")

# Model
parser.add_argument("--n_embd", type=int, default=256)
parser.add_argument("--n_head", type=int, default=4)
parser.add_argument("--n_layer", type=int, default=6)
parser.add_argument("--dropout", type=float, default=0.1)

# Diffusion / corruption
parser.add_argument("--T", type=int, default=100)
parser.add_argument("--t_min", type=float, default=0.45,
                    help="Lower clipping bound for t/T (0.0 = no lower clip)")
parser.add_argument("--t_max", type=float, default=0.95,
                    help="Upper clipping bound for t/T (1.0 = no upper clip)")
parser.add_argument(
    "--noise_schedule",
    type=str,
    default="linear",
    choices=["linear", "cosine"],
)

# Eval
parser.add_argument("--eval_t_frac", type=float, default=0.6,
                    help="Fixed time fraction for validation denoising CE")
parser.add_argument("--gpt2_eval_samples", type=int, default=50,
                    help="Number of samples for GPT2-large CE metric")
parser.add_argument(
    "--gpt2_eval_interval",
    type=int,
    default=600,
    help="How often to run GPT2-large evaluation (0 disables)",
)
parser.add_argument(
    "--sample_interval",
    type=int,
    default=None,
    help="How often to print a generated sample during training (defaults to eval_interval, 0 disables)",
)

# Data / misc
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--checkpoint_path", type=str, default=None,
                    help="Defaults to ckpt_{model}.pt")
parser.add_argument(
    "--resume_from",
    type=str,
    default=None,
    help="Warm-start from a checkpoint by loading model weights only; optimizer and LR schedule restart from scratch",
)
parser.add_argument("--loss_log_path", type=str, default=None,
                    help="Defaults to loss_{model}.pkl")
parser.add_argument("--prompt_len", type=int, default=16)
parser.add_argument("--num_final_samples", type=int, default=10)
parser.add_argument("--use_compile", type=str2bool, default=True)
parser.add_argument("--warmup_stable", type=str2bool, default=False,
                    help="When True, hold LR constant after warmup (no cosine decay)")
parser.add_argument("--skip_final_eval", type=str2bool, default=False,
                    help="Skip the forced final eval at the end of training")
parser.add_argument("--skip_final_checkpoint", type=str2bool, default=False,
                    help="Skip writing the final checkpoint at the end of training")

args = parser.parse_args()

# NOTE: block_len validation is deferred until after the data source resolves
# block_size (ClimbMix overrides it to 2048).  See "Resolve block_len" below.

if args.checkpoint_path is None:
    args.checkpoint_path = f"ckpt_{args.model}.pt"
if args.loss_log_path is None:
    args.loss_log_path = f"loss_{args.model}.pkl"

batch_size = args.batch_size
block_size = args.block_size  # may be overridden by climbmix below
# block_len is resolved after data source sets final block_size
max_iters = args.max_iters
eval_interval = args.eval_interval
train_log_interval = args.train_log_interval
warmup_iters = args.warmup_iters
learning_rate = args.learning_rate
min_lr = args.min_lr
eval_iters = args.eval_iters
save_interval = args.save_interval
save_steps_set = set()
if args.save_steps:
    save_steps_set = {int(s.strip()) for s in args.save_steps.split(",") if s.strip()}
grad_clip = args.grad_clip
grad_accum_steps = args.grad_accum_steps
gpt2_eval_interval = (
    eval_interval if args.gpt2_eval_interval is None else args.gpt2_eval_interval
)
sample_interval = (
    eval_interval if args.sample_interval is None else args.sample_interval
)
progress_log_interval = max(1, train_log_interval)
if eval_interval > 0:
    progress_log_interval = min(progress_log_interval, eval_interval)

n_embd = args.n_embd
n_head = args.n_head
n_layer = args.n_layer
dropout = args.dropout
assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
head_dim = n_embd // n_head

T = args.T
t_min = args.t_min
t_max = args.t_max
noise_schedule = args.noise_schedule
checkpoint_path = args.checkpoint_path
loss_log_path = args.loss_log_path
eval_t_frac = args.eval_t_frac
gpt2_eval_samples = args.gpt2_eval_samples
prompt_len = args.prompt_len
num_final_samples = args.num_final_samples
warmup_stable = args.warmup_stable
data_source = args.data

# --- Dynamic model import ---
MODEL_MODULE_MAP = {
    "mdlm":  "model_MDLM",
    "ar":    "model_AR",
    "bd3lm": "model_bd3lm",
}
model_module = importlib.import_module(MODEL_MODULE_MAP[args.model])
Model = model_module.Model
generate_from = model_module.generate_from
model_make_batch = model_module.make_batch
model_make_eval_batch = model_module.make_eval_batch
model_compute_loss = model_module.compute_loss
model_compute_eval_loss = model_module.compute_eval_loss

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# Ensure Flash Attention is preferred by SDPA when available.
if device == "cuda":
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    # Log which SDPA backends are available for debugging VRAM issues.
    print(f"SDPA backends — flash: {torch.backends.cuda.flash_sdp_enabled()}, "
          f"mem_efficient: {torch.backends.cuda.mem_efficient_sdp_enabled()}, "
          f"math: {torch.backends.cuda.math_sdp_enabled()}")
    print(f"CUDA alloc conf: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '(not set)')}")

torch.manual_seed(args.seed)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def autocast_ctx():
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def token_epochs_from_steps(num_steps, ntok):
    if ntok is None or ntok == 0:
        return float("nan")
    return (num_steps * batch_size * block_size * grad_accum_steps) / ntok


def print_run_info(model):
    n_params = count_parameters(model)
    micro_tokens = batch_size * block_size
    effective_tokens_per_step = micro_tokens * grad_accum_steps

    print("=" * 80)
    print("Training config")
    print("=" * 80)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print(f"device: {device}")
    print(f"vocab_size: {vocab_size}")
    if num_train_tokens is not None:
        print(f"train_tokens: {num_train_tokens:,}")
    if grad_accum_steps > 1:
        print(f"micro_batch_size: {batch_size}")
        print(f"effective_batch_size: {batch_size * grad_accum_steps}")
        print(f"grad_accum_steps: {grad_accum_steps}")
    print(f"tokens_per_step: {effective_tokens_per_step:,}")
    print(f"model_parameters: {n_params:,} ({n_params / 1e6:.3f}M)")
    if num_train_tokens is not None:
        te_step = token_epochs_from_steps(1, num_train_tokens)
        te_total = token_epochs_from_steps(max_iters, num_train_tokens)
        print("-" * 80)
        print(f"token_epochs_per_step: {te_step:.4f}")
        print(f"expected_total_token_epochs: {te_total:.2f}")
    print("=" * 80)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

# Data source interface:
#   vocab_size, mask_token_id, block_size (possibly overridden)
#   get_data_batch(split) -> (B, block_size) tensor of clean tokens on device
#   decode(list_of_ids) -> str
#   num_train_tokens -> int or None

if device != "cuda":
    raise RuntimeError(
        "ClimbMix training requires CUDA (the dataloader uses pinned memory "
        "and GPU buffers)."
    )

from prepare import Tokenizer, make_dataloader, MAX_SEQ_LEN

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
mask_token_id = tokenizer.mask_token_id
block_size = MAX_SEQ_LEN  # 2048 — fixed ClimbMix context length
args.block_size = block_size

_train_loader = make_dataloader(tokenizer, batch_size, block_size, "train")
_val_loader = make_dataloader(tokenizer, batch_size, block_size, "val")

def get_data_batch(split):
    loader = _train_loader if split == "train" else _val_loader
    inputs, _targets, _epoch = next(loader)
    return inputs  # (B, block_size) on CUDA

def decode(ids):
    return tokenizer.decode(ids)

num_train_tokens = None  # unknown for streaming shards

print(f"Data: ClimbMix (vocab_size={vocab_size}, block_size={block_size}, "
      f"mask_token_id={mask_token_id})")


# ---------------------------------------------------------------------
# Resolve block_len (deferred until data source has set block_size)
# ---------------------------------------------------------------------

if args.block_len is None:
    args.block_len = block_size
block_len = args.block_len
if block_size % block_len != 0:
    raise ValueError(
        f"block_len={block_len} must divide block_size={block_size}"
    )

print(f"Vocab size: {vocab_size}")


# ---------------------------------------------------------------------
# Diffusion / corruption schedule
# ---------------------------------------------------------------------

def time_fraction_tensor(t_steps: torch.Tensor) -> torch.Tensor:
    """
    Map integer timesteps t in {1, ..., T} to a float in [t_min, t_max].
    """
    t_frac = t_steps.float() / T
    t_frac = t_frac.clamp(min=t_min, max=t_max)
    return t_frac


def time_fraction_scalar(t_step: int) -> float:
    t_frac = float(t_step) / T
    t_frac = min(max(t_frac, t_min), t_max)
    return t_frac


def survival_prob_tensor(t_steps: torch.Tensor) -> torch.Tensor:
    """
    Probability a token stays unmasked at timestep t.
    """
    t_frac = time_fraction_tensor(t_steps)
    if noise_schedule == "linear":
        a_t = 1.0 - t_frac
    elif noise_schedule == "cosine":
        a_t = torch.cos(0.5 * math.pi * t_frac)
    else:
        raise ValueError(f"unknown noise_schedule: {noise_schedule}")
    return a_t.clamp(0.0, 1.0)


def survival_prob_scalar(t_step: int) -> float:
    t_frac = time_fraction_scalar(t_step)
    if noise_schedule == "linear":
        a_t = 1.0 - t_frac
    elif noise_schedule == "cosine":
        a_t = math.cos(0.5 * math.pi * t_frac)
    else:
        raise ValueError(f"unknown noise_schedule: {noise_schedule}")
    return max(0.0, min(1.0, a_t))


# ---------------------------------------------------------------------
# Eval: fixed-t batch for validation
# ---------------------------------------------------------------------

def eval_t_step_from_frac():
    return int(max(1, min(T, round(eval_t_frac * T))))


def get_model_eval_batch(split):
    """Build a model-specific eval batch from clean tokens."""
    x0 = get_data_batch(split)
    return model_make_eval_batch(
        x0, cfg, fixed_t_step=eval_t_step_from_frac(),
    )


@torch.no_grad()
def estimate_loss(run_model):
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb, mb = get_model_eval_batch(split)
            with autocast_ctx():
                loss = model_compute_eval_loss(run_model, xb, yb, mb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    return out


# ---------------------------------------------------------------------
# Eval: GPT2-large CE on generated samples
# ---------------------------------------------------------------------

@torch.no_grad()
def estimate_gpt2_ce(diffusion_model, gpt2_model, gpt2_tokenizer,
                     num_samples=100):
    """
    Generate samples from the diffusion model, then score the generated
    (non-prefix) portion under GPT2-large.  Returns average CE in
    GPT2's BPE token space.
    """
    diffusion_model.eval()
    gpt2_model.eval()

    total_ce = 0.0
    total_tokens = 0

    for i in range(num_samples):
        sample_text = generate(diffusion_model, prompt_len=prompt_len)

        # Tokenize full text with offset mappings so we can find the
        # exact BPE-token boundary between prefix and continuation.
        # BPE is NOT additive: BPE(a+b) != BPE(a) || BPE(b), so we
        # cannot just tokenize the prefix separately to find the split.
        enc = gpt2_tokenizer(
            sample_text, return_offsets_mapping=True, return_tensors="pt",
        )
        full_ids = enc["input_ids"][0]           # (L,)
        offsets = enc["offset_mapping"][0]        # (L, 2) — char spans

        # First token whose char span starts strictly at or after the
        # prompt boundary.  Tokens straddling the boundary are excluded.
        cont_start = None
        for idx_tok, (char_start, char_end) in enumerate(offsets.tolist()):
            if char_start >= prompt_len:
                cont_start = idx_tok
                break
        if cont_start is None or cont_start >= len(full_ids):
            continue  # entire text falls within / straddles the prefix

        if len(full_ids) <= cont_start + 1:
            continue  # too few continuation tokens to score

        input_ids = full_ids.unsqueeze(0).to(device)
        logits = gpt2_model(input_ids).logits  # (1, L, V_gpt2)

        # CE for next-token prediction, scored only on continuation.
        # logits[0, i] predicts token i+1, so to score tokens from
        # cont_start onward we use logits[cont_start-1 : -1].
        start = max(cont_start - 1, 0)
        target_ids = full_ids[start + 1:].to(device)
        pred_logits = logits[0, start : start + len(target_ids)]

        if len(target_ids) == 0:
            continue

        ce = F.cross_entropy(pred_logits, target_ids, reduction="sum").item()
        total_ce += ce
        total_tokens += len(target_ids)

    if total_tokens == 0:
        return float("inf")
    return total_ce / total_tokens


# ---------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------

def generate(model, prompt_len=16, prompt_tokens=None):
    """Generate a sample.  Uses prompt_tokens if provided, else draws from val."""
    model.eval()

    x = torch.full((1, block_size), mask_token_id, device=device)
    if prompt_tokens is None:
        x0 = get_data_batch("val")  # (B, block_size) on device
        x[0, :prompt_len] = x0[0, :prompt_len]
    else:
        x[0, :prompt_len] = prompt_tokens[:prompt_len].to(device)

    prompt_mask = torch.zeros((1, block_size), dtype=torch.bool, device=device)
    prompt_mask[:, :prompt_len] = True

    return generate_from(
        model, x, prompt_mask,
        T=T, block_size=block_size, block_len=block_len,
        vocab_size=vocab_size,
        mask_token_id=mask_token_id,
        survival_prob_scalar=survival_prob_scalar,
        decode=decode,
    )


# ---------------------------------------------------------------------
# Model config dict (passed to model's get_batch / compute_loss)
# ---------------------------------------------------------------------

cfg = {
    "batch_size": batch_size,
    "block_size": block_size,
    "block_len": block_len,
    "T": T,
    "mask_token_id": mask_token_id,
    "vocab_size": vocab_size,
    "device": device,
    "survival_prob_tensor": survival_prob_tensor,
}


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters

    if warmup_stable:
        return learning_rate  # constant after warmup

    decay_ratio = (it - warmup_iters) / max(1, (max_iters - warmup_iters))
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def load_model_weights(model, init_path):
    checkpoint = torch.load(init_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)

    source_model = None
    source_iter = None
    if isinstance(checkpoint, dict):
        source_iter = checkpoint.get("iter")
        source_args = checkpoint.get("args")
        if isinstance(source_args, dict):
            source_model = source_args.get("model")

    details = [f"Loaded model weights from {init_path}"]
    if source_model is not None:
        details.append(f"source_model={source_model}")
    if source_iter is not None:
        details.append(f"source_iter={source_iter}")
    details.append("optimizer_reset=true")
    details.append("schedule_reset=true")
    print(" | ".join(details))


if __name__ == "__main__":
    model = Model(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        head_dim=head_dim,
        block_size=block_size,
        dropout=dropout,
        block_len=block_len,
    ).to(device)
    if args.resume_from is not None:
        load_model_weights(model, args.resume_from)

    optimizer_family = args.optimizer
    normuon_lrs = None  # populated only for normuon

    if optimizer_family == "normuon":
        from normuon import setup_normuon_optimizer
        optimizer, normuon_lrs = setup_normuon_optimizer(
            model, n_embd,
            adam_mult=args.adam_mult,
            matrix_mult=args.matrix_mult,
            weight_decay=args.normuon_weight_decay,
        )
    else:
        adamw_kwargs = {"lr": min_lr}
        if "fused" in inspect.signature(torch.optim.AdamW).parameters:
            adamw_kwargs["fused"] = (device == "cuda")
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)

    use_compile = (device == "cuda") and hasattr(torch, "compile") and args.use_compile
    compiled_model = torch.compile(model) if use_compile else model

    gpt2_model = None
    gpt2_tokenizer = None
    gpt2_enabled = gpt2_eval_samples > 0 and gpt2_eval_interval > 0
    if gpt2_enabled:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        # Load GPT2-large for evaluation only when that metric is enabled.
        print("Loading GPT2-large for evaluation...")
        gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large")
        gpt2_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        gpt2_model = GPT2LMHeadModel.from_pretrained(
            "gpt2-large", torch_dtype=gpt2_dtype
        ).to(device)
        gpt2_model.eval()
        print(f"GPT2-large loaded ({gpt2_dtype}).")

    print_run_info(model)

    train_losses = []
    val_losses = []
    gpt2_ces = []

    model.train()

    for iter in range(max_iters):
        lr = get_lr(iter)
        if optimizer_family == "normuon":
            from normuon import update_normuon_schedule
            # For normuon, lr acts as a multiplier relative to initial_lr
            lr_mult = lr / learning_rate if learning_rate > 0 else 1.0
            progress = iter / max(1, max_iters - 1)
            update_normuon_schedule(
                optimizer, lr_mult, iter,
                weight_decay_base=args.normuon_weight_decay,
                progress=progress,
            )
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        # Skip step-0 loss eval: for short runs and LR sweeps this adds
        # substantial overhead before any training has happened.
        should_run_loss_eval = (
            eval_interval > 0 and iter > 0 and (iter % eval_interval == 0)
        )
        should_run_gpt2_eval = gpt2_enabled and (iter % gpt2_eval_interval == 0)
        should_print_sample = sample_interval > 0 and (iter % sample_interval == 0)

        if should_run_loss_eval or should_run_gpt2_eval or should_print_sample:
            model.eval()
            metrics = None
            gpt2_ce = None

            if should_run_loss_eval:
                metrics = estimate_loss(compiled_model)
                train_losses.append((iter, metrics["train"]))
                val_losses.append((iter, metrics["val"]))

            if should_run_gpt2_eval:
                gpt2_ce = estimate_gpt2_ce(
                    model, gpt2_model, gpt2_tokenizer,
                    num_samples=gpt2_eval_samples,
                )
                gpt2_ces.append((iter, gpt2_ce))

            current_token_epoch = token_epochs_from_steps(iter, num_train_tokens)
            log_parts = [
                f"step {iter}",
                f"tok_epoch {current_token_epoch:.2f}",
            ]
            if metrics is not None:
                log_parts.append(f"train {metrics['train']:.4f}")
                log_parts.append(f"val {metrics['val']:.4f}")
            if gpt2_ce is not None:
                log_parts.append(f"gpt2_ce {gpt2_ce:.4f}")
            log_parts.append(f"lr {lr:.6f}")
            print(" | ".join(log_parts))

            if should_print_sample:
                print("Generating sample...")
                print(generate(model, prompt_len=prompt_len))
            model.train()

        # --- Gradient accumulation loop ---
        # Each optimizer step accumulates grad_accum_steps micro-batches.
        # Loss is scaled by 1/grad_accum_steps so that the averaged gradient
        # is independent of the accumulation count.
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _micro in range(grad_accum_steps):
            x0 = get_data_batch("train")
            batch = model_make_batch(x0, cfg)
            with autocast_ctx():
                micro_loss = model_compute_loss(compiled_model, batch, cfg)
                scaled_loss = micro_loss / grad_accum_steps
            scaled_loss.backward()
            accum_loss += micro_loss.item()
        loss_val = accum_loss / grad_accum_steps  # average for logging

        if optimizer_family == "normuon":
            # Muon's orthogonalization handles gradient scaling;
            # compute grad norm for logging but don't clip.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float('inf')
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            )
        optimizer.step()

        # Checkpoint saving.  Use (iter + 1) so the step number in the
        # filename equals the number of completed optimizer updates:
        #   iter=199  → 200 updates done → ckpt_step200.pt
        #   iter=799  → 800 updates done → ckpt_step800.pt
        # This ensures max_iters=800 actually produces ckpt_step800.pt
        # (the loop runs iter 0..799, so iter=800 never occurs).
        completed_steps = iter + 1
        should_save = (
            (completed_steps in save_steps_set) if save_steps_set
            else (save_interval > 0 and completed_steps % save_interval == 0)
        )
        if should_save:
            ckpt = {
                "iter": completed_steps,
                "model_state_dict": model.state_dict(),
                "loss": loss_val,
                "args": vars(args),
                "vocab_size": vocab_size,
                "mask_token_id": mask_token_id,
                "optimizer_family": optimizer_family,
            }
            if not args.save_weights_only:
                ckpt["optimizer_state_dict"] = optimizer.state_dict()
            if normuon_lrs is not None:
                ckpt["normuon_lrs"] = normuon_lrs
            # Save step-numbered checkpoint (for checkpoint sharing across
            # curriculum variants) AND overwrite the latest checkpoint.
            base, ext = os.path.splitext(checkpoint_path)
            step_path = f"{base}_step{completed_steps}{ext}"
            torch.save(ckpt, step_path)
            torch.save(ckpt, checkpoint_path)
            print(f"saved checkpoint to {step_path} (and {checkpoint_path}) at step {completed_steps}")

        # GC management: Python's cyclic GC causes ~500ms stalls when it
        # scans the autograd graph.  Run a full collection on step 0 (to
        # clean up model-init garbage), then freeze and disable.  Re-collect
        # every 5000 steps to prevent slow leaks.
        if iter == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (iter + 1) % 5000 == 0:
            gc.collect()

        if iter % progress_log_interval == 0 or iter == max_iters - 1:
            current_token_epoch = token_epochs_from_steps(iter, num_train_tokens)
            print(
                f"step {iter} | tok_epoch {current_token_epoch:.2f} | "
                f"loss {loss_val:.4f} | grad_norm {grad_norm:.4f} | lr {lr:.6f}"
            )

    # --- Forced final eval (avoids stale metrics when max_iters % eval_interval != 0) ---
    if not args.skip_final_eval:
        last_eval_iter = max(s for s, _ in val_losses) if val_losses else -1
        if last_eval_iter < max_iters - 1:
            model.eval()
            final_metrics = estimate_loss(compiled_model)
            train_losses.append((max_iters, final_metrics["train"]))
            val_losses.append((max_iters, final_metrics["val"]))
            log_parts = [f"step {max_iters} (final)",
                         f"train {final_metrics['train']:.4f}",
                         f"val {final_metrics['val']:.4f}"]

            if gpt2_enabled:
                last_gpt2_iter = max(s for s, _ in gpt2_ces) if gpt2_ces else -1
                if last_gpt2_iter < max_iters - 1:
                    gpt2_ce = estimate_gpt2_ce(
                        model, gpt2_model, gpt2_tokenizer,
                        num_samples=gpt2_eval_samples,
                    )
                    gpt2_ces.append((max_iters, gpt2_ce))
                    log_parts.append(f"gpt2_ce {gpt2_ce:.4f}")

            print(" | ".join(log_parts))
            model.train()

    if not args.skip_final_checkpoint:
        final_ckpt = {
            "iter": max_iters,
            "model_state_dict": model.state_dict(),
            "loss": loss_val,
            "args": vars(args),
            "vocab_size": vocab_size,
            "mask_token_id": mask_token_id,
            "optimizer_family": optimizer_family,
        }
        if not args.save_weights_only:
            final_ckpt["optimizer_state_dict"] = optimizer.state_dict()
        if normuon_lrs is not None:
            final_ckpt["normuon_lrs"] = normuon_lrs
        torch.save(final_ckpt, checkpoint_path)
        print(f"saved final checkpoint to {checkpoint_path}")

    samples = []
    if num_final_samples > 0:
        print("\n" + "=" * 60)
        print(f"Generating {num_final_samples} samples")
        print("=" * 60)
        model.eval()

        for i in range(num_final_samples):
            sample_text = generate(model, prompt_len=prompt_len)
            samples.append(sample_text)
            print(f"\n--- Sample {i + 1} ---")
            print(sample_text)

    with open(loss_log_path, "wb") as f:
        pickle.dump({
            "train": train_losses,
            "val": val_losses,
            "gpt2_ce": gpt2_ces,
        }, f)
