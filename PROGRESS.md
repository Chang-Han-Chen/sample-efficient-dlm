# Progress

This file tracks concrete work completed in this repository so future runs do
not depend on conversational memory. It should be updated whenever code,
profiling, or conclusions change materially.

## Current Objective

Build a JAX AR training stack and transformer backbone that is faithful to the
PyTorch reference, trains on ClimbMix, and is faster/more memory efficient on a
single A100. The immediate optimization focus is native JAX/XLA/cuDNN first:
larger fused matmuls, memory-efficient attention, and then chunked/fused final
linear cross entropy if logits memory remains the limiter.

## User Decisions

- Ignore git for now; the user will handle it later.
- Tie token embeddings and LM head by default.
- Use the fixed small GPT shape for sanity/profiling:
  `d_model=768`, `d_ff=2048`, `n_layers=8`, `n_heads=12`,
  `n_kv_heads=None`, `vocab_size=32768`, bf16.
- Treat baseline architecture as old intervention flags off.
- Treat the "old architecture" profile as QK-norm + value residual +
  layernorm/depth scaling + per-head attention gating, with weight tying still
  enabled. If comparing to the strict `gpt_small_faster.py` old bundle, label
  that separately because that PyTorch small config clearly includes QK-norm,
  value residual, and layernorm/depth scaling, while gating appears in larger
  final configs.

## Implemented Backbone Features

- JAX transformer supports RMSNorm, RoPE, SwiGLU, QK-norm, value residual,
  per-head/elementwise attention gating, optional weight tying, depth/layernorm
  scaling, selectable attention implementation, value embeddings, and hidden
  state return before logits.
- Weight tying is real in JAX: `lm_head.weight` is the exact same `nnx.Param`
  object as `embedding.weight`, and NNX state exposes only `embedding.weight`.
- Value embeddings follow PLAN.md Option D:
  first cache raw first-layer value stream, apply value-residual normalization,
  then add token value embeddings after the residual mixture.
- Attention uses `jax.nn.dot_product_attention`; cuDNN implementation is used
  for A100 profiling when requested.
- Added diffusion mask helpers for future MDLM/BD3LM work.

## Implemented Training/Data Stack

- Added AR loss with z-loss reporting.
- Added jitted train/eval steps and true gradient accumulation over a leading
  microbatch axis.
- Added first-pass NorMuonCWD+AdamW optimizer:
  matrix parameters use Muon; embeddings, lm head, scalar/vector and other
  non-matrix params use AdamW.
- Added ClimbMix preparation and inspection utilities.
- Smoke ClimbMix data exists at `data/climbmix_smoke/` and is ignored by git.
- Tokenizer sanity checks passed:
  vocab size `32768`, specials IDs `0..3`, diffusion mask ID `32768`,
  sample roundtrips matched, token byte stats looked normal.
- Loader sanity checks passed across multiple batch sizes:
  shapes correct, IDs in range, shift relation correct.

## Correctness Checks Passed

- `python jax/tests/test_parity.py`
  passed after native fusion changes. Max full-transformer logits diff was
  about `2.74e-06`.
- `python jax/tests/test_parity_extras.py`
  passed after native fusion changes for tying, layernorm scaling, QK-norm,
  gating, remat, value embeddings, masks, and hidden/logit API.
- `python jax/tests/test_training_stack.py`
  passed after native fusion changes.
- `python -m py_compile` passed for changed Python files.
- Small fixed-batch overfit on ClimbMix succeeded:
  loss dropped from about `10.87` to about `0.005`.

## A100 Environment

- GPU: NVIDIA A100-SXM4-80GB.
- Escalated Python sees `cuda:0`; sandboxed Python sees CPU only because CUDA
  init is blocked in the sandbox.
- `train_ar.py` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` before importing
  JAX so `nvidia-smi` memory readings are meaningful.
- MFU estimates use A100 SXM dense BF16 peak `312e12` FLOP/s.

## Initial Pre-Fusion Profiles

All runs used tied embeddings, bf16, NorMuonCWD+AdamW, cuDNN attention unless
noted.

Fixed batch size 16:

| seq_len | peak HBM by nvidia-smi | step time | tok/s | est TFLOP/s | MFU |
|---:|---:|---:|---:|---:|---:|
| 512 | ~6.6 GB | 0.0747s | 109.6k | 74.5 | 23.9% |
| 1024 | ~11.7 GB | 0.1050s | 156.0k | 111.9 | 35.9% |
| 2048 | ~23.0 GB | 0.1771s | 185.0k | 146.7 | 47.0% |

Interpretation: memory scales roughly linearly with sequence length, not
quadratically, consistent with cuDNN memory-efficient/flash-style attention.
MFU increases with sequence length because larger `T` gives larger matmuls and
higher arithmetic intensity; with memory-efficient attention, the extra
attention compute is useful work rather than quadratic activation storage.

Batch scaling at `seq_len=512`:

- batch 64: ~41.4 GB peak HBM, 0.2117s/step, 154.8k tok/s, 33.7% MFU.
- batch 96: ~61.3 GB peak HBM, 0.2900s/step, 169.5k tok/s, 36.9% MFU.

## Native Fusion Optimization

Implemented native JAX fusion changes:

- Q/K/V are separate trainable parameters (`W_q`, `W_k`, `W_v`) so Muon sees
  three independent matrices. When `fuse_qkv=True`, the forward pass
  concatenates the three weights and performs one QKV matmul; gradients still
  flow back to the separate parameters. This intentionally differs from the
  PyTorch reference, whose single trainable `qkv` parameter is considered wrong
  for the intended Muon semantics.
- Replaced split SwiGLU `w_up` and `w_gate` projections with one fused
  `w_up_gate` projection and static split.
- Updated parity and training tests to copy/check PyTorch fused QKV slices into
  the separate JAX Q/K/V parameters, and to verify the optimizer groups
  `W_q`, `W_k`, and `W_v` as separate Muon parameters.

Post-fusion baseline profile at `seq_len=1024`, batch 16:

- Architecture flags: QK-norm off, value residual off, gating off,
  layernorm/depth scaling off, value embeddings off, tied embeddings on.
- JAX command log: `/tmp/jax_ar_fused_baseline_seq1024_b16.jsonl`.
- `nvidia-smi` monitor log: `/tmp/smi_fused_baseline_seq1024.csv`.
- `avg_measured_step=0.0933s`, measured from step 3 over 9 steps.
- `tokens_per_sec=175618`.
- `est_tflops=126.0`.
- `mfu=40.4%`.
- `jax_peak_hbm_gb=8.54`; `nvidia-smi` observed peak was about `11739 MiB`.
- Compile plus first step: `36.102s`.

Compared to the earlier non-fused `seq_len=1024` baseline:

- Step time improved from `0.1050s` to `0.0933s`.
- Tokens/sec improved from `156.0k` to `175.6k`.
- MFU improved from `35.9%` to `40.4%`.
- `nvidia-smi` peak HBM stayed about `11.7 GB`.

## Old-Architecture Profile

The first old-architecture bf16 profile exposed and fixed a dtype bug:

- Config: QK-norm on, value residual on, per-head gating on,
  layernorm/depth scaling on, tied embeddings on.
- Failure: `jax.nn.dot_product_attention` rejected mixed Q/K/V dtypes.
- Cause: value residual scalar parameters are fp32; combining them with bf16
  `v_raw` promoted the attention value tensor to fp32 while Q/K remained bf16.
- Fix applied: cast the value-residual result back to `v_raw.dtype` before
  calling attention.
- Added a bf16 old-architecture MHSA smoke test to catch this dtype path.
- After the fix, `python jax/tests/test_parity.py`,
  `python jax/tests/test_parity_extras.py`, and
  `python jax/tests/test_training_stack.py` passed.

Longer matched profiles at `seq_len=1024`, batch 16, 30 total steps,
measured from step 5:

| config | flags | step time | tok/s | est TFLOP/s | MFU | nvidia-smi peak |
|---|---|---:|---:|---:|---:|---:|
| fused baseline | old flags off | 0.0926s | 176.9k | 126.9 | 40.7% | 11739 MiB |
| fused old-all | QK-norm + value residual + layernorm scaling + per-head gating | 0.1128s | 145.2k | 104.2 | 33.4% | 11745 MiB |

Interpretation: native QKV/SwiGLU fusion improved the baseline materially, but
the full old-all stack is still about `22%` slower at this shape. Peak HBM is
nearly identical by `nvidia-smi`, so the old-all cost is mostly extra compute
and/or fusion boundaries, not extra activation memory. The likely specific
costs are QK-norm materialization before cuDNN attention and the per-head
attention gate projection/multiply.

## Seq-512 Large-Batch Profiles

These runs use separate trainable Q/K/V parameters. `fuse_qkv=True` means
forward-only weight concatenation into one QKV matmul; it does not create a
single trainable QKV parameter.

Baseline architecture, `seq_len=512`, batch 128, tied embeddings, bf16,
cuDNN attention, measured from step 5:

| config | step time | tok/s | est TFLOP/s | MFU | JAX peak | nvidia-smi peak |
|---|---:|---:|---:|---:|---:|---:|
| forward QKV fusion on, SwiGLU fusion on | 0.2653s | 247.0k | 167.9 | 53.8% | 29.18 GB | 43489 MiB |
| forward QKV fusion off, SwiGLU fusion on | 0.2673s | 245.1k | 166.6 | 53.4% | 29.18 GB | 51681 MiB |

Interpretation: after restoring separate Q/K/V trainable parameters, forward
QKV fusion gives only a tiny speed difference at this shape (`~0.8%`), which is
small enough to treat as noise unless a longer profile proves otherwise.

Old-all architecture, `seq_len=512`, batch 128, tied embeddings, bf16,
cuDNN attention, measured from step 5:

| config | step time | tok/s | est TFLOP/s | MFU | JAX peak | nvidia-smi peak |
|---|---:|---:|---:|---:|---:|---:|
| forward QKV fusion on, SwiGLU fusion on | 0.2941s | 222.8k | 151.5 | 48.6% | 33.91 GB | 43495 MiB |
| forward QKV fusion off, SwiGLU fusion on | 0.2959s | 221.5k | 150.6 | 48.3% | 32.30 GB | 43495 MiB |

Relative to the seq-512 batch-128 baseline with forward QKV fusion, old-all is
about `11%` slower in step wall time, but still has reasonable MFU at this
larger batch size. Forward-only QKV fusion is also only a tiny difference under
old-all (`~0.6%`), again within the noise of these short profiles.

## Next Optimization Candidates

1. In progress: chunked/fused final linear cross entropy. Current AR loss still
   materializes full `(B*T, vocab)` logits; this is likely the largest
   remaining memory lever. PLAN.md says to refactor the model/loss API so the
   model returns final hidden states and the loss owns chunked
   `hidden @ lm_head.weight.T` projection.
2. Profile strict old bundle separately if needed:
   QK-norm + value residual + layernorm/depth scaling, without gating.
3. Use longer profile windows once functionality is stable, ideally 50 measured
   steps after warmup/compile.
4. Add structured config files before launching real sweeps.
5. Continue toward PyTorch/JAX training parity on same data order.
