# Progress

This file tracks concrete work completed in this repository so future runs do
not depend on conversational memory. It should be updated whenever code,
profiling, or conclusions change materially.

## Current Objective

Run the MDLM value-embedding/no-value-residual transfer check after the AR
matrix showed that the stable value-embedding variant is competitive with the
tuned baseline but behind the old bundle.

## User Decisions

- Ignore git for now; the user will handle it later.
- Tie token embeddings and LM head by default.
- Use the fixed small GPT shape for sanity/profiling:
  `d_model=768`, `d_ff=2048`, `n_layers=8`, `n_heads=12`,
  `n_kv_heads=None`, active `vocab_size=8192`, bf16.
- Treat baseline architecture as old intervention flags off.
- Treat the "old architecture" profile as QK-norm + value residual +
  layernorm/depth scaling + per-head attention gating, with weight tying still
  enabled.
- Treat MDLM peak LRs as likely shared with the AR-selected LR family unless a
  run becomes unstable. Do not spend time on a separate MDLM LR sweep; use only
  short sanity probes when changing model/objective shape.
- For the next MDLM run, transfer the AR value-embedding/no-value-residual
  recipe directly: QK-norm on, per-head attention gating on,
  layernorm/depth scaling on, value embedding on, value residual off,
  value-embedding gain off, `lr_mult=2.0`, scalar Adam base LR `0.0005`.

## Implemented Backbone Features

- JAX transformer supports RMSNorm, RoPE, SwiGLU, QK-norm, value residual,
  per-head/elementwise attention gating, optional weight tying, depth/layernorm
  scaling, selectable attention implementation, value embeddings, and hidden
  state return before logits.
- Weight tying is real in JAX: `lm_head.weight` is the exact same `nnx.Param`
  object as `embedding.weight`, and NNX state exposes only `embedding.weight`.
- Value embeddings follow the single PLAN.md value-embedding definition:
  first cache raw first-layer value stream, apply value-residual normalization,
  then add token value embeddings after the residual mixture.
- PLAN.md now keeps only this value-embedding definition; the earlier A/B/C
  alternatives were removed.
- Non-muP initialization is no longer part of the Part 1 ablation matrix; the
  trainer uses the default PyTorch-compatible initialization path.
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
- Active AR ablation configs use `data/climbmix_smoke_8192/`:
  base vocab `8192`, diffusion vocab `8193`, mask token id `8192`.
- Tokenizer sanity checks passed for both the original `32768` setup and the
  active `8192` setup: specials IDs `0..3`, diffusion mask ID equal to base
  vocab size, sample roundtrips matched, token byte stats looked normal.
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

## Chunked Logits/Loss Work

Implementation status:

- Added trainer flags `--loss-impl full|chunked` and `--logit-chunk-size`.
- Full loss keeps the old path: model returns full logits, then CE/z-loss.
- Chunked loss calls `model(..., return_hidden=True)` and computes
  `hidden @ lm_head.weight.T` in chunks over flattened `(B*T)`.
- The chunked final linear CE has a custom VJP. Backward recomputes each
  chunk's logits and accumulates gradients into hidden states and the LM-head
  weight, so full logits should not be saved as a backward residual.
- BPB reporting is not implemented for chunked loss yet; training loss and
  z-loss are implemented.

Correctness:

- `python -m py_compile jax/training/loss.py jax/training/step.py jax/train_ar.py jax/tests/test_training_stack.py` passed.
- `python jax/tests/test_training_stack.py` passed, including:
  chunked linear CE value/gradient parity against full logits, and chunked
  model AR loss parity against full model logits.

Initial A100 profiles, baseline architecture, `seq_len=512`, tied embeddings,
bf16, cuDNN attention, chunk size `4096`:

| loss path | batch | step time | tok/s | MFU | JAX peak | nvidia-smi peak |
|---|---:|---:|---:|---:|---:|---:|
| full logits | 128 | 0.2653s | 247.0k | 53.8% | 29.18 GB | 43489 MiB |
| chunked logits | 128 | 0.3108s | 210.8k | 45.9% | 17.50 GB | 18911 MiB |
| chunked logits | 192 | 0.4477s | 219.6k | 47.8% | 25.54 GB | 37343 MiB |
| full logits | 192 | 0.3767s | 261.0k | 56.8% | 43.07 GB | 61345 MiB |
| chunked logits, chunk 8192 | 192 | 0.4406s | 223.1k | 48.6% | 25.91 GB | 37343 MiB |
| chunked logits, chunk 16384 | 192 | 0.4365s | 225.2k | 49.1% | 27.27 GB | 41439 MiB |
| chunked logits, chunk 16384 | 256 | 0.5661s | 231.5k | 50.4% | 34.57 GB | 41439 MiB |
| chunked logits, chunk 32768 | 256 | 0.5658s | 231.6k | 50.5% | 37.89 GB | 61343 MiB |

Interpretation: chunked loss substantially reduces memory, but `chunk_size=4096`
is slower than full logits at batch 128 and does not recover raw throughput via
larger batches in the first implementation. Larger chunks help only slightly.
For the Phase A target of fixed effective batch size 512 sequences, compare
microbatch/accumulation wall time, not just raw microbatch throughput:

- Full logits microbatch 128 -> grad accumulation 4:
  about `4 * 0.2653 = 1.061s` before any accumulation-specific overhead.
- Chunked logits microbatch 256 -> grad accumulation 2:
  about `2 * 0.566 = 1.13s` before accumulation-specific overhead.

So this first pure-JAX custom-VJP chunked loss saves substantial memory but is
not yet faster for the fixed effective batch target. It may still matter for
longer sequence lengths, old-all/diffusion, or shapes where full logits cannot
fit. Next useful checks are true gradient-accumulation profiles at effective
batch 512 and/or optimizing the chunked loss implementation.

## Diffusion Reference Read

Read `PLAN.md` diffusion section and the main `baby-dLM/` references:
`model_MDLM.py`, `model_bd3lm.py`, `backbone.py`, `block_utils.py`,
`prepare.py`, `train.py`, `README.md`, `WORKFLOW.md`, and the diffusion tests.

Key semantics to port:

- MDLM batch construction samples one integer timestep per sequence, converts
  it through the survival-probability schedule, then independently masks each
  token with probability `1 - a_t`. The model uses unmasked/bidirectional
  attention. The loss is CE against `x0`, averaged only over masked positions.
- BD3LM batch construction samples one integer timestep per block, expands
  the per-block survival probabilities to tokens, masks tokens independently,
  then trains with a dual stream `x_t || x_0`. The transformer returns logits
  only for the noisy `x_t` half and applies CE only on masked positions.
- BD3 train attention mask is the four-quadrant rule:
  noisy-to-noisy block diagonal, noisy-to-clean strict previous clean blocks,
  clean-to-noisy disallowed, and clean-to-clean block-causal. Sampling uses
  the one-stream block-causal mask.
- When `block_len == seq_len`, `forward_train(xt, x0)` must skip dual-stream
  concatenation; otherwise the clean stream would leak. This is explicitly
  tested in `baby-dLM/tests/test_block_diffusion.py`.
- `baby-dLM` generation is simplified confidence-based progressive unmasking:
  already revealed tokens are never re-masked, and top-confidence masked
  positions are committed as the reverse timestep decreases. Training can be
  implemented before generation.
- `baby-dLM/README.md` says this is not the full upstream BD3 reverse process;
  for this repo the first port should match the simplified masked-token CE
  semantics and BD3 masks, then consult upstream only if attention details are
  ambiguous or performance forces it.

JAX changes implied:

1. Generalize the AR training stack into objective-specific batch/loss
   functions for `ar`, `mdlm`, and `bd3lm`.
2. Add JAX diffusion batch builders that operate on clean `(inputs or x0)`
   token batches and return `(xt, x0, supervise_mask)`.
3. Add a masked CE loss path that can consume either full logits or hidden
   states plus chunked final projection, using the existing chunked CE machinery
   with `valid_mask=supervise_mask`.
4. For MDLM, instantiate/use the shared `Transformer` with `is_causal=False`
   and no attention mask.
5. For BD3LM, concatenate `xt` and `x0`, provide repeated positions
   `0..L-1, 0..L-1`, call the shared `Transformer` with `is_causal=False` and
   the dense BD3 mask for correctness first, then slice logits/hidden states to
   the noisy half before loss.
6. Add deterministic JAX tests mirroring `baby-dLM` tests for masking rates,
   MDLM loss-only-masked positions, BD3 mask exactness, no clean-stream leakage
   when `block_len == seq_len`, block0 independence from `x0`, block1
   dependence on prior clean blocks, and chunked/full diffusion-loss parity.
7. After correctness, profile BD3 attention. Dense boolean masks over length
   `2L` may force inefficient attention; if XLA/cuDNN cannot handle the mask
   efficiently, the likely next optimization is a block-aware implementation
   that avoids materializing disallowed quadrants before considering Pallas.

## Diffusion Implementation Bring-Up

User clarification:

- The diffusion timestep fraction should be clipped to a configured window by
  default. Current JAX defaults are `t_min=0.45`, `t_max=0.95`, matching the
  `baby-dLM/train.py` defaults.
- For BD3LM, default diffusion block length is `128`. In code this is
  `--bd3-block-len` / `--block-len`, because `block_size` is already used in
  `baby-dLM` to mean sequence length. If the sequence length is smaller than
  128 in tiny smoke runs, the trainer clamps the effective block length to the
  sequence length unless explicitly set smaller.
- User clarified the intended near-term profiling target is sequence length
  `512`, while BD3 block length remains `128`.

Implemented:

- Added `jax/training/diffusion.py`:
  - `DiffusionConfig`
  - clipped linear/cosine survival schedules
  - MDLM batch construction: one timestep per sequence, iid token masking
  - BD3LM batch construction: one timestep per block, expanded to tokens
  - static transformer call context for AR/MDLM/BD3LM, including repeated
    `0..L-1, 0..L-1` RoPE positions and the dense BD3 train mask.
- Extended `jax/training/loss.py`:
  - `cross_entropy_with_z_loss` now accepts `valid_mask`.
  - chunked final linear CE now accepts an external supervision mask.
  - new `supervised_lm_loss` covers AR-style, MDLM, and BD3LM masked CE.
- Extended `jax/training/step.py`:
  - added jitted supervised train/eval steps and accumulated supervised steps.
- Extended `jax/train_ar.py`:
  - added `--objective/--model ar|mdlm|bd3lm`.
  - diffusion model vocab defaults to `base_vocab_size + 1`; default mask ID is
    `base_vocab_size`, so ClimbMix `32768` becomes diffusion vocab `32769` with
    mask ID `32768`.
  - MDLM uses bidirectional attention with no mask.
  - BD3LM uses dense dual-stream masks for correctness first and slices only
    noisy-stream outputs for loss.
  - logs `supervised_tokens` for diffusion objectives.

Correctness tests added:

- `jax/tests/test_diffusion_stack.py` checks:
  - clipped survival schedule values
  - MDLM masked/unmasked token invariants
  - BD3 fixed high-t masking behavior
  - BD3 context repeated positions and dual-stream mask shape
  - MDLM supervised loss equals manual masked CE
  - chunked/full diffusion masked CE parity
  - BD3 single-block mode skips the clean stream
  - BD3 dual-stream mask prevents clean-target leakage into noisy block 0 while
    allowing noisy block 1 to depend on prior clean block 0.

Validation run so far:

- `python -m py_compile jax/training/loss.py jax/training/diffusion.py jax/training/step.py jax/train_ar.py jax/tests/test_diffusion_stack.py` passed.
- `python jax/tests/test_diffusion_stack.py` passed.
- `python jax/tests/test_training_stack.py` passed.
- `python jax/tests/test_parity.py` passed after diffusion changes.
- `python jax/tests/test_parity_extras.py` passed after diffusion changes.
- Tiny synthetic MDLM train loop passed:
  `--objective mdlm --batch-size 2 --context-length 16 --vocab-size 32
  --d-model 32 --d-ff 64 --n-layers 2 --n-heads 4 --overfit-batch`.
  The preview showed model input shape `(2, 16)`, target shape `(2, 16)`,
  mask rate `0.75` at eval timestep fraction `0.6` under the clipped default
  window, and supervised-token counts in the expected range.
- Tiny synthetic BD3LM train loop passed:
  `--objective bd3lm --bd3-block-len 4 --batch-size 2 --context-length 16
  --vocab-size 32 --d-model 32 --d-ff 64 --n-layers 2 --n-heads 4
  --overfit-batch`.
  The preview showed model input shape `(2, 32)`, target shape `(2, 16)`,
  doubled `x_t || x_0` layout, mask rate `0.75`, and supervised-token counts
  in the expected range.
- Added `jax/inspect_diffusion.py`, a reusable "vibe test" script that prints
  schedule values, clean/masked examples, logits statistics, entropy, top-k
  token IDs, position-logit cosine similarity, mask-token ranks, and a BD3
  clean-block perturbation check.
- `python jax/inspect_diffusion.py --seq-len 16 --bd3-block-len 4
  --batch-size 3 --base-vocab-size 64 --d-model 48 --d-ff 96 --n-layers 2
  --n-heads 4` passed on CPU. It showed finite logits for MDLM and BD3LM.
  The BD3 perturbation check was correct: changing clean block 0 caused exactly
  zero max-logit difference in noisy block 0 and large differences from noisy
  block 1 onward.
- Printed attention masks at sequence length 4:
  - MDLM effective mask is full bidirectional `4x4`.
  - BD3 with `block_len=4` is the single-block fast path; the trainer skips
    dual-stream concat and uses full bidirectional `x_t` attention.
  - BD3 with `block_len=2` has the expected `8x8` dual-stream mask: noisy
    block 0 sees only noisy block 0; noisy block 1 sees noisy block 1 plus
    clean block 0; clean stream is block-causal and never attends to noisy.
- Verified that the actual model uses those masks, not just that the mask
  arrays look right. Added regression tests that perturb source regions and
  compare output logits:
  - MDLM: changing token 3 changes earlier token-0 logits, confirming
    bidirectional attention behavior.
  - BD3 `L=4, block_len=2`: changing clean block 0 leaves noisy block 0
    logits exactly unchanged and changes noisy block 1; changing clean block 1
    leaves noisy blocks 0 and 1 unchanged; changing noisy block 1 leaves noisy
    block 0 and the clean stream unchanged; changing noisy block 0 leaves noisy
    block 1 and the clean stream unchanged.
  - `python jax/tests/test_diffusion_stack.py` passed after adding these
    through-model perturbation checks.
- Tiny ClimbMix GPU smoke runs passed with `nvidia-smi` monitoring:
  - MDLM: `--train-path data/climbmix_smoke/tokens/train --eval-path
    data/climbmix_smoke/tokens/val --objective mdlm --batch-size 4
    --context-length 64 --vocab-size 32768 --d-model 64 --d-ff 128
    --n-layers 2 --n-heads 4 --dtype bfloat16 --eval-batches 1
    --overfit-batch --max-steps 3`.
  - BD3LM: same, with `--objective bd3lm --bd3-block-len 16`.
  - Both used diffusion vocab `32769` and mask ID `32768`; previews showed
    plausible masked ClimbMix examples and finite losses/grad norms.
  - Background `nvidia-smi` peak was only about `975 MiB`, so this validates
    plumbing and CUDA execution, not performance.
- Systematically checked noised data fed to the model on a `64 x 512`
  ClimbMix batch:
  - Clean batch shape `(64, 512)`, token min/max were in range, and
    `mask_token_id=32768` did not appear in clean data.
  - The underlying loader's AR `(x, y)` shift relation was also checked as a
    loader sanity check, but diffusion objectives ignore `y`: MDLM and BD3LM
    use the clean `x0 = x` directly as the denoising target with no logit or
    target shift.
  - For fixed timesteps, observed mask ratios matched the clipped schedule:
    `t=1` expected `0.45`, observed `0.4519`; `t=45` expected `0.45`,
    observed `0.4495`; `t=60` expected `0.60`, observed `0.6001`;
    `t=95` expected `0.95`, observed `0.9504`; `t=100` expected clipped
    `0.95`, observed `0.9510`.
  - These checks passed for both MDLM and BD3LM. For BD3 fixed `t=60`,
    per-block observed rates with `block_len=128` were
    `[0.5961, 0.5988, 0.6008, 0.6047]`.
  - For random-t BD3, first few per-sample block mask rates differed across
    blocks as expected because BD3 samples one timestep per block.
- Optimized the BD3 full-loss path: when `output_length` is set for dual-stream
  BD3, the loss now requests hidden states, slices to the noisy stream, and
  projects only the supervised half. This avoids materializing clean-stream
  `(B*T, vocab)` logits. `python jax/tests/test_diffusion_stack.py` and
  `python jax/tests/test_training_stack.py` passed after the change.
- Added gradient/optimizer sanity checks:
  - Diffusion loss/gradients are unchanged when targets are changed only at
    unsupervised positions, confirming MDLM/BD3 use no shifted target and CE is
    applied only where `supervise_mask=True`.
  - Direct NorMuon+AdamW transformation check verifies optimizer state count
    increments and nonzero updates are produced for Muon matrix params
    (`W_q`, `W_k`, `W_v`) and AdamW params (`embedding.weight`, RMSNorm gamma).
  - `python jax/tests/test_diffusion_stack.py` and
    `python jax/tests/test_training_stack.py` passed with these checks.

## First Diffusion Seq-512 Profiles

First non-toy A100 profiles used the 70M-ish shape:
`d_model=768`, `d_ff=2048`, `n_layers=8`, `n_heads=12`, bf16, tied embeddings,
NorMuonCWD+AdamW, ClimbMix smoke tokens, `seq_len=512`, microbatch `8`.

- MDLM, full logits:
  - `model_sequence_length=512`, `tokens_per_optimizer_step=4096`.
  - Compile plus first step `40.476s`.
  - Steady-state avg from step 4: `0.0829s`, `49.4k tok/s`.
  - Estimated `33.6 TFLOP/s`, `10.8% MFU`.
  - JAX peak HBM `4.29 GB`.
- BD3LM, `block_len=128`, dense dual-stream mask, full loss with noisy-stream
  hidden slice before projection:
  - `model_sequence_length=1024`, `tokens_per_optimizer_step=4096`.
  - Compile plus first step `42.719s`.
  - Steady-state avg from step 4: `0.1251s`, `32.8k tok/s`.
  - Estimated `47.0 TFLOP/s`, `15.1% MFU`.
  - JAX peak HBM `9.29 GB`.
- A shared background `nvidia-smi` monitor observed peak memory about
  `10723 MiB` across both runs.

Interpretation: dense BD3 at this small batch compiles and runs, but it is
about `1.5x` slower wall-clock than MDLM at the same clean-token batch because
the model sequence is doubled and the dense mask path is more expensive. The
memory increase from MDLM to BD3 is more than 2x by JAX peak HBM at this shape.

## Diffusion Length And Batch Scaling

All numbers here use the 70M-ish shape, bf16, tied embeddings,
NorMuonCWD+AdamW, ClimbMix smoke tokens, and no old-architecture flags unless
stated otherwise. Diffusion uses clean sequence length as the token accounting
unit; BD3 internally doubles the model sequence length with `x_t || x_0`.

Sequence-length sweep at microbatch `8`:

| objective | clean seq len | model seq len | step time | clean tok/s | JAX peak HBM |
| --- | ---: | ---: | ---: | ---: | ---: |
| AR | 512 | 512 | `0.0560s` | `73.1k` | `4.38 GB` |
| MDLM | 512 | 512 | `0.0829s` | `49.4k` | `4.29 GB` |
| BD3LM | 512 | 1024 | `0.1251s` | `32.8k` | `9.29 GB` |
| AR | 768 | 768 | `0.0697s` | `88.2k` | `7.06 GB` |
| MDLM | 768 | 768 | `0.1036s` | `59.3k` | `6.98 GB` |
| BD3LM | 768 | 1536 | `0.1756s` | `35.0k` | `16.69 GB` |
| AR | 1024 | 1024 | `0.0850s` | `96.4k` | `9.99 GB` |
| MDLM | 1024 | 1024 | `0.1152s` | `71.1k` | `9.99 GB` |
| BD3LM | 1024 | 2048 | `0.2243s` | `36.5k` | `26.93 GB` |

Interpretation: BD3 wall time relative to MDLM worsens with clean sequence
length (`1.51x`, `1.69x`, `1.95x`), consistent with the doubled model context.
The memory growth is clearly steeper for BD3, but not the catastrophic `4x`
attention-score materialization that would show up if every dense attention
matrix were fully retained at bf16/fp32 scale. Native JAX attention is doing
something reasonably memory-aware, but the dense dual-stream path is still much
heavier than MDLM.

Microbatch scaling at clean `seq_len=512`:

| objective | loss impl | microbatch | status | step time | clean tok/s | JAX peak HBM |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| MDLM | full | 32 | fit | `0.1573s` | `104k` | `13.49 GB` |
| MDLM | full | 64 | fit | `0.2483s` | `132k` | `24.99 GB` |
| MDLM | full | 128 | OOM | allocation `43.22 GiB` | - | - |
| MDLM | full | 192 | OOM | allocation `47.11 GiB` | - | - |
| MDLM | chunked, 1k | 128 | fit | `0.4993s` | `131k` | `36.41 GB` |
| MDLM | chunked, 16k | 192 | OOM | allocation `51.23 GiB` | - | - |
| BD3LM | full | 32 | fit | `0.3146s` | `52.1k` | `31.79 GB` |
| BD3LM | chunked, 1k | 32 | fit | `0.3110s` | `52.7k` | `29.49 GB` |
| BD3LM | full | 64 | OOM | allocation `53.34 GiB` | - | - |
| BD3LM | chunked, 16k | 64 | OOM | allocation `52.62 GiB` | - | - |
| BD3LM | chunked, 1k | 64 | OOM | allocation `51.28 GiB` | - | - |

The important comparison is fixed effective batch `512`, because changing the
microbatch only changes gradient accumulation count. Approximate optimizer-step
wall times from the measured microsteps:

| objective | loss impl | microbatch | grad accum for 512 | optimizer-step wall time |
| --- | --- | ---: | ---: | ---: |
| MDLM | full | 32 | 16 | `2.52s` |
| MDLM | full | 64 | 8 | `1.99s` |
| MDLM | chunked, 1k | 128 | 4 | `2.00s` |
| BD3LM | full | 32 | 16 | `5.03s` |
| BD3LM | chunked, 1k | 32 | 16 | `4.98s` |

Chunked logits conclusion for training:

- Chunked logits are mathematically correct by tests and can reduce the reported
  BD3 batch-32 peak from `31.79 GB` to `29.49 GB`.
- Chunking lets MDLM fit microbatch `128`, but the microstep becomes about `2x`
  slower, so fixed-effective-batch wall time is essentially tied with MDLM full
  logits at microbatch `64`.
- Chunking does not solve BD3 microbatch `64`: full, chunk-16k, and chunk-1k all
  OOM on the same scale of `~51-53 GiB` temporary allocation.
- Therefore chunked logits are not currently a useful training optimization for
  the planned seq512 diffusion runs. They may still be useful for eval or for a
  future custom fused linear-CE implementation, but the current pure-JAX
  chunked custom-VJP path is not the bottleneck to optimize next.

Practical batch setting for fixed effective batch `512` at clean `seq_len=512`:

- MDLM: use full logits with microbatch `64`, grad accumulation `8`.
- BD3LM: use full logits with microbatch `32`, grad accumulation `16`. Chunked
  logits at the same microbatch is effectively neutral and can be left off for
  simplicity.

The OOM behavior means a naive linear extrapolation from small-batch peak HBM is
not reliable. Runs fail when live buffers plus a large XLA temporary/workspace
exceed 80 GB, even if sampled `nvidia-smi` usage before the allocation is much
lower. Since chunked logits does not move the BD3 batch-64 OOM, the likely
next memory target is the dense dual-stream attention/backward path or XLA
workspace choices, not the final CE logits.

Meaningful next inspections/tests before profiling:

1. Print small MDLM/BD3 logits statistics by position and stream:
   mean/std/min/max, top-k token IDs, entropy, and pairwise cosine similarity
   between positions. Look for repeated constant logits, NaNs, or the mask token
   dominating every position at init.
2. For BD3, compare logits after changing `x0` block 0:
   noisy block 0 should be unchanged, noisy block 1 should change, and later
   blocks should change more strongly.
3. For MDLM, change an unmasked future token and verify bidirectional logits
   can change; for AR, the same check should not affect earlier positions.
4. Sweep fixed timesteps (`t/T` low, mid, high) and print mask rates plus loss
   denominators to confirm clipping and supervision counts are sane.
5. Run a tiny fixed-clean-batch overfit for MDLM and BD3LM with random masks:
   loss should trend down, grad norm should stay finite, and no position should
   collapse to identical logits across the batch.
6. Run small ClimbMix batches through MDLM and BD3LM and print decoded clean
   vs masked examples to catch token-ID or mask-ID mistakes.
7. Profile BD3 with dense mask at small and medium sequence lengths to see
   whether `jax.nn.dot_product_attention` is materializing dense attention
   state before optimizing kernels.

## BD3 Blocked Attention Prototype

Implemented an experimental pure-JAX BD3 attention decomposition, enabled with
`--bd3-attention blocked` and leaving the dense mask path as the default.

Reasoning:

- The BD3 train mask over `x_t || x_0` allows only `L^2 + L * block_len`
  query/key token pairs out of the dense `4L^2` dual-stream matrix. For the
  current `L=512`, `block_len=128` target, only `31.25%` of the dense
  dual-stream attention matrix is semantically live.
- A dense boolean mask passed to `jax.nn.dot_product_attention` preserves
  correctness but does not provide FlexAttention-style block metadata. cuDNN
  may still be memory-efficient/flash-style, but it has no reason to skip the
  arbitrary disallowed BD3 KV blocks.
- PyTorch FlexAttention gets its win from `BlockMask`: the mask rule is
  compiled into sparse block metadata, so the kernel can skip masked KV blocks
  rather than simply writing `-inf` into a dense logical score matrix.
- The first JAX approximation does not write a custom GPU kernel. Instead,
  `bd3_block_sparse_attention(q, k, v, block_len=...)` splits the dual stream
  into exact per-block full-attention calls:
  noisy block `b` attends to noisy block `b` plus clean blocks `< b`; clean
  block `b` attends to clean blocks `<= b`. This computes exactly the allowed
  BD3 pairs while reusing normal `jax.nn.dot_product_attention` for each
  rectangular subproblem.

Files changed:

- `jax/transformer/attention.py`: added `bd3_block_sparse_attention` and an
  opt-in `bd3_block_len` path in MHSA.
- `jax/training/diffusion.py`: `ModelContext` now carries optional
  `bd3_block_len`; `make_model_context(..., bd3_attention="blocked")` returns
  repeated positions with no dense mask.
- `jax/train_ar.py`: added `--bd3-attention dense|blocked`.
- Training/eval/loss plumbing now forwards `bd3_block_len` through model calls.

Validation:

- `python -m py_compile jax/transformer/attention.py jax/transformer/transformer.py jax/training/diffusion.py jax/training/loss.py jax/training/step.py jax/train_ar.py jax/tests/test_diffusion_stack.py`
  passed.
- `python jax/tests/test_diffusion_stack.py` passed, including direct
  dense-mask vs blocked-attention parity and full tiny-model output parity.
- `python jax/tests/test_training_stack.py` passed.
- `python jax/tests/test_parity_extras.py` passed.
- Tiny synthetic BD3 training smoke with `--bd3-attention blocked` passed on CPU.

Next profile:

- Compare dense vs blocked BD3 at clean `seq_len=512`, `block_len=128`,
  microbatch `32`, bf16, `--attention-impl cudnn`.
- Then test whether blocked attention plus `--grad-checkpoint-layers 8` allows
  BD3 microbatch `64`, because reducing grad accumulation from `16` to `8` is
  the practical target for fixed effective batch `512`.

## cuDNN Attention Retest and Microbatch Ceiling

Retested the seq512 baseline architecture with explicit
`--attention-impl cudnn`, full logits, tied embeddings, bf16, NorMuonCWD+AdamW,
and effective batch target `512` clean sequences.

Implementation detail:

- AR uses `is_causal=True` and no explicit mask.
- MDLM uses bidirectional attention and no explicit mask.
- BD3 dense mode uses a broadcast dense mask of shape `(1, 1, 2L, 2L)`, not
  `(B, H, 2L, 2L)`.
- Added a default persistent JAX compilation cache to `jax/train_ar.py`:
  `/tmp/sample_efficient_gpt_jax_cache`. It only helps repeated identical
  shape/static-argument launches; new batch sizes still compile once.

Full-logit cuDNN ceiling at clean `seq_len=512`:

| objective | largest divisor microbatch | grad accum for 512 | microstep time | optimizer-step wall | 5k-step wall | clean tok/s | JAX peak HBM | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AR | 128 | 4 | `0.2640s` | `1.06s` | `1.47h` | `248k` | `29.02 GB` | b256 OOM, b512 OOM |
| MDLM | 128 | 4 | `0.3463s` | `1.39s` | `1.92h` | `189k` | `29.02 GB` | b256 OOM |
| BD3LM | 128 | 4 | `0.7386s` | `2.95s` | `4.10h` | `88.7k` | `43.42 GB` | b192 OOM, so b256 is not viable |

Specific OOMs:

- AR b512 full logits: `87.25 GiB` allocation.
- AR b256 full logits: `46.81 GiB` allocation.
- MDLM b256 full logits: `46.81 GiB` allocation.
- BD3LM b192 full logits: `49.73 GiB` allocation.

cuDNN dense vs blocked BD3 at b32:

| BD3 attention | microbatch | microstep time | clean tok/s | JAX peak HBM |
| --- | ---: | ---: | ---: | ---: |
| dense cudnn | 32 | `0.2404s` | `68.1k` | `11.97 GB` |
| blocked cudnn | 32 | `0.2391s` | `68.5k` | `13.32 GB` |

The pure-JAX blocked decomposition is not a useful win at this shape. cuDNN's
dense masked path is already strong, and splitting into several DPA calls
increases compile complexity without improving steady-state throughput.

Before-vs-after cuDNN, fixed effective batch `512`:

| objective | earlier best setting | earlier optimizer step | cuDNN best setting | cuDNN optimizer step | result |
| --- | --- | ---: | --- | ---: | --- |
| AR | full b128 x accum4 | `1.061s` | full b128 x accum4 | `1.056s` | essentially unchanged |
| MDLM | full b64 x accum8 | `1.99s` | full b128 x accum4 | `1.39s` | clearly better |
| BD3LM | full b32 x accum16 | `5.03s` | full b128 x accum4 | `2.95s` | clearly better |

Chunked-CE retest under cuDNN:

| objective | loss impl | microbatch | status | microstep time | optimizer-step wall for 512 | JAX peak HBM | interpretation |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| AR | chunked, 32768 | 512 | OOM | - | - | - | `54.25 GiB` allocation |
| AR | chunked, 32768 | 256 | fit | `0.5592s` | `1.12s` | `37.68 GB` | more memory, slower than full b128 |
| MDLM | chunked, 32768 | 256 | fit | `0.7432s` | `1.49s` | `37.67 GB` | more memory, slower than full b128 |
| BD3LM | chunked, 32768 | 256 | OOM | - | - | - | `56.51 GiB` allocation |

Conclusion:

- cuDNN attention is a major win for MDLM and BD3, and should be the default on
  the A100.
- AR at seq512 is not attention-bound; the final vocabulary projection/loss is
  the practical limiter.
- The current pure-JAX chunked CE can increase the fitting microbatch for
  AR/MDLM but does not improve fixed-effective-batch wall time.
- Smaller vocab would reduce the bottleneck linearly in `vocab_size`, but it
  would change the tokenizer/text-per-token/BPB setup. Keep vocab `32768` for
  the main experiments and treat smaller vocab only as a profiling or tokenizer
  ablation.
- The next real optimization target is a faster fused linear cross entropy or a
  better custom VJP/Pallas implementation, not the current chunked CE.

## Vocab 8192 Profiling Ablation

Retrained a separate smoke tokenizer/data directory with base vocab size
`8192`:

- data root: `data/climbmix_smoke_8192`
- diffusion mask token id: `8192`
- train shard token count: `64,600,095`
- old vocab32768 train shard token count: `55,468,937`
- token-count ratio for the same raw text: `1.1646x`

Sanity checks:

- Token IDs in the train/val arrays are in range `0..8191`.
- Encode/decode roundtrips passed for English, code, numbers, and Unicode.
- Loader batches at sizes `4`, `17`, and `128` had the expected AR shift and
  no out-of-range IDs.

Seq512, cuDNN attention, full logits, tied embeddings, bf16:

| objective | vocab | best microbatch | accum for 512 | microstep time | optimizer-step wall | 5k-step wall | clean tok/s | est TFLOP/s | est MFU | JAX peak HBM | OOM notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AR | 32768 | 128 | 4 | `0.2640s` | `1.056s` | `1.47h` | `248k` | `168.7` | `54.1%` | `29.02 GB` | b256/b512 OOM |
| AR | 8192 | 256 | 2 | `0.3925s` | `0.785s` | `1.09h` | `334k` | `151.3` | `48.5%` | `36.38 GB` | b512 OOM |
| MDLM | 32768 | 128 | 4 | `0.3463s` | `1.385s` | `1.92h` | `189k` | `128.6` | `41.2%` | `29.02 GB` | b256 OOM |
| MDLM | 8192 | 256 | 2 | `0.5084s` | `1.017s` | `1.41h` | `258k` | `116.8` | `37.4%` | `36.38 GB` | b512 OOM |
| BD3LM | 32768 | 128 | 4 | `0.7386s` | `2.954s` | `4.10h` | `88.7k` | `127.3` | `40.8%` | `43.42 GB` | b192 OOM |
| BD3LM | 8192 | 256 | 2 | `1.1562s` | `2.312s` | `3.21h` | `113k` | `111.3` | `35.7%` | `62.03 GB` | b512 OOM |

Token-normalized speedups from vocab32768 to vocab8192:

- AR: `1.35x` faster per effective token batch.
- MDLM: `1.36x` faster per effective token batch.
- BD3LM: `1.28x` faster per effective token batch.

Text-normalized caveat:

The 8192 tokenizer emits `1.1646x` more tokens on the same raw training shard.
If comparing equal raw-text throughput instead of equal token sequence length,
the rough adjusted speedups are smaller:

- AR: `~1.16x`.
- MDLM: `~1.17x`.
- BD3LM: `~1.10x`.

Interpretation:

- Smaller vocab does help the final projection/loss bottleneck and lets
  AR/MDLM move from microbatch `128` to `256`.
- BD3 also moves from microbatch `128` to `256`, but its model sequence length
  is still doubled, so attention/backward memory remains important. The b256
  run uses much more of the A100 (`~62 GB` JAX peak), which is closer to the
  target utilization.
- Estimated MFU goes down with vocab8192 because the model has fewer embedding
  and effective LM-head FLOPs in the estimator. Wall time improves, but the
  denominator-normalized utilization estimate drops from `54.1%` to `48.5%` for
  AR, `41.2%` to `37.4%` for MDLM, and `40.8%` to `35.7%` for BD3.
- User decided to proceed with vocab8192 for the current AR ablation matrix.
  Scientific comparisons should keep this tokenizer fixed within the matrix.

## AR Ablation Configs

Updated `PLAN.md` and created a concrete config tree under `jax/configs/`:

- `jax/configs/data/climbmix_8192.yaml`
- `jax/configs/model/gpt_small_70m.yaml`
- `jax/configs/optimizer/normuon_adamw.yaml`
- `jax/configs/schedule/ws_100_5k.yaml`
- `jax/configs/experiments/ar_baseline.yaml`
- `jax/configs/experiments/ar_old_bundle.yaml`
- `jax/configs/experiments/ar_value_embedding.yaml`

All three AR experiment configs use:

- `vocab_size=8192`, train/eval paths under `data/climbmix_smoke_8192/tokens/`
- `seq_len=512`, microbatch `256`, grad accumulation `2`, effective batch
  `512`
- tied embeddings, bf16, cuDNN attention, full logits, QKV/SwiGLU forward
  fusion enabled
- NorMuonCWD+AdamW with base `adam_lr=0.007`, `muon_lr=0.015`, LR sweep
  multipliers `[0.25, 0.5, 1.0, 2.0, 4.0]`
- schedule `100` warmup steps plus `5000` constant steps (`5100` total)

The configs are directly consumable by `train_ar.py` through `--config`, which
loads the top-level `train_args` mapping. `--lr-mult` scales AdamW and Muon
peak LRs together for sweeps.
When `output_dir` is set, `train_ar.py` writes `resolved_config.json` there at
launch time.
Raw code defaults were also aligned to vocab8192:
`train_ar.py --vocab-size`, `prepare_climbmix.py BASE_VOCAB_SIZE`, and
`DiffusionConfig.mask_token_id`.

Initialization decision:

- Non-muP initialization was dropped from Part 1. There is no experiment-level
  `init_mode` argument or non-muP run config.

Validation:

- `python -m py_compile jax/train_ar.py jax/transformer/core.py jax/transformer/attention.py jax/transformer/transformer.py`
- `python -m py_compile jax/train_ar.py jax/data/prepare_climbmix.py jax/training/diffusion.py`
- `python jax/train_ar.py --config ... --max-steps 0 --eval-batches 0`
  succeeded for all four AR experiment configs.
- YAML load check over `jax/configs/**/*.yaml` passed.
- `python jax/tests/test_parity.py` passed.
- `python jax/tests/test_parity_extras.py` passed.
- `python jax/tests/test_training_stack.py` passed.
- `python jax/tests/test_diffusion_stack.py` passed.

## Karpathy `train.py` Optimizer/Init Comparison

Checked only `karpathy/train.py` as the reference for Karpathy-style value
embeddings, initialization, and optimizer scaling.

Conclusion: keep the implementation anchored to the `pytorch/` optimizer
constants unless an ablation explicitly says otherwise. Karpathy's `train.py`
has useful LR grouping ideas, but its initialization and untied head recipe do
not transfer directly to our tied-embedding JAX baseline.

Current JAX LR families:

- table Adam: token embedding / tied LM head and value-embedding tables
- scalar Adam: RMSNorm gains, QK gains, value-residual scalars, and other
  vector/scalar params
- Muon: the remaining matrix weights, with the PyTorch-style shape LR multiplier

Current non-LR defaults: Adam betas `(0.95, 0.99)`, Adam eps `1e-8`, Muon
cautious WD `1e-4`, momentum warmup `0.85 -> 0.95`.

## Baseline AR LR Sweep Notes

Setup: baseline AR, tied embeddings, old interventions off, vocab8192, seq512,
effective batch 512, cuDNN attention, full logits, W&B group
`ar_lr_sweep_baseline*`. Current probes are 300 steps: 100 warmup + 200
constant.

Corrected PyTorch-default 300-step probes:

| table | scalar | muon | result |
| --- | --- | --- | --- |
| `0.01` | `0.005` | `0.04` | best so far: loss `3.504`, grad `0.250`, z `98`, `0.817s/step`, MFU `46.6%` |
| `0.00333` | `0.005` | `0.04` | tied on CE but worse z: loss `3.505`, grad `0.176`, z `~136`, `0.807s/step`, MFU `47.1%` |
| `0.03` | `0.005` | `0.04` | not top-2: loss `3.536`, grad `0.407`, z `~87`, `0.815s/step`, MFU `46.7%` |
| `0.01` | `0.0005` | `0.04` | close but slightly worse: loss `3.513`, grad `0.309` |
| `0.01` | `0.005` | `0.004` | rejected; step-100 loss `5.536`, too slow |
| `0.01` | `0.005` | `0.1` | fast early, then flattened/drifted: final loss `3.874`, grad `0.285` |
| `0.01` | `0.005` | `0.4` | rejected; grad spikes above `30` |

Older probes before the PyTorch-default correction are qualitative only. They
showed that very high tied-table LR is dangerous: `table=0.03-0.04` was already
unstable under the old constants, and `table=0.6` blew up immediately.

Current interpretation: `table=0.01`, `scalar=0.005`, `muon=0.04` remains the
best completed baseline bracket. `muon=0.1` is not mainly a weight-decay issue:
with `muon_wd=1e-4`, the direct decay term is only `lr * wd = 1e-5` per step
before masking. The likely issue is simply that the normalized Muon matrix
update is too large after the loss has entered the low-gradient regime.

If we return to longer AR baseline probes, use these two candidates:

- Candidate A: `table=0.01`, `scalar=0.005`, `muon=0.04`.
- Candidate B: `table=0.00333`, `scalar=0.005`, `muon=0.04`.

Do not spend more short-run time on the baseline AR LR bracket unless a 2K run
shows a contradiction.

## Multi-GPU Data Parallel Bring-Up

Added an opt-in pmap data-parallel path to the trainer:

- CLI flags: `--data-parallel` and `--num-devices`.
- `--batch-size` is the global microbatch size per accumulation step.
- The trainer reshapes batches to `[num_devices, per_device_batch, ...]`.
- Gradients are averaged across devices with `lax.pmean` before clipping and
  the NorMuon+AdamW update.
- Metrics are averaged across devices before logging.
- Existing single-device JIT path is unchanged unless `--data-parallel` is set.

The expected default experiment hardware is now 2 A100s. For the target
fixed-effective-batch regime, launch with:

```text
--data-parallel --num-devices 2 --batch-size 512 --grad-accum-steps 1
```

This means 256 examples/GPU, no gradient accumulation, and the same effective
batch of 512 sequences. It should be faster than the current single-GPU
`batch_size=256, grad_accum=2` path because it replaces the second microstep
with one gradient all-reduce. If 4 GPUs are available later, use
`--num-devices 4 --batch-size 512 --grad-accum-steps 1` for 128 examples/GPU.

Validation so far:

- `python -m py_compile jax/train_ar.py jax/training/step.py` passed.
- CPU pmap smoke passed for AR with `--grad-accum-steps 1`.
- CPU pmap smoke passed for AR with `--grad-accum-steps 2`.
- CPU pmap smoke passed for MDLM with `--grad-accum-steps 1`.
- Simulated 4-device CPU pmap passed for AR with `--num-devices 4`.
- Simulated 4-device CPU pmap passed for MDLM with `--num-devices 4`.
- Simulated 4-device CPU pmap passed for BD3LM blocked attention with
  `--num-devices 4`.
- Real-GPU pmap smoke passed on the current 1-A100 VM with
  `--data-parallel --num-devices 1`.
- Existing `jax/tests/test_training_stack.py` and
  `jax/tests/test_diffusion_stack.py` still pass on CPU after the pmap changes.

Still needed when the VM exposes multiple GPUs:

- Confirm `jax.local_device_count()` sees 2 GPUs.
- Smoke AR on real GPUs with tiny synthetic data and `--num-devices 2`.
- Compare AR fixed effective batch 512:
  single GPU `batch_size=256, grad_accum=2` versus 2-GPU
  `batch_size=512, grad_accum=1`.
- Repeat the same fixed-effective comparison for MDLM.

Experiment configs now default to the 2-GPU shape: `data_parallel=true`,
`num_devices=2`, `batch_size=512`, and `grad_accum_steps=1`.

## MDLM Baseline Notes

Single-A100 MDLM probes used the same effective batch as the target runs:
`batch_size=256`, `grad_accum=2`, effective batch 512. Compare these loss/eval
metrics directly with future 2-GPU runs, but compare wall time separately
because the accumulation and hardware shape differ.

| setting | result |
| --- | --- |
| `table=0.01`, `scalar=0.005`, `muon=0.04`; single A100; `batch_size=256`, `grad_accum=2`; 300 steps | final train loss `5.103`, last fixed-t eval loss `4.811`, grad `0.394`, z `97.8`, avg measured step `1.987s`, clean tok/s `132k`, MFU `19.2%`, peak HBM `61.14 GB` |
| `table=0.003`, `scalar=0.005`, `muon=0.04`; same shape; 300 steps | final train loss `5.093`, last fixed-t eval loss `4.778`, grad `0.310`, z `105.7`, avg measured step `2.059s`, clean tok/s `127k`, MFU `18.5%`, peak HBM `61.23 GB` |

Interpretation: `table=0.003` and `table=0.01` are in the same performance
band for MDLM. The slightly better eval at `0.003` is not enough to justify a
separate MDLM-specific LR search. For now, carry the AR LR candidates into MDLM
and spend experiment time on architecture/objective comparisons. If a future
change needs a stability check, use a cheap 100-step probe with 10 warmup steps
and 90 constant steps.

Why MDLM is slower per step than AR at the same clean sequence length:

- AR uses causal attention, so the attention score/value work is triangular.
- MDLM is bidirectional with no attention mask, so attention pays the full
  `T x T` work.
- The final vocab projection/loss and most MLP work are similar between AR and
  MDLM; MDLM-specific token noising and masked CE bookkeeping are small.
- The measured MFU is lower for MDLM because the full-attention path is more
  expensive while still not dominating enough to make the step as matmul-dense
  as large AR microbatches.

## Checkpoint/W&B Artifact Status

- W&B model artifact upload/download was smoke-tested with a tiny linear
  regression checkpoint on 2026-04-25. The downloaded artifact matched the
  original weights exactly by array comparison and SHA256.
- The temporary smoke script was deleted after the check.
- `train_ar.py` now saves real checkpoints. Each checkpoint directory contains
  `model.msgpack`, `optimizer.msgpack`, `metadata.json`, and
  `resolved_config.json`.
- Checkpoint metadata records the step, metrics, RNG states, and SHA256 hashes
  for model and optimizer state files.
- If `--output-dir` is set, the default local checkpoint directory is
  `<output_dir>/checkpoints`. The trainer saves `final` by default, saves `best`
  when eval loss is available and improves, and supports sparse periodic saves
  through `--checkpoint-interval`.
- W&B checkpoint artifacts are enabled by default when W&B is enabled. The
  trainer uploads final and best local checkpoints as model artifacts. Disable
  this with `--no-wandb-checkpoints`.
- Restore paths:
  - local: `--restore-checkpoint path/to/checkpoint_dir`
  - W&B: `--restore-wandb-artifact artifact-name:alias`
- Verified:
  - unit round-trip restores every model parameter and optimizer-state leaf.
  - actual `train_ar.py` local save and local restore smoke passed.
  - actual `train_ar.py` W&B final-checkpoint artifact upload passed.
  - actual `train_ar.py --restore-wandb-artifact
    train-ar-checkpoint-smoke-checkpoint-final:final` downloaded the artifact,
    restored step `0`, resumed at step `1`, and saved a new final checkpoint.
  - simulated 2-device CPU pmap checkpoint save passed, and that checkpoint
    restored into a non-pmap trainer smoke.

## Multi-GPU Code Review Fixes

Reviewed the data-parallel path in `jax/training/step.py` and
`jax/train_ar.py` and landed correctness, eval-memory, and reporting fixes.

Sum-form supervised loss reduction:

- The previous supervised DP/accumulated paths used `pmean` of per-shard
  per-microbatch *means*. That is not the intended diffusion objective once
  MDLM/BD3 are treated as unweighted masked-token CE sums.
- Fix: added `supervised_lm_loss_sums` in `jax/training/loss.py` which
  returns `(total_sum, {loss_sum, z_loss_sum, valid_count})`. The
  differentiable scalar is now a sum, not a mean, so each step type can
  do its own correct reduction. `valid_count` is mask-derived (no param
  dependency), so differentiating `total_sum = total_mean * valid_count`
  yields the gradient of the true sum-form loss when `valid_count > 0`,
  and zero when `valid_count == 0`.
- Diffusion training now uses:
  `sum(masked CE) / (batch_size * seq_len * ((t_min + t_max) / 2))`,
  with `batch_size` interpreted as the effective optimizer-step batch
  (`global_microbatch * grad_accum_steps`). The same denominator is applied
  to the z-loss sum. Under the default `t_min=0.45`, `t_max=0.95`, this is
  division by expected mask rate `0.7`, not by the actual sampled masked-token
  count.
- Refactored four supervised step functions in `jax/training/step.py`:
  - `train_step_supervised`: divides gradient/metric sums by the provided
    loss normalizer, falling back to local `valid_count` if none is supplied.
  - `train_step_supervised_accumulated`: sums grads and counts across
    accum microsteps, then divides by the provided effective-batch normalizer.
  - `train_step_supervised_data_parallel`: `psum`s grads and counts
    across devices, then divides by the global loss normalizer.
  - `train_step_supervised_accumulated_data_parallel`: sums locally over
    accum microsteps, `psum`s across devices, then divides by the
    global effective-batch loss normalizer.
- Type-aware metrics dict: `loss`, `z_loss`, `total_loss` are reported
  using the configured denominator; `supervised_tokens` is reported as the
  actual global count (not a per-device mean); `loss_normalizer` is logged;
  `grad_norm` is computed once on the globally-reduced gradients. This
  replaces the previous blanket `tree_map(pmean)` over the whole metrics dict.
- The AR step functions (`train_step`, `train_step_data_parallel`,
  `train_step_accumulated`, `train_step_accumulated_data_parallel`) are
  unchanged because every shard always has the same valid-token count.
  If AR ever gets padding/masking, port the same machinery.

Eval batch-size fix (likely 2-GPU OOM trap):

- `mean_eval_loss` was being called with `batch_size=args.batch_size`
  even when `--data-parallel` was on. Eval runs through a single-device
  JIT, so a global batch sized to fit only after `num_devices` sharding
  would OOM in the eval path even though training fits.
- Fix: `eval_batch_size = per_device_batch_size if args.data_parallel
  else args.batch_size`. A future improvement is a real
  `eval_step_data_parallel`, but this lower-risk change closes the
  2-GPU eval-OOM trap.

Max peak HBM across devices:

- Replaced `jax.devices()[0].memory_stats()` at end of run with a max
  across `jax.local_devices()[:num_devices]`. Per-device peaks are also
  captured in `performance_summary["jax_per_device_peaks"]` so any
  imbalance shows up in logged results instead of being hidden behind a
  single-device summary.

Data-parallel parity tests:

- Added `jax/tests/test_data_parallel_parity.py`, designed to run on simulated
  CPU devices via `XLA_FLAGS=--xla_force_host_platform_device_count=2`. Current
  checks:
  1. `test_supervised_loss_sums_basic_invariants`: end-to-end sanity
     for the new sum/count helper.
  2. `test_ar_single_vs_dp_parity`: AR DP on `(2, 4, 16)` sharded batch
     vs single-device on the equivalent `(8, 16)` global batch produces
     the same loss, grad norm, and post-update params; replicated model and
     optimizer leaves are checked for bit-identical device copies when an
     explicit device axis is present.
  3. `test_ar_accumulated_single_vs_dp_parity`: same for AR with a leading
     accumulation axis.
  4. `test_mdlm_uneven_mask_dp_parity` (load-bearing): MDLM with shard
     0 supervising ~25% of tokens and shard 1 supervising ~75%
     reproduces the single-device sum-form objective under a fixed expected
     mask denominator. It also asserts `supervised_tokens` equals the global
     supervised-token count, not a per-device average.
  5. `test_mdlm_accumulated_uneven_mask_dp_parity`: same for the combined
     device-by-accumulation path.
- Loss/grads parity is checked at `atol=1e-5`. Post-update params are
  checked at `atol=1e-3` because Muon's Newton-Schulz iteration
  amplifies the floating-point summation-order differences between
  single-device sums and `psum`/`pmean` aggregations.

Validation:

- `python -m py_compile jax/training/loss.py jax/training/step.py
  jax/train_ar.py jax/tests/test_data_parallel_parity.py` passed.
- `XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu
  python jax/tests/test_data_parallel_parity.py` passed (all 5 tests).
- `python jax/tests/test_diffusion_stack.py` still passes after the
  supervised-loss refactor.
- `python jax/tests/test_training_stack.py` still passes.
- Tiny synthetic MDLM trainer smoke passed on single-device CPU and simulated
  2-device CPU pmap after switching the denominator.
- `python jax/tests/test_parity.py` still passes (PyTorch parity, AR
  forward path unchanged).
- `python jax/tests/test_parity_extras.py` still passes.

Open follow-ups:

- DP eval (`eval_step_data_parallel` and supervised counterpart) so the
  full global batch can be evaluated across devices. Current fix only
  prevents OOM; it does not restore single-device-batch-size eval
  throughput on multi-GPU.
- Confirm post-pmap model state shape on a real 2-GPU box before the
  first DP run, since the CPU smokes do not exercise eval immediately
  after a DP step.

## Phase 0 AR W&B Round-Trip and Old-Bundle LR Probes

Runtime environment:

- Escalated JAX sees two GPUs: `cuda:0` and `cuda:1`, both A100-SXM4-80GB.
- `data/climbmix_smoke_8192/` was rebuilt with
  `python jax/data/prepare_climbmix.py --output-dir data/climbmix_smoke_8192 --num-shards 1 --vocab-size 8192`.
  The manifest reports `64,373,247` train tokens and `64,423,581` val tokens.
- Tokenizer and loader sanity checks passed for train and val at
  `context_length=512`; token IDs were in `0..8191` and the AR shift relation
  held.
- AR configs and `jax/configs/optimizer/normuon_adamw.yaml` were aligned back
  to the PLAN.md LR family: table Adam `0.01`, scalar Adam `0.005`, Muon
  `0.04`.

AR baseline Phase 0:

- 300-step baseline run:
  `runs/ar_matrix/ar_baseline_phase0_300/train.jsonl`.
- W&B run id: `7b0ky41i`.
- Final checkpoint uploaded as W&B artifact
  `ar_baseline-checkpoint-final:final`.
- Step 299 metrics: train loss `3.5198`, grad norm `0.2158`,
  avg measured step `0.4290s`, `611k` tokens/s, estimated `88.7%` MFU,
  JAX peak HBM `36.53 GB`.
- W&B restore run downloaded `ar_baseline-checkpoint-final:final`, restored
  step `299`, and resumed at step `300`:
  `runs/ar_matrix/ar_baseline_phase0_restore/train.jsonl`.
- Resume seam was clean: step 299 train loss `3.5198`; step 300 train loss
  `3.5377`, eval loss `3.6350`, no NaN/spike. Step 350 eval loss was `3.5654`.

AR baseline LR re-check:

- Re-swept the baseline after the old-bundle LR probe found a much higher
  preferred multiplier.
- `lr_mult=2.0`: completed 300 steps at
  `runs/ar_matrix/ar_baseline_probe_lr2p0/train.jsonl`. Best logged eval loss
  was `3.7949` at step 225, then the curve worsened to `3.8133` by step 275.
  Avg measured step `0.4325s`, `606.2k` tokens/s, estimated `88.0%` MFU, JAX
  peak HBM `36.52 GB`.
- `lr_mult=0.5`: intentionally interrupted after it was clearly
  noncompetitive. Last eval point before interrupt was step 200 with eval loss
  `3.8154`.
- Conclusion: baseline default `lr_mult=1.0` remains the best baseline LR from
  the checked bracket. Do not transfer the old-bundle `5.0` multiplier back to
  the no-intervention baseline.

AR old-bundle LR probes:

- All probes used the PLAN shape: `--data-parallel --num-devices 2
  --batch-size 512 --grad-accum-steps 1`, seq512, vocab8192, bf16, cuDNN
  attention, full logits.
- Live `nvidia-smi` during `lr_mult=0.5` caught active use of both GPUs:
  about `62 GB` on each A100, GPU utilization `100%` / `81%`, and roughly
  `367-383 W`.
- `lr_mult=1.0`: completed 300 steps at
  `runs/ar_matrix/ar_old_bundle_probe_lr1p0/train.jsonl`. Best logged eval
  loss was `3.5843` at step 275. Avg measured step `0.4743s`, `552.7k`
  tokens/s, estimated `80.3%` MFU, JAX peak HBM `46.22 GB`.
- `lr_mult=2.0`: completed 300 steps at
  `runs/ar_matrix/ar_old_bundle_probe_lr2p0/train.jsonl`. Best logged eval
  loss was `3.5644` at step 275. Avg measured step `0.4764s`, `550.3k`
  tokens/s, estimated `80.0%` MFU, JAX peak HBM `46.22 GB`.
- `lr_mult=5.0`: completed 300 steps at
  `runs/ar_matrix/ar_old_bundle_probe_lr5p0/train.jsonl`. Best logged eval
  loss was `3.5253` at step 275. Avg measured step `0.4780s`, `548.4k`
  tokens/s, estimated `79.7%` MFU, JAX peak HBM `46.22 GB`. Live
  `nvidia-smi` samples during the run showed about `62 GB` allocated on each
  A100 and active utilization up to `100%` / `100%`.
- `lr_mult=10.0`: intentionally interrupted after step 200 because it was
  stable but clearly behind `5.0`. Last eval point before interrupt was step
  200 with eval loss `3.6398` versus `3.5960` for `5.0` at the matched step.
- `lr_mult=0.5`: intentionally interrupted after it was clearly
  noncompetitive. Last eval point before interrupt was step 225 with eval loss
  `3.7053`; matched-step losses trailed both `1.0` and `2.0`.

Current AR-old-bundle probe conclusion:

- `lr_mult=5.0` is the best of the tried short probes at 300-step scale:
  `3.5253` eval versus `3.5644` for `2.0` and `3.5843` for default `1.0`.
  It was stable over 300 steps, with low final grad norm (`0.052`) and no
  NaN/spike.
- `lr_mult=10.0` does not look useful; treat `5.0` as the selected
  old-bundle multiplier for the next run.
- `lr_mult=0.5` should not be promoted.
- Later W&B runs on another machine superseded the short-probe baseline
  conclusion. The default-`lr_mult=1.0` baseline run destabilized late: best
  eval was `3.2753` at step `1200`, then eval worsened to `3.4982` by step
  `2650` before crashing around step `2692`. The tuned baseline
  `ar_baseline_lr0p8` finished 5100 steps with best eval `2.8529` at step
  `4450` and final eval `2.8841` at step `5050`.
- A completed external W&B `ar_old_bundle` 5100-step run with `lr_mult=5.0`
  reached final eval `2.8000` at step `5050`, with late evals mostly in the
  `2.77-2.82` band. This remains the best AR configuration observed so far.

## AR Value Embedding Without Value Residual

The stable value-embedding variant removed value residual and removed the
trainable VE gain:

- Config base: `jax/configs/experiments/ar_value_embedding.yaml`.
- Runtime overrides: `--no-attn-val-residual --no-value-embedding-gain
  --lr-mult 2.0`.
- Data/tokenizer: `data/climbmix_24x_newtok_8192`, with 24 train shards and
  about `1.542B` train tokens. A 5100-step run at batch 512, seq 512 consumes
  about `1.337B` clean tokens, or about `0.867` train epochs.
- Parameter count for the no-VR/no-gain VE model was `88,168,712`, matching
  expectation: old bundle `63,001,376` minus 24 value-residual scalars plus
  four VE tables (`25,165,824`) plus four VE gates (`1,536`).
- A from-scratch 5100-step launch with this recipe destabilized almost
  immediately, so the final long run was resumed from the stable 500-step
  checkpoint at
  `runs/ar_matrix/newtok_smoke_500/ar_value_embedding_no_vr_nogain_lr2p0/checkpoints/final`.
- Completed run:
  `runs/ar_matrix/ar_value_embedding_no_vr_nogain_lr2p0_5k1_from500/train.jsonl`.
- W&B run id: `dsezmr2u`.
- Restored at step `499` and resumed from step `500`; the final checkpoint was
  saved at
  `runs/ar_matrix/ar_value_embedding_no_vr_nogain_lr2p0_5k1_from500/checkpoints/final`.
- Best checkpoint:
  `runs/ar_matrix/ar_value_embedding_no_vr_nogain_lr2p0_5k1_from500/checkpoints/best`.
- Best eval loss was `2.8480` at step `5000`. The last logged eval was
  `2.8895` at step `5050`; final train loss at step `5099` was `2.8495` with
  grad norm `0.5751`.
- Through the long run, eval stayed stable after step 500 and late grad norms
  stayed around `0.55-0.63`; no late runaway was observed.
- Performance summary: average measured step `0.6258s`, `418.9k` tokens/s,
  estimated `253.2` TFLOP/s, `81.2%` MFU, JAX peak HBM `42.26 GB`.

AR conclusion after the corrected baseline comparison:

- `ar_old_bundle` remains best (`~2.8000` final eval).
- `ar_value_embedding_no_vr_nogain_lr2p0` is competitive with the tuned
  baseline and slightly better by best eval (`2.8480` versus baseline
  `2.8529`), but its final eval (`2.8895`) is similar to the tuned baseline
  final eval (`2.8841`).
- The default baseline LR conclusion was wrong for full-length training:
  `lr_mult=0.8` is the stable baseline setting, while `lr_mult=1.0`
  destabilizes late.
