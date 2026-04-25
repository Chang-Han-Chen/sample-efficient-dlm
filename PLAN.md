# PLAN: Ablation Matrix for an Efficient Language Diffusion Model Backbone

This plan picks up after the JAX bring-up is complete. The training stack,
optimizer, data pipeline, parity checks against PyTorch, and seq-512 profiling
are all in place (see `PROGRESS.md`). The next phase is the controlled
ablation matrix that answers the project's core scientific question.

This document is written for a future AI agent. Keep working independently
through the experiment matrix below until each phase has a documented winner
or a clear reason to stop.

## Goal

Find an efficient transformer backbone plus optimizer recipe for language
diffusion models (LDMs).

The scientific question:

> Do architecture and optimization interventions that help autoregressive
> language modeling also help diffusion language modeling?

The strategy is greedy and transitive: first find the AR winner, then find
the MDLM winner. If both objectives converge on the same intervention bundle,
we trust that the bundle also generalizes to BD3LM and only run BD3LM once at
the end for confirmation. If AR and MDLM diverge, decide on the fly whether
to run a targeted BD3LM ablation. This avoids the much-higher BD3LM wall-time
cost (~2× slower per optimizer step at this shape) while still producing a
recommendation we can defend.

## Repository Map

- `jax/` — active code. Backbone, optimizer, data, training, diffusion,
  configs, tests. This is where all experiments live now.
- `jax/configs/experiments/` — canonical experiment configs. New runs MUST be
  added here; do not rely on CLI flags as the source of truth.
- `pytorch/` — reference autoregressive implementation. Keep only for parity
  reference and historical comparison.
- `baby-dLM/` — PyTorch MDLM/BD3LM reference. Consult for diffusion semantics
  when in doubt; the upstream BD3LMs repo
  (https://github.com/kuleshov-group/bd3lms.git) is the higher-fidelity
  source.
- `karpathy/` — reference data prep and training notes.

## Fixed Choices (do not re-litigate)

These are locked in for Part 1. Treat them as constants for every run; if
profiling or a sanity check forces a revisit, document the reason in
`PROGRESS.md` before launching the affected sweep.

- **Architecture shape**: `gpt_small_faster` (~70M params).
  `d_model=768`, `d_ff=2048`, `n_layers=8`, `n_heads=12`, `n_kv_heads=None`.
- **Tokenizer / data**: ClimbMix BPE, base vocab `8192`, diffusion vocab
  `8193`, mask token id `8192`. Data root: `data/climbmix_smoke_8192/`.
- **Sequence length**: `512` clean tokens. (BD3 internally doubles to 1024
  via `x_t || x_0`.)
- **Effective batch**: 512 sequences per optimizer step (262,144 tokens).
- **Precision and attention**: bf16 with cuDNN attention
  (`--attention-impl cudnn`).
- **Weight tying**: ON (LM head shares the embedding `nnx.Param`).
- **Optimizer**: NorMuon+AdamW with three LR families:
  - table Adam (token embeddings + tied LM head + value-embedding tables)
  - scalar Adam (RMSNorm gains, QK gains, value-residual scalars)
  - Muon (matrix weights, with PyTorch-style shape LR multiplier)
  Adam betas `(0.95, 0.99)`, Adam eps `1e-8`, Muon cautious WD `1e-4`,
  momentum warmup `0.85 -> 0.95`.
- **Default LR center**: `table=0.01`, `scalar=0.005`, `muon=0.04`. Sweep via
  `--lr-mult` which scales all three together.
- **Schedule**: 100-step linear warmup + 5000-step constant peak = 5100
  total optimizer steps.
- **Initialization**: default sample-efficient GPT init. No muP/non-muP
  ablation.
- **Loss path**: full logits for AR and MDLM; BD3 uses the noisy-stream
  hidden slice + full projection. Chunked CE exists but is NOT used in main
  runs (does not improve fixed-effective-batch wall time at vocab 8192).
- **BD3 attention**: dense-mask path with cuDNN. The pure-JAX blocked
  decomposition exists (`--bd3-attention blocked`) but is not faster on the
  A100 at this shape; leave it off.
- **Old intervention bundle (toggled together)**: QK-norm + value residual +
  per-head attention gating + layernorm/depth scaling. This is a single
  combined switch, NOT a factorial axis for the AR matrix.
- **Value embeddings**: Karpathy-style alternating-layer placement (last
  layer always included). Normalize the value-residual mixture first, then
  add token value embeddings on top. See `jax/transformer/transformer.py`
  for the exact formula. `gamma_ve_l=0` reproduces the no-value-embedding
  baseline.

## Hardware Plan

Default target: **2× A100** with

```text
--data-parallel --num-devices 2 --batch-size 512 --grad-accum-steps 1
```

That gives 256 examples/GPU and no accumulation. The single-A100 fallback is
`--batch-size 256 --grad-accum-steps 2`. Compare loss/eval metrics across
these two shapes freely; compare wall time only within the same shape.

If 4+ GPUs become available later: `--num-devices 4 --batch-size 512
--grad-accum-steps 1` (128 examples/GPU). Same effective batch.

## Compilation Cache

A persistent JAX compilation cache is enabled by default in `train_ar.py` at
`/tmp/sample_efficient_gpt_jax_cache`. It survives process restart, so any
later run with identical shape and static args reuses the cached XLA
executable and skips the ~30–40 s compile step.

What this means in practice for the matrix:

- **Within one experiment configuration** (same architecture flags, same
  microbatch, same seq len, same dtype, same attention impl): all LR-sweep
  runs after the first reuse the cache. Compile cost is paid once.
- **Between configurations whose static args differ** (e.g., toggling
  `qk_norm`, `value_residual`, `gating`, or switching
  `--objective ar/mdlm/bd3lm`): one recompile per new key. The cache then
  holds all keys, so future identical launches are free.
- **Microbatch changes** recompile because the batch dim is part of the
  static shape under `jit`. Keep microbatch fixed across the matrix
  (current configs already do).

Two pitfalls to watch:

1. `/tmp/` may be cleared on VM reboot. For long-running experiments,
   override the default to a persistent path before launching:

   ```text
   export JAX_COMPILATION_CACHE_DIR=/path/to/persistent/jax_cache
   ```

   `train_ar.py` honors the env var.
2. JAX/XLA version changes invalidate the cache. If you upgrade either,
   expect one recompile per configuration.

So yes — moving between experiments and phases with the same shape/static
args does NOT recompile. Moving between objectives or architecture toggles
recompiles once per key.

## Experiment Plan

### Phase 0: W&B Checkpoint Round-Trip Verification

Before relying on W&B checkpoints to bridge sweeps and resume failed runs,
verify the round-trip end-to-end on a real training loop. The unit and pmap
checkpoint tests already passed (see `PROGRESS.md`), but the
real-training-loop W&B resume path has not been load-bearing yet.

This check is folded into Phase A (baseline AR) and Phase B (baseline MDLM)
rather than running as a separate phase:

1. Train for 300 steps at default LR with W&B checkpoint upload enabled.
2. Save the final checkpoint to W&B.
3. Restart the trainer with `--restore-wandb-artifact <name>:final`.
4. Train another 50–100 steps and confirm:
   - loss continues to decay smoothly from where it left off (no jump,
     NaN, or spike at the seam),
   - tokens-per-second matches the original steady state,
   - SHA256s in `metadata.json` match what W&B has.

If any check fails, fix the W&B checkpoint path before launching the rest
of the matrix.

### Phase A: AR Ablation Matrix

Run three AR configurations:

1. **Baseline** — `jax/configs/experiments/ar_baseline.yaml`. Old
   interventions off, value embeddings off. Use the default LR center
   directly (`--lr-mult 1.0`). Prior 300-step probes already established
   that `table=0.01, scalar=0.005, muon=0.04` is the best bracket here;
   do not re-sweep the baseline. The first 300 steps of this run double as
   the Phase 0 W&B round-trip target for the AR path.
2. **Old bundle** — `jax/configs/experiments/ar_old_bundle.yaml`. QK-norm
   + value residual + per-head gating + layernorm/depth scaling.
3. **Old bundle + value embeddings** —
   `jax/configs/experiments/ar_value_embedding.yaml`.

Configs 2 and 3 may need a small LR check because the architecture change
can shift the useful LR. Probe `--lr-mult ∈ [0.5, 1.0, 2.0]` for 300 steps
each (100 warmup + 200 constant); widen only if the best result lands at
an edge.

LR sweep tactics:

- Kill probes early when the signal is obvious: grad norm spiking above
  ~30, loss above ~5.5 at step 100, or an obviously flat curve.
- Promote the top 1–2 LRs per configuration to the full 5100-step run.

Decision rule:

- Compare each configuration by its best-LR full-length run on eval loss,
  while logging all probe results.
- Old bundle is expected to beat baseline; if it does not, treat that as
  a bug or hyperparameter problem and investigate before moving to (3).
- Keep value embeddings only if config 3 beats config 2 on eval loss at
  matched best LR.

The AR winner is the input to Phases B–D as the default LR family.

### Phase B: MDLM Baseline Validation

Confirm MDLM trains cleanly at `seq_len=512`, effective batch 512 with the
AR-selected LR family. Run a 300-step probe at default LR:

1. The same probe doubles as the Phase 0 W&B round-trip test for the
   diffusion path: save to W&B at step 300, restore, train another 50–100
   steps, confirm smooth resumption.
2. If train loss decays monotonically and grad norm stays bounded,
   proceed to Phase C.
3. If unstable, run a small LR probe (3 multipliers, 100 steps) only for
   MDLM. Prior probes (`PROGRESS.md`) suggest the AR LR family transfers
   to MDLM; only deviate if forced.

A BD3LM check is NOT required at this point; BD3LM only runs in Phase D.
If you want to de-risk BD3LM earlier (recommended if you have spare time
during the long AR runs), launch one 100-step BD3LM probe with the dense
mask path at the AR LR family and just confirm it is stable. Do not
ablate.

### Phase C: MDLM Ablation Matrix (greedy sequential)

Run a greedy sequential ablation for MDLM only. The working hypothesis is
that interventions which help AR will also help MDLM; ablating MDLM
directly tests the AR→diffusion transfer cheaply.

Start from the MDLM baseline (all interventions off, value embeddings
off). Intervention order:

1. QK-norm
2. Value residual
3. Per-head attention gating
4. Layernorm / depth scaling
5. Value embeddings (combined placement)

For each step:

1. Add the intervention on top of the previously-kept set.
2. Train at the standard schedule (5100 steps, effective batch 512).
3. If eval loss improves vs. the previous best for MDLM, keep it.
4. If it does not improve, drop it and move to the next.

Use the AR best-LR family by default. If a run becomes unstable, run a
3-multiplier 100-step LR probe before deciding.

Greedy assumption: positive interventions stay positive with or without
each other. If results look order-sensitive (e.g., #4 helps only when #3
is dropped), pause and document in `PROGRESS.md` rather than expanding to
a factorial. As a sanity check, also run "all kept interventions on" vs.
"greedy bundle" once at the end of the matrix if there is doubt.

### Phase D: Cross-Objective Comparison and BD3LM Confirmation

Compare the AR winner (Phase A) and the MDLM winner (Phase C):

- **If the winning intervention sets agree**: run BD3LM once with the
  agreed bundle plus once with the BD3LM baseline (no interventions).
  Compare on eval loss. The expected outcome is that the bundle wins; if
  it does, the recipe is confirmed.
- **If they disagree**: decide on the fly between
  - a targeted BD3LM ablation on only the disputed interventions, or
  - running BD3LM at both candidate bundles and picking the better one.
  Document the decision and rationale in `PROGRESS.md` before launching.

Final comparison report:

- Train and eval loss curves for the AR best, MDLM best, and BD3LM
  confirmation runs.
- Eval bits-per-byte for AR; comparable diffusion eval metric for MDLM
  and BD3LM.
- Tokens/sec, wall time, peak HBM, MFU.
- Configuration diffs across the three winners.

Deliverable: a recommendation in `PROGRESS.md` of the form "for LDM
training under this regime, use {architecture set} plus NorMuon+AdamW
with {LR family}." Note explicitly whether AR and MDLM agreed, and what
BD3LM showed.

## Per-Run Reporting Requirements

Log every run to W&B (when configured) and to local JSONL. Required fields:

- Resolved config (saved as `resolved_config.json` next to checkpoints
  automatically when `--output-dir` is set).
- Train loss, eval loss, z-loss, global grad norm — as scalar histories.
- Tokens/sec, MFU (denominator: A100 SXM dense bf16 peak `312e12` FLOP/s).
- Step time, compile time (logged separately), peak HBM (max across
  devices, plus per-device peaks).
- Effective batch size, microbatch, grad accumulation count, LR multiplier.
- Run seed, data shard/window order.
- W&B tags/groups: one of `ar` / `mdlm` / `bd3lm`; one of `baseline` /
  `old_bundle` / `value_embedding` / sequential-ablation step name; plus
  `seq512`, hardware tag.

For diffusion runs, also log objective-specific metrics (mask rate stats,
supervised-token count, BD3 block length) and a comparable validation
metric across MDLM/BD3LM.

Checkpointing:

- Save `final` always; save `best` whenever eval loss improves.
- Use sparse periodic checkpoints (`--checkpoint-interval`) only when
  needed for a specific run.
- W&B artifact upload is on by default; disable per-run with
  `--no-wandb-checkpoints` if storage/bandwidth is tight.
- Restore: `--restore-checkpoint <dir>` for local;
  `--restore-wandb-artifact <name>:<alias>` for W&B.
- The Phase 0 round-trip test (folded into Phase A and B) is what
  certifies the W&B path before the matrix depends on it.

## Done Criteria for Part 1

Part 1 is done when:

1. Phase 0 W&B checkpoint round-trip is verified for both AR (during the
   baseline AR run) and MDLM (during the MDLM baseline run).
2. Phase A AR matrix has run, with a winning AR configuration selected
   from `{baseline, old_bundle, old_bundle + value_embedding}`.
3. Phase C MDLM ablation has run, with a documented winning intervention
   subset.
4. Phase D ran BD3LM at least once at the appropriate bundle (agreed or
   resolved-on-the-fly), and produced the recommended LDM
   backbone+optimizer recipe in `PROGRESS.md`.
5. All experiment configs are committed under
   `jax/configs/experiments/`, and resolved configs are saved with their
   checkpoints.
6. Any divergence between AR-winning and MDLM-winning configurations is
   documented with a hypothesis.

## Operational Notes

- `train_ar.py` consumes configs via `--config` and supports `--lr-mult` for
  sweeps. `resolved_config.json` is written automatically next to local
  checkpoints when `--output-dir` is set; it is also logged to W&B.
- W&B is the primary logger, but local JSONL logs MUST be sufficient to
  inspect runs if W&B is not configured.
- Checkpoints are msgpack files plus a `metadata.json` with step, metrics,
  RNG states, and SHA256 hashes.
- Multi-GPU data-parallel uses the `pmap` path. Eval currently runs single-
  device; the trainer auto-sets `eval_batch_size = per_device_batch_size`
  when `--data-parallel` is on. `eval_step_data_parallel` is an open
  follow-up; only block on it if eval becomes the wall-time bottleneck.
- For long runs across VM reboots, set `JAX_COMPILATION_CACHE_DIR` to a
  persistent path so compile cost is paid only once per static-arg key
  for the entire matrix.
- When in doubt about diffusion semantics, consult `baby-dLM/` first, then
  the upstream BD3LMs repo. Do not re-derive from scratch.
- Git is handled by the user. Do not automate commits or pushes.
- Persistent volume setup is not required during ablation runs; local
  `data/` and checkpoint paths on the VM are acceptable. Move artifacts to
  W&B for durability.
