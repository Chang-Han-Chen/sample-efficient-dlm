# Scaling Block Diffusion on ClimbMix

## Overview

We study how block-length curriculum training affects scaling laws for discrete
diffusion LMs.  All experiments use ClimbMix-400B (BPE, vocab 8192, seq len 2048)
and three model families: AR, MDLM, BD3-LM.

In parallel, we compare the current training setup against one new optimizer
option:
- `adamw`: the existing code path, unchanged. One global learning rate per run.
- `normuon`: a Karpathy-inspired grouped optimizer path with AdamW on
  non-matrix parameter groups and NorMuon on matrix parameter groups.
  This path uses Karpathy's reference learning rates and width scaling formulas
  as the center of its learning-rate sweep.

The initial optimizer comparison covers the no-curriculum baselines and
curricula 0 and 1 only.  Later curricula can be revisited after we understand
whether the grouped optimizer recipe materially changes the ranking.

---

## 1  Model sizes

We scale up from the TinyShakespeare regime to three sizes:

| Label | n_embd | n_layer | n_head | Approx non-embedding params |
|-------|--------|---------|--------|-----------------------------|
| 50M   | 768    | 7       | 12     | 49.5M                       |
| 98M   | 896    | 10      | 14     | 96.3M                       |
| 170M  | 1024   | 14      | 16     | 176.2M                      |

We count only non-embedding transformer parameters:
`N = 12 * n_layer * n_embd^2`.
This includes attention and MLP weights and excludes the token embedding and
LM head.

Head dim is always 64 (= n_embd / n_head).

These sizes are still small enough to support short LR sweeps and broad
IsoFLOP experiments on a single H100.

---

## 2  Data

ClimbMix via `prepare.py`.  Download ≥10 shards for training + 1 pinned
validation shard (shard_06542).  BPE tokenizer with vocab_size = 8192.

Token count per run is determined by the chosen FLOP budget and model family,
not by a fixed 1B-token target.  Once tokens_processed is known, step count is
`tokens_processed / (batch_size * seq_len)`.

---

## 3  Curricula

A "curriculum" is a sequence of stages, each defined by its model family and,
for BD3-LM stages, its block_len.
Within each stage the model trains for a fixed fraction of the total FLOP budget.
Stage transitions warm-start from the previous stage's checkpoint (weights only;
optimizer and LR schedule reset).

### Curriculum 0 — plain
```text
stage: AR -> BD3(block_len=16)
FLOP split: p_AR on AR, 1 - p_AR on diffusion
```
Sweep p_AR ∈ {0.2, 0.3, 0.5, 0.8}.

### Curriculum 1 — Geometric doubling (SDAR / NBDiff style)
```text
stage: AR -> BD3(2) -> BD3(4) -> BD3(8) -> BD3(16)
FLOP split: 20% each (5 equal stages)
```
Starts with true autoregressive training and doubles the BD3 block length at
each stage.

### Curriculum 2 — Aggressive early jump (LLaDA 2 style)
```text
stage: AR -> BD3(8) -> BD3(64) -> BD3(512) -> BD3(16)
FLOP split: 20% each (5 equal stages)
```
Jumps quickly to large blocks, then narrows back to block_len=16 for the final
stage.

### Curriculum 3 — AR-heavy warmup, then two diffusion stages
```text
stage: AR -> BD3(4) -> BD3(16)
FLOP split: p_AR on stage 1, 0.8 - p_AR on stage 2, 0.2 on stage 3
```
Sweep p_AR ∈ {0.2, 0.3, 0.5}.

### Curriculum 4 — To be designed
Informed by results of curricula 1–3.  Placeholder for now.

### Baselines (no curriculum)
- **Pure AR**: AR model for the full budget.
- **Pure MDLM**: standard MDLM (no blocks) for the full budget.
- **Pure BD3-LM**: block_len=16 (or other fixed value) for the full budget.

---

## 4  Learning-rate calibration

This is the first experiment.

For each `(optimizer_family, model_family, size)` pair:
- Train for 200 steps with a warmup-stable schedule (5% linear warmup, then
  constant LR).
- Keep the selection rule the same for both optimizers:
  - track train loss and grad norm throughout the 200-step run
  - discard unstable runs (loss blow-up / grad norm blow-up / NaNs)
  - choose the stable run with the best train loss
- `adamw` calibration:
  - use the existing single-LR sweep exactly as before
  - one candidate `learning_rate` per run
- `normuon` calibration:
  - use an adapted Karpathy-style parameter grouping:
    - embeddings
    - `lm_head`
    - matrix parameters (all 2D weights except embeddings / `lm_head`)
    - scalar/vector residual parameters (any remaining params with ndim < 2;
      this group may be empty for the current models)
  - use Karpathy-style reference LRs and width scaling as the center of the
    sweep:
    - `embedding_lr = 0.6 * sqrt(768 / d_model)`
    - `unembedding_lr = 0.004 * sqrt(768 / d_model)`
    - `scalar_lr = 0.5` for the scalar/vector group
    - `matrix_lr = 0.04` with no additional width scaling
  - sweep a small 2D multiplier grid around those references:
    - `adam_mult` rescales `embedding_lr`, `unembedding_lr`, and `scalar_lr`
    - `matrix_mult` rescales `matrix_lr`
  - log optimizer family, `adam_mult`, `matrix_mult`, and the realized
    per-group learning rates in addition to loss and grad norm

For BD3-LM, use the fixed-block baseline setting (`block_len=16`) as the
representative calibration run.  After calibration, reuse the chosen LR config
for that `(optimizer_family, model_family, size)` pair across the baselines and
all curricula that use that optimizer family.

The key point is that `adamw` remains untouched.  The only new optimizer path
is `normuon`.

### 4.1  Calibration results (50M)

**Scope decision:** Phase 1 runs at 50M only.  LR sweeps were run for AR at
50M and the selected configs are reused for MDLM and BD3-LM at the same scale.
98M and 170M sweeps are deferred until after Phase 1 curriculum results.

**AdamW** — 1D sweep over `[1e-4, 3e-4, 1e-3, 3e-3, 1e-2]` on AR/50M.
All 5 runs stable.  Selected LR = **1.0e-3** (loss=4.507).

| LR     | final loss | stable |
|--------|-----------|--------|
| 1e-4   | 5.522     | yes    |
| 3e-4   | 5.041     | yes    |
| **1e-3** | **4.507** | yes  |
| 3e-3   | 4.665     | yes    |
| 1e-2   | 6.205     | yes    |

Applied to all three families at 50M → `calibrated_lrs.json`.

**NorMuon** — 2D sweep over `adam_mult × matrix_mult ∈ {0.3, 1.0, 3.0}²`
on AR/50M.  All 9 runs stable.  Selected config = **adam_mult=0.3,
matrix_mult=1.0** (loss=5.109).

| adam_mult | matrix_mult | final loss |
|-----------|-------------|-----------|
| **0.3**   | **1.0**     | **5.109** |
| 0.3       | 0.3         | 5.131     |
| 0.3       | 3.0         | 5.143     |
| 1.0       | 1.0         | 5.136     |
| 1.0       | 0.3         | 5.155     |
| 1.0       | 3.0         | 5.166     |
| 3.0       | 1.0         | 5.343     |
| 3.0       | 0.3         | 5.387     |
| 3.0       | 3.0         | 5.364     |

Pattern: lower `adam_mult` is consistently better; `matrix_mult=1.0` is the
sweet spot.  Applied to all three families at 50M → `calibrated_normuon.json`.

---

## 5  Curriculum experiments (Phase 1)

Phase 1 runs curriculum 0, curriculum 1, and a pure BD3-LM baseline at 50M
scale with both optimizer families, using a fixed step budget rather than a
FLOP budget.

**Training scope:**
- Model size: 50M only
- Optimizer families: `adamw`, `normuon`
- Runs: pure BD3-LM baseline (`block_len=16`), C0 (sweep p_AR ∈ {0.2, 0.3, 0.5, 0.8}), and C1 (geometric)
- Total steps per run: 1000, split across stages by `flop_frac`
- LR schedule: warmup-stable (5% linear warmup, then constant)
- AR warmup uses `batch_size=128, grad_accum_steps=2` (effective batch 256)
- BD3-LM stages use `batch_size=32, grad_accum_steps=8` due to memory limits
  while preserving the same effective batch 256
- Current Phase 1 execution disables in-run eval/sampling
  (`eval_interval=0`, `skip_final_eval=true`, `num_final_samples=0`) and logs
  train loss / grad norm every 10 steps

This gives **12 runs** total (6 run types × 2 optimizers).

### 5.1  Step allocation examples

C0 with p_AR=0.2: 200 AR steps → 800 BD3(16) steps.
C0 with p_AR=0.8: 800 AR steps → 200 BD3(16) steps.
C1 (5 equal stages): 200 steps each for AR → BD3(2) → BD3(4) → BD3(8) → BD3(16).
Pure BD3 baseline: 1000 BD3(16) steps from scratch.

### 5.2  Execution order
1. ~~Implement `--optimizer normuon`~~ Done.
2. ~~Run LR calibration for 50M.~~ Done (§4.1).
3. ~~Freeze calibrated configs.~~ Done (`calibrated_lrs.json`,
   `calibrated_normuon.json`).
4. Run pure BD3-LM baseline for both optimizers at 50M/1000 steps
   (`run_phase1_baselines.sh`).
5. Run curriculum 0 and curriculum 1 for both optimizers at 50M/1000 steps
   (`run_phase1.sh`).
6. Analyze results: compare optimizer families, compare against the pure BD3
   baseline, compare p_AR schedules, compare C0 vs C1.
7. Decide whether to proceed to IsoFLOP scaling (§5A) or additional curricula.

### 5.3  Metrics
- Primary: validation BPB (bits per byte) from post-training evaluation of the
  final checkpoint.
- Secondary: train-loss traces logged every 10 steps during training.
- Phase 1 training runs do not perform in-run val / GPT2 / sample eval.
- Log optimizer family and calibrated LR settings in every result.

---

## 5A  Scaling laws (IsoFLOP) — deferred

Full IsoFLOP sweeps are deferred until after Phase 1 curriculum results clarify
which curricula and optimizer families are worth scaling up.

Compute budgets: `C ∈ {1e18, 2e18, 4e18, 1e19}` FLOPs.

For each (C, curriculum) pair, sweep model sizes (50M, 98M, 170M) and record
the best validation loss.  This gives one IsoFLOP curve per curriculum.

FLOP accounting (unchanged from `experiment_config.py`):
- Count only non-embedding parameters: `N = 12 * n_layer * n_embd^2`
- AR / MDLM: C = 6 · N · tokens_processed
- BD3-LM (dual-stream training): C = 12 · N · tokens_processed

Equivalent token budgets are therefore determined by the compute budget, model
family, and model size.

---

## 6  Implementation plan

### 6.1  Update train.py for ClimbMix
- Replace the character-level `data.txt` pipeline with `prepare.py`'s
  `make_dataloader` / `Tokenizer`.
- Set `block_size = 2048` (MAX_SEQ_LEN), `vocab_size = 8192`.
- Mask token = a reserved special token (not a regular vocab entry).
- Add `--warmup_stable` flag: when set, hold LR constant after warmup
  (no cosine decay).
- Log grad norm alongside train loss so LR sweeps can identify the best stable
  LR.
- Add optimizer-family support:
  - `adamw` (current path, unchanged)
  - `normuon` (new grouped optimizer path)
- For `normuon`, implement the Karpathy-style parameter grouping and
  width-scaled AdamW subgroup LRs.
- Log the realized per-group learning rates for `normuon`, and keep the current
  scalar-LR logging for `adamw`.
- Keep checkpointing / resume behavior streamlined:
  - `adamw` checkpoints save one optimizer state
  - `normuon` checkpoints save the grouped optimizer state without changing the
    warm-start semantics already used by curricula

### 6.2  Update experiment_config.py
- Add 50M, 98M, 170M to MODEL_SIZES.
- Count FLOPs with non-embedding parameters only.
- Add IsoFLOP budgets: {1e18, 2e18, 4e18, 1e19}.
- Add curriculum definitions as structured config (list of
  `(model_family, block_len, flop_fraction)` stages).
- Add optimizer families and a place to store the selected LR config for each
  `(optimizer_family, model_family, size)` triple.
- For `adamw`, store the selected scalar LR exactly as today.
- For `normuon`, store the selected `adam_mult`, `matrix_mult`, and optionally
  the realized per-group learning rates for logging / reproducibility.

### 6.3  Add LR sweep script
New script `run_lr_sweep.py`:
1. For each `(optimizer_family, model_family, size)`, run 200-step sweeps.
2. For `adamw`, run the existing 1D LR sweep.
3. For `normuon`, run a small 2D grid over `(adam_mult, matrix_mult)` around
   the Karpathy reference setting.
4. In both cases, record train loss and grad norm curves and pick the best
   stable run using the same rule.
5. Write a summary table + pickle/JSON with the selected optimizer-specific LR
   config.

### 6.4  Add curriculum runner
New script `run_curriculum.py`:
1. Takes a curriculum spec + (total FLOP budget OR total steps) + model size.
2. For each stage: compute steps from FLOP fraction (budget mode) or directly
   from `round(total_steps * flop_frac)` (steps mode).
3. Launch `train.py` with the stage's `--model`, `--block_len` when needed,
   the calibrated optimizer config for that `(optimizer_family, model_family,
   size)` pair, and `--resume_from` pointing to the previous stage's checkpoint.
4. For `adamw`, pass the selected scalar LR.
5. For `normuon`, pass the selected multiplier config and let `train.py`
   realize the per-group learning rates from model width.
6. Log per-stage and final validation loss, along with optimizer family and LR
   settings.

### 6.5  Add IsoFLOP sweep script
New script `run_isoflop.py`:
1. For each (budget, curriculum, optimizer_family, model_size): run the
   curriculum.
2. In the first pass, restrict curricula to the baselines plus curriculum 0 and
   curriculum 1.
3. Collect `(model_size -> val_loss)` curves, keyed by optimizer family.
4. Output a table + pickle for plotting.

### 6.6  Evaluation
- Primary metric: validation BPB (bits per byte) via `prepare.py`'s
  `evaluate_bpb`.
- Secondary: generation quality samples for qualitative inspection.
- Keep a per-run results table with:
  - optimizer family
  - curriculum id
  - model size
  - FLOP budget
  - selected LR setting
  - for `normuon`, selected `adam_mult`, selected `matrix_mult`, and realized
    per-group learning rates
  - final train / val metrics

### 6.7  Plotting
Extend `reproduce.ipynb` (or new notebook) to plot:
- LR sweep train-loss and grad-norm diagnostics for both optimizers.
- For `normuon`, LR sweep heatmaps over `(adam_mult, matrix_mult)`.
- IsoFLOP curves (val loss vs model size, one curve per curriculum and
  optimizer family).
- Curriculum learning curves (val loss vs FLOPs, showing stage transitions).
- Optimal model size vs compute for each curriculum.

---

## 7  Open questions

- Should we also sweep block_len within the "no curriculum" BD3-LM baseline
  (e.g., block_len ∈ {4, 8, 16, 32})?
- Is a single BD3-LM LR per size good enough across block_len ∈
  {2, 4, 8, 16, 64, 512}, or should we add block_len-specific LR calibration
  only if instability appears?
- Does the Karpathy width-scaling rule transfer cleanly from AR-heavy training
  to MDLM / BD3-LM, or will diffusion models want a different `adam_mult`
  center?
- If `normuon` helps, is the gain large enough to justify carrying the more
  complicated grouped-LR sweep into the later curricula?
- Curriculum 4 design: consider reverse schedules (large→small block_len),
  or non-monotonic schedules informed by curricula 1–3.

---

## 8  Fast Iteration Infra (brainstorm)

As we collect more systems and recipe ideas (e.g. FlexAttention for BD3-LM,
EMA, Z-loss, value residual paths, norm scaling, packed QKV, FlashAttention
variants), the repo should support *fast screening* before ideas graduate into
the main training stack.

Recommended direction:

- Keep the main training code as the production path.
- Add a lightweight experiment area such as `prototypes/` or `research/` for
  ideas that are not yet stable enough to merge into the main path.
- Use notebooks primarily for analysis / plots / debugging, not as the source
  of truth for runnable experiments.

### 8.1  Promotion ladder for new ideas

Each idea should move through three stages:

1. **Microbench**
   - Goal: measure `tokens/s`, step time, peak memory, compile overhead.
   - Best for: attention kernels, mask representations, packed QKV, fused ops.
   - Use synthetic data with realistic tensor shapes.

2. **Proxy train**
   - Goal: measure whether the idea helps optimization on a small but real
     training run.
   - Best for: EMA, Z-loss, optimizer tweaks, value residual, norm scaling.
   - Use a small model (roughly 10M–20M) and short runs before scaling up.

3. **Full-stack smoke**
   - Goal: check that the idea still helps in the real regime.
   - Use a very small number of realistic 50M / seq=2048 runs.
   - Only winners from the proxy stage should reach this step.

### 8.2  Recommended folder split

- `prototypes/bench_attention.py`
  - Synthetic throughput / memory benchmark for AR, MDLM, BD3-LM attention.
- `prototypes/run_proxy.py`
  - Short proxy training harness for architecture / optimizer / recipe ideas.
- `prototypes/configs/`
  - Tiny configs for quick experiments.
- `notebooks/`
  - Analysis and plotting only.
- `results/ideas.jsonl` (or similar)
  - Simple log of idea name, config, metrics, and notes.

### 8.3  Separate systems vs recipe vs model ideas

Not all ideas need the same harness. Keep these buckets distinct:

- **Systems ideas**
  - FlexAttention, FlashAttention paths, fused ops, mask construction,
    compile settings.
- **Recipe ideas**
  - EMA, Z-loss, schedules, weight decay rules.
- **Model ideas**
  - Value residual, layer-wise norm scaling, GQA, other architecture tweaks.

This avoids mixing kernel speed questions with modeling questions too early.

### 8.4  Practical proxy-design notes

- A tiny notebook model with very short sequence length is good for logic
  debugging, but not always reliable for systems conclusions.
- Attention kernels can behave very differently at seq=256 vs seq=2048, or
  for masked vs unmasked attention.
- Therefore:
  - use toy shapes for correctness,
  - use realistic shapes for throughput benchmarking,
  - and only use 50M-scale runs for finalists.

### 8.5  Target iteration loop

Aim for an idea workflow like:

1. 5-minute correctness smoke test
2. 15-minute synthetic throughput benchmark
3. 1–2 hour proxy train
4. 50M confirmation only for the best ideas

This should make it much cheaper to decide whether a speed or quality idea is
worth integrating into the production training path.

---

## 9  Data-constrained scaling (Phase 2)

Motivation: diffusion LMs are data-efficient but token-inefficient
(Ni et al., 2025).  With the same unique data, a DLM eventually overtakes AR
on downstream tasks — but it takes many more gradient steps to get there.
This experiment directly measures the crossover under controlled conditions.

### 9.1  Scope

Token-matched data-constrained PoC at 50M only.

- Model: 50M (`n_embd=768, n_layer=7, n_head=12`, ~49.5M non-embedding params).
- Data: ClimbMix, BPE vocab 8192, seq_len 2048.
- Methods: pure AR, pure BD3-LM (`block_len=16`), one selected curriculum.
- Unique-token budgets: `U ∈ {25M, 50M, 100M}`.
- Total training tokens: `D_target = 500M` (≈954 steps at effective batch 256).
- Implied repeat counts: U=25M → 20 epochs, U=50M → 10, U=100M → 5.
- Optimizer: AdamW only in the first PoC; NorMuon support ready but deferred.

### 9.2  Budget conversion

- `tokens_per_step = 256 × 2048 = 524,288`.
- `total_steps = round(D_target / tokens_per_step) = 954`.
- Log both `D_target` and `actual_total_tokens = 954 × 524,288 = 500,170,752`.

### 9.3  Dataset construction

Deterministic nested unique-data subsets from a single canonical train-token
stream (fixed shard list, fixed shard order, fixed seed):

- `dc_u25m` = first 25M tokens of the canonical stream.
- `dc_u50m` = first 50M tokens (⊇ dc_u25m).
- `dc_u100m` = first 100M tokens (⊇ dc_u50m).

Training cycles over the U-token prefix until `total_steps` is reached.
All methods under the same U use exactly the same subset and token order.
Save one manifest per U (shard list, seed, token range, actual unique count).

### 9.4  Data cursor semantics

Curriculum stage transitions must **not** restart the data stream at token 0.
The repeated-data stream behaves like one continuous 500M-token run across all
stages.  Track a global token offset per run; stage k+1 resumes where stage k
ended (modulo U).  If `train.py` does not support this, add
`--data_manifest`, `--data_offset_tokens`, `--cycle_data true`.

### 9.5  Micro-batch settings

Same memory-safe settings as Phase 1:

- AR: `batch_size=128, grad_accum_steps=2` → effective 256.
- BD3-LM: `batch_size=32, grad_accum_steps=8` → effective 256.
- Curriculum stages switch between the two.

### 9.6  Optimizer and LR

- AdamW: reuse calibrated 50M LR = 1e-3.
- NorMuon (deferred): `adam_mult=0.3, matrix_mult=1.0`.
- Warmup-stable schedule, `warmup_frac=0.05`.
- Reset optimizer state and LR at every curriculum stage transition;
  warm-start from previous stage weights only.

### 9.7  Stage allocation

Token-matched, not FLOP-matched.  Stage fractions = step fractions.
`stage_steps_i = round(total_steps × stage_fraction_i)`, rounding remainder
to the final stage.

### 9.8  Checkpoints and evaluation

- Save weights-only checkpoints at steps
  `{100, 200, 300, 400, 500, 600, 700, 800, 900, 954}`.
- Post-hoc evaluation of every saved checkpoint:
  - Validation BPB.
  - HellaSwag `acc_norm`.
- Primary metric: HellaSwag vs total tokens.
- Secondary metric: BPB vs total tokens.

### 9.9  Crossover summary

For each non-AR method, record:

- `first_step_above_ar`, `first_tokens_above_ar`, `margin_at_crossover`.
- `peak_margin_vs_ar`, `final_margin_vs_ar`.
- Meaningful crossover = HellaSwag exceeds AR by ≥0.5 points for ≥2
  consecutive evaluated checkpoints.

### 9.10  Plotting

- HellaSwag and BPB vs total tokens, one panel per U.
- Final and peak HellaSwag vs U.
- Crossover step vs U for each curriculum.
