# BD3 Curriculum Monitor Instructions

This sweep tests whether AR or MDLM initialization helps BD3LM training when
the architecture is the current best MoE old-bundle recipe.

Use:

```bash
python scripts/bd3_curriculum_launcher.py list --eligible --limit 20
python scripts/bd3_curriculum_launcher.py launch-next --source ar
python scripts/bd3_curriculum_launcher.py launch-next --source mdlm
python scripts/bd3_curriculum_launcher.py summarize
```

The launcher state lives in `status.json`. Treat that file as the source of
truth for queue state, manual decisions, and result summaries.

## Scientific Question

For each BD3 block length in `[256, 64, 16, 4]`, compare:

- BD3-from-scratch with MoE old-bundle.
- AR-initialized BD3 at `p_ar in [0.30, 0.50, 0.80, 0.90, 0.95]`.
- MDLM-initialized BD3 at `p_mdlm in [0.30, 0.50, 0.80, 0.90, 0.95]`.

Do not add BD3 block length `512`. At sequence length `512`, that endpoint is
MDLM and is trained separately.

The metric is validation loss only. Lower is better.

## Fixed Recipe

Use the MoE old-bundle architecture throughout:

- QK-norm on.
- Attention value residual on.
- Per-head attention gating on.
- LayerNorm scaling on.
- Value embeddings off.
- Alternating Switch MoE layers.
- `4` top-1 experts.
- Capacity factor `1.25`.
- Non-split router input.
- Router z-loss weight `0.01`.
- Base LR multiplier `2.0`.

All checkpoints must save and restore both model weights and optimizer state.
Do not intentionally do weights-only restoration in the main sweep.

Important implementation detail: AR source runs use model vocab size `8193`,
with the extra token row unused during AR training. This makes AR checkpoints
shape-compatible with diffusion/BD3 checkpoints, where token id `8192` is the
mask token. Do not remove this unless `train_ar.py` grows explicit partial
checkpoint loading.

## Queue Structure

The launcher initializes these phases:

- `source`: segmented AR and MDLM source training to exact checkpoint counts
  `3100`, `5100`, `8100`, `9100`, `9600`, and `10100`.
- `scratch_full`: BD3-from-scratch full runs for block lengths
  `[256, 64, 16, 4]`.
- `switch_probe`: AR/MDLM checkpoint -> BD3 short LR probes.
- `switch_full`: created later with `promote` after a probe is selected.

Source checkpoints are segmented because `train_ar.py` only saves regular
intervals or final checkpoints. Segmenting avoids saving a large checkpoint
every 100 steps while still preserving optimizer state at the exact source
fractions.

## Two-VM Split

On the AR-init VM:

```bash
python scripts/bd3_curriculum_launcher.py launch-next --phase source --source ar
python scripts/bd3_curriculum_launcher.py launch-next --phase switch_probe --source ar
```

On the MDLM-init VM:

```bash
python scripts/bd3_curriculum_launcher.py launch-next --phase source --source mdlm
python scripts/bd3_curriculum_launcher.py launch-next --phase switch_probe --source mdlm
```

Run `scratch_full` on whichever VM has free capacity:

```bash
python scripts/bd3_curriculum_launcher.py launch-next --phase scratch_full
```

If the two VMs do not share a filesystem, keep separate copies of
`status.json` and merge the result summaries manually. Do not run two agents
against the same `status.json` on a non-locking shared filesystem.

## Probe Policy

Each switch probe uses one of:

- `lr_mult = 0.6666667`
- `lr_mult = 2.0`
- `lr_mult = 6.0`

The probe length is `1050` post-switch optimizer steps. Because optimizer state
is restored, the LR schedule does not reset to a fresh 50-step warmup; this is
intentional for the main sweep. Interpret the first few evals after the switch
as adaptation diagnostics rather than final-budget comparisons.

Promote at most one LR per `(source, p, block_len)` to a full continuation.
Prefer promoting only after at least three validation points after the objective
switch unless a run is clearly unstable.

Promotion command:

```bash
python scripts/bd3_curriculum_launcher.py promote RUN_ID --reason "best val loss among LR probes"
python scripts/bd3_curriculum_launcher.py launch FULL_RUN_ID
```

Full continuation runs always end at total step `10100`, so comparisons against
scratch are equal-total-step validation-loss comparisons.

## Early Stop Rules

Only use validation loss for early-stop decisions.

Reasonable early stops:

- NaN/inf loss or repeated crashed runs.
- Router collapse: sustained high dropped fraction together with worsening
  validation loss.
- A probe is clearly worse than the best same `(source, p, block_len)` LR by
  at least `0.05` eval loss for three consecutive evals.
- A probe is worse than the scratch same-block run by at least `0.10` eval loss
  after several post-switch evals and is not improving.

When stopping manually, mark the run:

```bash
python scripts/bd3_curriculum_launcher.py mark RUN_ID stopped --note "reason"
```

Do not change architecture flags mid-sweep. If instability appears, first try a
lower LR multiplier as a new status entry or by rerunning an existing failed
probe with `--force`.

## Result Comparison

After runs finish:

```bash
python scripts/bd3_curriculum_launcher.py summarize
```

For each block length, report:

- Best scratch final validation loss.
- Best AR-init full final validation loss and its `p_ar`.
- Best MDLM-init full final validation loss and its `p_mdlm`.
- Absolute and percent improvement over same-block scratch:
  `(scratch_final_eval - curriculum_final_eval) / scratch_final_eval`.

Do not compare different block lengths as if they were the same model family.
The first question is whether initialization helps at fixed block length; the
second question is which block length wins overall.
