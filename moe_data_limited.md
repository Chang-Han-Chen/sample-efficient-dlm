# MoE Data-Limited AR Findings

Date: 2026-04-28

This note summarizes the 25M-token AR MoE data-limited probes launched with
`scripts/ar_data_limited.py`. The goal is to understand whether weight decay and
learning-rate changes can reduce rapid overfitting on the small fixed training
split.

## Setup

- Model preset: `--model moe`
- Config: `configs/experiments/ar_moe_old_bundle.yaml`
- Train split: `data/climbmix_25m_newtok_8192/tokens/train`
- Validation split: `data/climbmix_25m_newtok_8192/tokens/val`
- Train tokens: `25,000,000`
- Batch size: `512`
- Context length: `512`
- Tokens per optimizer step: `262,144`
- Steps per data epoch: about `95.37`
- Default full run length: `5100` steps, about `53.5` epochs
- Checkpoints: disabled
- Eval: every `10` steps, `4` eval batches
- Default optimizer WD values:
  - `muon_wd = 0.0001`
  - `adam_wd = 0.0`
  - `router_adam_wd = 0.0`

The default MoE preset adds:

- `lr_mult = 2.0`
- `moe_router_z_loss_weight = 0.01`
- `--no-moe-split-router-input`

## Run Summary

Snapshot from local JSONL logs. All runs below were launched without checkpoint
saving, so "best eval" is an observed validation minimum, not a saved checkpoint.

| Run | Status | LR mult | Muon WD | Adam WD | Router WD | Last step | Best eval | Best step | Last eval | Last train loss | MoE notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `dl25m_ar_moe_defaultwd_s5100_seed42` | stopped | 2.0 | 0.0001 | 0.0 | 0.0 | 1059 | 3.577985 | 400 | 4.226126 | 2.024114 | Drop clean; severe train/val divergence |
| `dl25m_ar_moe_allwd0p1_s5100_seed42` | stopped | 2.0 | 0.1 | 0.1 | 0.1 | 3189 | 3.589204 | 2150 | 3.718380 | 2.613893 | Slower overfit; recent drop about 1.1% |
| `dl25m_ar_moe_muonwd0p001_s5100_seed42` | stopped | 2.0 | 0.001 | 0.0 | 0.0 | 856 | 3.619962 | 430 | 3.919021 | 2.346807 | Drop clean; still overfit |
| `dl25m_ar_moe_muonwd0p01_s5100_seed42` | stopped | 2.0 | 0.01 | 0.0 | 0.0 | 1585 | 3.644322 | 480 | 4.574289 | 1.905006 | Drop clean; overfit badly |
| `dl25m_ar_moe_muonwd0p0125_lr2_s5100_seed42` | stopped early | 2.0 | 0.0125 | 0.0 | 0.0 | 310 | 3.796802 | 290 | 3.807392 | 3.489748 | Too short; no useful signal |
| `dl25m_ar_moe_muonwd0p025_lr2_s5100_seed42` | stopped | 2.0 | 0.025 | 0.0 | 0.0 | 2282 | 3.617718 | 830 | 3.869123 | 2.878026 | Worse than 0.1; late drop rose to 7.4% |
| `dl25m_ar_moe_muonwd0p05_lr2_s5100_seed42` | stopped | 2.0 | 0.05 | 0.0 | 0.0 | 1795 | 3.578665 | 400 | 3.746347 | 2.614028 | Close to 0.1, but slightly worse and overfit earlier |
| `dl25m_ar_moe_muonwd0p1_s5100_seed42` | stopped | 2.0 | 0.1 | 0.0 | 0.0 | 2519 | 3.575484 | 650 | 3.757663 | 2.488980 | Strong local setting; still drifts up |
| `dl25m_ar_moe_muonwd0p2_lr2_s5100_seed42` | completed | 2.0 | 0.2 | 0.0 | 0.0 | 5099 | **3.557512** | 3110 | 5.044731 | 4.934090 | Best minimum, then late instability; recent drop 6.2%, last drop 9.2% |
| `dl25m_ar_moe_muonwd1_s5100_seed42` | stopped | 2.0 | 1.0 | 0.0 | 0.0 | 2269 | 3.590670 | 1430 | 3.812131 | 3.547177 | Unstable; recent MoE drop spiked near 8-11% |
| `dl25m_ar_moe_muonwd1_lr1_s2500_seed42` | completed | 1.0 | 1.0 | 0.0 | 0.0 | 2499 | 3.562334 | 2250 | 3.636439 | 3.038061 | Best high-WD LR probe; MoE drop stayed low |
| `dl25m_ar_moe_muonwd1_lr0p5_s2500_seed42` | completed | 0.5 | 1.0 | 0.0 | 0.0 | 2499 | 3.616173 | 1790 | 3.774226 | 3.430976 | Worse; recent drop about 5.2%, last drop 10.0% |
| `dl25m_ar_moe_muonwd1_lr0p25_s2500_seed42` | stopped | 0.25 | 1.0 | 0.0 | 0.0 | 1582 | 3.586979 | 650 | 3.857357 | 2.539731 | Lower LR underfit/then drifted; entropy collapsed lower |

## Findings

### 1. The default run overfits very quickly

The default-WD run peaked at step `400` with eval `3.577985`, then validation
loss rose sharply while training loss kept falling:

- Best eval: `3.577985 @ 400`
- Last eval before stop: `4.226126 @ 1050`
- Last train loss: `2.024114 @ 1059`

MoE drop was clean near the stop point, so this was not primarily a routing
collapse. It was plain small-data overfitting.

### 2. Applying WD to all parameter groups delayed overfit

The all-WD run used:

- `muon_wd = 0.1`
- `adam_wd = 0.1`
- `router_adam_wd = 0.1`

It did not improve the best eval relative to default, but it delayed the best
point from step `400` to step `2150`:

- Default best: `3.577985 @ 400`
- All-WD 0.1 best: `3.589204 @ 2150`

This suggests stronger regularization can keep the model from memorizing as
quickly, but all-WD at `0.1` was not enough to improve the best validation loss.

### 3. Muon-only WD has a non-monotonic effect

Muon-only WD keeps Adam/table and router WD at zero. At `lr_mult=2.0`, the sweep
so far is:

| Muon WD | Best eval | Best step | Last eval | Comment |
| ---: | ---: | ---: | ---: | --- |
| 0.0001 | 3.577985 | 400 | 4.226126 | Default; fast overfit |
| 0.001 | 3.619962 | 430 | 3.919021 | Worse than default |
| 0.01 | 3.644322 | 480 | 4.574289 | Worse and later overfit was severe |
| 0.0125 | 3.796802 | 290 | 3.807392 | Stopped too early to interpret |
| 0.025 | 3.617718 | 830 | 3.869123 | Worse than 0.1 |
| 0.05 | 3.578665 | 400 | 3.746347 | Close to 0.1, but slightly worse |
| 0.1 | 3.575484 | 650 | 3.757663 | Strong local setting |
| 0.2 | **3.557512** | 3110 | 5.044731 | Best minimum but unstable late |
| 1.0 | 3.590670 | 1430 | 3.812131 | Delayed peak but unstable at LR 2.0 |

The new best observed eval is `muon_wd=0.2, lr_mult=2.0`, with
`3.557512 @ 3110`. This is a meaningful improvement over both default
(`3.577985`) and `muon_wd=0.1` (`3.575484`).

However, the same run became unstable near the end:

- Best eval: `3.557512 @ 3110`
- Last eval: `5.044731 @ 5090`
- Last train loss: `4.934090 @ 5099`
- Recent MoE drop average: about `6.2%`
- Last MoE drop: about `9.2%`

So `muon_wd=0.2` is the best setting by validation minimum, but it needs early
stopping or best-checkpoint saving. With checkpoints disabled, the observed best
point is not recoverable from this run.

### 4. LR tuning helped at very high Muon WD

At `muon_wd=1.0, lr_mult=2.0`, the run was unstable:

- Best eval: `3.590670 @ 1430`
- Last eval: `3.812131 @ 2260`
- Recent MoE drop: about `11.3%`

Lowering LR while holding `muon_wd=1.0` fixed gave:

| LR mult | Best eval | Best step | Last eval | Comment |
| ---: | ---: | ---: | ---: | --- |
| 2.0 | 3.590670 | 1430 | 3.812131 | Unstable; high drop |
| 1.0 | 3.562334 | 2250 | 3.636439 | Much better and healthier |
| 0.5 | 3.616173 | 1790 | 3.774226 | Worse; drop rose late |
| 0.25 | 3.586979 | 650 | 3.857357 | Worse; stopped early |

The best high-WD LR probe is `muon_wd=1.0, lr_mult=1.0`, with
`3.562334 @ 2250`. It is slightly worse than `muon_wd=0.2, lr_mult=2.0`, but it
had cleaner late MoE health through 2500 steps.

## Interpretation

The dominant problem is still not simple router collapse. Most bad runs start
overfitting before MoE drop becomes severe. But stronger Muon WD changes the
failure mode:

- Low WD memorizes quickly and validation loss rises while training loss falls.
- Moderate WD (`0.1` to `0.2`) delays the best point and can improve the minimum.
- Too much WD or too high LR eventually causes optimizer/router instability,
  visible as eval spikes, gradient spikes, and increased MoE drop.

The best observed validation minimum is now:

`muon_wd=0.2, adam_wd=0.0, router_adam_wd=0.0, lr_mult=2.0`

with best eval `3.557512 @ 3110`.

The most stable strong candidate is:

`muon_wd=1.0, adam_wd=0.0, router_adam_wd=0.0, lr_mult=1.0`

with best eval `3.562334 @ 2250` and much cleaner MoE drop through the 2500-step
probe.

Because the current runs disable checkpoint saving, selecting by best eval alone
is risky. Any serious rerun should enable best-checkpoint saving or use an early
stop around the expected best window.

## Next Steps

1. Rerun `muon_wd=0.2, lr_mult=2.0` with best-checkpoint saving, or cap/early
   stop around `3000-3300` steps.
2. Probe around the new best with values such as `muon_wd=0.15`, `0.2`, and
   `0.3`, preferably with best checkpoint enabled.
3. Consider `muon_wd=0.2` with a slightly lower LR, for example `lr_mult=1.5` or
   `1.0`, to see whether the late instability can be avoided without losing the
   lower validation minimum.
4. Keep selecting by both best eval and health. A run that reaches a low eval
   but explodes later is useful only if the best checkpoint is saved.
