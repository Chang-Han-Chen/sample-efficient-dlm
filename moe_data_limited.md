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

Validate that the curriculum result `p_ar=0.30, b=4` from
`BD3_CURRICULUM_FINDINGS.md` (compute-bound, 1.7 epochs) also holds in the
data-constrained, multi-epoch regime.

### Readiness

The repo now has the pieces needed to run this end to end once GPU/data prep is
available:

- `data/prepare_climbmix.py` prepares a tokenized ClimbMix source with N train
  shards, one eval shard, and a fresh byte-level BPE tokenizer.
- `scripts/moe_data_limited_curriculum.py` derives 0.5x, 1x, and 2x-shard
  datasets, computes exact token counts/32-epoch step counts from metadata, and
  launches the BD3 curriculum queue.
- The launcher applies weight decay to all optimizer families for this sweep:
  `muon_wd = adam_wd = router_adam_wd`.
- `train_ar.py` already supports the required mechanics: AR source checkpoints,
  BD3 restore from AR, local best/final checkpoints, directory-backed `.npy`
  datasets, data parallelism, and MoE health metrics.

I have not run the actual training locally because this needs GPUs.

### Data prep

Prepare the source once from `data/`:

```bash
cd data
python prepare_climbmix.py \
  --output-dir climbmix_10x_newtok_8192 \
  --num-shards 10 \
  --val-shard 6542 \
  --vocab-size 8192
cd ..
```

Then derive the three data-limited subsets:

```bash
python scripts/moe_data_limited_curriculum.py prepare-datasets \
  --source-data-root data/climbmix_10x_newtok_8192 \
  --labels u25mish u50mish u100mish
```

The derived datasets are:

| Label | Dataset dir | Selection | Target size |
| --- | --- | --- | --- |
| `u25mish` | `data/climbmix_0p5x_newtok_8192` | first half of `shard_00000` | ~25M tokens |
| `u50mish` | `data/climbmix_1x_newtok_8192` | all of `shard_00000` | ~50M tokens |
| `u100mish` | `data/climbmix_2x_newtok_8192` | all of `shard_00000` and `shard_00001` | ~100M tokens |

Use the exact `train_token_count` and `steps_for_epochs` fields written to each
dataset's `metadata.json` for reporting.

### Grid

12 runs, single seed each, on `4xH100`.

| Axis | Values |
| --- | --- |
| Method | `p_ar=0` (scratch BD3 b=4); `p_ar=0.30` (AR -> BD3 b=4) |
| U (unique tokens) | `u25mish`, `u50mish`, `u100mish` |
| WD | 0, 0.1 applied as `muon_wd = adam_wd = router_adam_wd` |

Total: 2 methods x 3 U x 2 WD = 12 runs.

For `p_ar=0.30`, each experiment cell is two process runs: an AR prefix run
through 30% of the total steps, then a BD3 b=4 run restored from the AR final
checkpoint through the full 32-epoch step count.

### Steps per U

Tokens per optimizer step = `512 x 512 = 262,144`. Steps are computed after
data prep as:

`ceil(32 x train_token_count / 262,144)`

Approximate planning numbers:

| U | Total training-token budget | Approx steps |
| ---: | ---: | ---: |
| 25M | 800M | 3,050 |
| 50M | 1.6B | 6,100 |
| 100M | 3.2B | 12,200 |

### Fixed hyperparameters

| Parameter | Value |
| --- | --- |
| AR phase config | `configs/experiments/ar_moe_old_bundle.yaml` with `--vocab-size 8193` for BD3 restore compatibility |
| BD3 config | `configs/experiments/mdlm_moe_old_bundle.yaml` |
| BD3 block length | 4 |
| BD3 / scratch `lr_mult` | 2.0 |
| AR-phase `lr_mult` (curriculum only) | 5.0 |
| WD scope | `muon_wd = adam_wd = router_adam_wd = {0.0, 0.1}` |
| MoE override | `--no-moe-split-router-input`, `--moe-router-z-loss-weight 0.01` |
| Dropout | 0.0 |
| Eval cadence | every 200 steps |
| Checkpoints | local best and final enabled; W&B checkpoint artifact upload disabled |
| Devices | 4 H100 |
| Seeds | 1 per cell |

### Initial launch

Initialize the full queue, but start with only `u25mish`:

```bash
python scripts/moe_data_limited_curriculum.py init \
  --data-labels u25mish u50mish u100mish \
  --force

python scripts/moe_data_limited_curriculum.py list --data-label u25mish
python scripts/moe_data_limited_curriculum.py next --data-label u25mish --command
python scripts/moe_data_limited_curriculum.py launch-next --data-label u25mish
```

For `u25mish`, this corresponds to 4 experiment cells and 6 process runs:

- `p_ar=0.0`, `wd=0.0`: scratch BD3 b=4.
- `p_ar=0.0`, `wd=0.1`: scratch BD3 b=4.
- `p_ar=0.3`, `wd=0.0`: AR prefix, then BD3 b=4 restored from AR final.
- `p_ar=0.3`, `wd=0.1`: AR prefix, then BD3 b=4 restored from AR final.

The launcher will only start a `p_ar=0.3` BD3 continuation after the matching AR
final checkpoint exists.

### Compute estimate (4xH100)

Step times: scratch BD3 b=4 ~= 0.75 s/step; curriculum p=0.30 b=4 ~= 0.57 s/step.

| U | scratch run | curriculum run |
| ---: | ---: | ---: |
| 25M | 38 min | 29 min |
| 50M | 76 min | 58 min |
| 100M | 153 min | 116 min |

Per-WD subtotal: ~7.8 h. Both WDs: **~15.7 h wall-clock total**.

### Reporting

For each (U, method, wd) tuple log: `best_eval`, `best_step`, `final_eval`,
recent MoE drop fraction, recent router entropy. Format the results as a
2 x 3 x 2 table and drop into `final_report.tex` Section 6.

### Pass criterion

The curriculum claim holds in the data-constrained regime if, at every U and
at the better-of-`{wd=0, wd=0.1}` setting, `p_ar=0.30` matches or beats
`p_ar=0` on `best_eval`.

### Launch order

1. Run all `u25mish` cells first.
2. Inspect loss curves, best checkpoints, MoE drop fraction, and router entropy.
3. If healthy, continue the queue for `u50mish` and `u100mish`.
