# Dense AR Data-Limited Findings

Date: 2026-04-28

This note tracks dense AR runs on the fixed 25M-token `climbmix` subset.

## Setup

- Model/config: `configs/experiments/ar_old_bundle.yaml`
- Data: `data/climbmix_25m_newtok_8192/tokens/train`
- Train tokens: `25,000,000`
- Eval data: held-out `shard_06542.npy` via `data/climbmix_25m_newtok_8192/tokens/val`
- Eval tokens per logged eval: `4 * 128 * 512 = 262,144`
- Context length: `512`
- Batch size: `512`
- Devices: `4` H100s
- Checkpoints: disabled locally and on W&B
- Logging: local JSONL plus W&B under group `data_limited_ar_25m`

At this batch/context, one optimizer step sees `262,144` train tokens. The
25M-token subset is therefore about `95.37` steps per epoch, and a 5100-step run
is about `53.48` epochs.

## Completed And Stopped Runs

| Run | lr_mult | Adam WD | Muon WD | Last step | Best eval | Best step | Last eval | Last train loss | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `dl25m_ar_dense_defaultwd_s5100_seed42` | `5.0` | `0.0` | `0.0001` | `1709` | `3.512644` | `400` | `4.196504` | `2.014689` | stopped, clear overfit |
| `dl25m_ar_dense_muonwd0p001_s5100_seed42` | `5.0` | `0.0` | `0.001` | `2453` | `3.524271` | `720` | `3.953798` | `2.346322` | stopped, clear overfit |
| `dl25m_ar_dense_muonwd0p01_s5100_seed42` | `5.0` | `0.0` | `0.01` | `2379` | `3.511364` | `1280` | `3.556996` | `2.762221` | stopped, noncompetitive |
| `dl25m_ar_dense_muonwd0p05_s5100_seed42` | `5.0` | `0.0` | `0.05` | `5099` | `3.417714` | `4820` | `3.504103` | `2.883603` | completed |
| `dl25m_ar_dense_muonwd0p1_s5100_seed42` | `5.0` | `0.0` | `0.1` | `5099` | `3.411788` | `4520` | `3.472106` | `3.037601` | completed |
| `dl25m_ar_dense_muonwd0p2_s5100_seed42` | `5.0` | `0.0` | `0.2` | `3207` | `3.499502` | `3000` | `5.672541` | `5.407581` | stopped, unstable/noncompetitive |
| `dl25m_ar_dense_allwd0p1_s5100_seed42` | `5.0` | `0.1` | `0.1` | `5099` | `3.430136` | `4260` | `3.534986` | `3.232335` | completed |
| `dl25m_ar_dense_muonwd0p1_lr2p0_s5100_seed42` | `2.0` | `0.0` | `0.1` | `3261` | `3.449874` | `2300` | `3.500124` | `2.789961` | stopped, slower/worse than lr=5 |
| `dl25m_ar_dense_muonwd0p1_lr0p8_s5100_seed42` | `0.8` | `0.0` | `0.1` | `1499` | `3.536885` | `650` | `3.709367` | `2.665873` | stopped, too slow/overfitting |

The accidental `dl25m_ar_dense_allwd0p1_muonlr0p001_s5100_seed42` run only
logged step 0 and should not be treated as an experiment.

## Findings So Far

The default dense AR config severely overfits in the 25M-token regime. Its best
held-out eval was `3.512644` at step `400`, but eval degraded to `4.196504` by
step `1709` while train loss kept falling to `2.014689`.

Increasing only Muon weight decay is the key improvement so far. `muon_wd=0.001`
and `0.01` were not enough. The 2x bracket around `0.1` shows the optimum is
near this range: `muon_wd=0.05` finished very close at `3.417714`, while
`muon_wd=0.1` remains narrowly best at `3.411788`.

The `muon_wd=0.05` run should not be described as simply overfitting. It looked
weak around steps `1000`-`2000`, but then recovered and improved through late
training, reaching its best eval at step `4820`. Its last train loss was lower
than the `0.1` run (`2.883603` versus `3.037601`), but eval was only slightly
worse. The better interpretation is that the data-limited dense optimum is a
regularization/optimization tradeoff around `0.05`-`0.1`, not a monotonic
overfit curve.

`muon_wd=0.2` was worse and became unstable after step `3000`, with eval jumping
to `5.672541`. This makes the high-WD side clearly noncompetitive.

Applying `0.1` weight decay to all optimizer groups helped substantially versus
default, but was slightly worse than Muon-only decay at the same value:
`3.430136` versus `3.411788`. This suggests the main useful regularization is on
the Muon matrix parameters, and Adam/table/scalar decay may not be necessary for
dense AR.

The first LR sweep points at this WD show that lower LR is worse. `lr_mult=2.0`
was slower and worse than the original `lr_mult=5.0` setting, reaching best eval
`3.449874` before it was stopped at step `3261`. `lr_mult=0.8` was worse still:
best eval `3.536885`, followed by eval drift to `3.709367` by step `1499`.

## Current Best Dense Configuration

`dense AR old bundle, lr_mult=5.0, muon_wd=0.1, adam_wd=0.0`

- Best eval: `3.411788`
- Best step: `4520`
- Last eval: `3.472106`
- Last train loss: `3.037601`

## Next Checks

- Test a high-side LR around `7.5` or `8.0` at `muon_wd=0.1`.
- Optionally test an intermediate Muon WD such as `0.075`, since `0.05` and
  `0.1` are nearly tied.
- For the dense conclusion, report both best eval and overfit behavior, not only
  final eval.
