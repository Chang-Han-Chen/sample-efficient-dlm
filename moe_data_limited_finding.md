# MoE Data-Limited Curriculum Interim Findings

Date: 2026-05-08

This is an interim note for the `u25mish` data-limited curriculum sweep from
`moe_data_limited.md`. The first scratch BD3 baseline, the `p_ar=0.30, wd=0`
AR prefix, and the matching BD3 continuation are complete.

## Setup

Prepared source data:

- Source root: `data/climbmix_10x_newtok_8192`
- Train shards: `shard_00000` through `shard_00009`
- Validation shard: `shard_06542`
- Tokenizer: fresh byte-level BPE, vocab size 8192

Derived `u25mish` dataset:

- Root: `data/climbmix_0p5x_newtok_8192`
- Selection: first half of source `shard_00000`
- Train tokens: `32,186,623`
- Tokens per optimizer step: `262,144`
- Steps per epoch: `122.78222274780273`
- 32-epoch steps: `3,930`

Queue controls after the first scratch run:

- Eval cadence: every 50 steps
- Route probe: 10 fixed eval samples
- Route probe source: eval split
- Local best and final checkpoints enabled
- W&B checkpoint artifact upload disabled

The first scratch BD3 run used the original 200-step eval cadence and did not
have route-probe logging, because the probe was added after that run started.

## Curve Recovery

Local `runs/moe_data_limited_curriculum/` outputs are intentionally ignored and
should not be pushed. To recover loss curves later, use W&B:

- Project: `y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm`
- Group: `moe_data_limited_curriculum`
- W&B run name: same as the local run ID in the first column
- Useful history keys: `step`, `loss`, `eval_loss`, `moe_dropped_fraction`,
  `moe_router_entropy`, `moe_expert_fraction_std`, and route-probe metrics when
  present

| W&B run name / local run ID | W&B id |
| --- | --- |
| `u25mish_p000_bd3_b4_allwd0` | [`cgzujzzq`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/cgzujzzq) |
| `u25mish_p030_ar_allwd0` | [`ox94pu29`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/ox94pu29) |
| `u25mish_p030_bd3_b4_allwd0` | [`edix6e1p`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/edix6e1p) |
| `u25mish_ar0p5ep_ar_allwd0` | [`q05utpyy`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/q05utpyy) |
| `u25mish_ar0p5ep_bd3_b4_allwd0` | [`eyyh2dx6`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/eyyh2dx6) |
| `u25mish_ar1ep_ar_allwd0` | [`h75xg07e`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/h75xg07e) |
| `u25mish_ar1ep_bd3_b4_allwd0` | [`l46jw78o`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/l46jw78o) |
| `u25mish_p000_bd3_b4_allwd0p1` | [`m6yvybpg`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/m6yvybpg) |
| `u25mish_ar1000_muw0p1_adamwd0p01_ar` | [`bhdily2x`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/bhdily2x) |
| `u25mish_ar1000_muw0p1_adamwd0_ar` | [`ad82jsrv`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/ad82jsrv) |
| `u25mish_ar1000_muw0p1_adamwd0p01_routerwd0_ar` | [`ij6v7stn`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/ij6v7stn) |
| `u25mish_ar1000_muw0p1_adamwd0p001_routerwd0_ar` | [`cgiuc8h8`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/cgiuc8h8) |
| `u25mish_ar1000_muw0p05_adamwd0p001_routerwd0_ar` | [`btsib84z`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/btsib84z) |
| `u25mish_p030_ar_muw0p1_adamwd0p01_routerwd0` | [`9ajohd3l`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/9ajohd3l) |
| `u25mish_p030_bd3_b4_from_ar_muw0p1_adamwd0p01_routerwd0_bd3wd0` | [`jdeix8ec`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/jdeix8ec) |
| `u25mish_p030_ar_muw0p1_adamwd0p01_routerwd0_seed43` | [`ll97lpn7`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/ll97lpn7) |
| `u25mish_p030_ar_muw0p1_adamwd0p001_routerwd0` | [`ll0n57sx`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/ll0n57sx) |
| `u25mish_p030_bd3_b4_from_ar_muw0p1_adamwd0p001_routerwd0_bd3wd0` | [`cp6i45e6`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/cp6i45e6) |
| `u25mish_ar1000_lr2_muw0_adamwd0_routerwd0_ar` | [`50j9od7k`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/50j9od7k) |
| `u25mish_ar1000_lr2_muw0p1_adamwd0p001_routerwd0_ar` | [`5rpi8ko7`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/5rpi8ko7) |
| `u25mish_ar1000_lr2_muw0p05_adamwd0p001_routerwd0_ar` | [`dh7f1cg0`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/dh7f1cg0) |
| `u25mish_ar1000_lr3_muw0p1_adamwd0p001_routerwd0_ar` | no synced W&B URL recorded in launcher log |
| `u50mish_p000_bd3_b4_allwd0` | [`rnxjykt5`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/rnxjykt5) |
| `u50mish_p030_ar_allwd0` | [`605i3ldz`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/605i3ldz) |
| `u50mish_p030_bd3_b4_allwd0` | [`xfdmtqpt`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/xfdmtqpt) |
| `u50mish_p030_ar_muw0p1_adamwd0p001_routerwd0` | [`jwal4t2x`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/jwal4t2x) |
| `u50mish_p030_bd3_b4_from_ar_muw0p1_adamwd0p001_routerwd0_bd3wd0` | [`kwn1rbxm`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/kwn1rbxm) |
| `u50mish_p030_ar_muw0p05_adamwd0p0003_routerwd0` | [`t1f9tj44`](https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/t1f9tj44) |

## Current Results

| Run | Status | Method | WD | Best eval | Best step | Best epoch | Final/latest eval | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `u25mish_p000_bd3_b4_allwd0` | complete | scratch BD3 b=4 | 0.0 | **3.207904** | 3600 | 29.32 | 3.246922 @ 3800 | Strong baseline; late mild router pressure but stable |
| `u25mish_p030_ar_allwd0` | complete | AR prefix | 0.0 | 3.528117 | 500 | 4.07 | 3.926234 @ 1150 | AR overfits after early best; final checkpoint feeds BD3 |
| `u25mish_p030_bd3_b4_allwd0` | complete | BD3 b=4 after AR | 0.0 | 3.488358 | 2200 | 17.92 | 3.589082 @ 3900 | Adapted quickly from AR, then plateaued far behind scratch |

The `p_ar=0.30, wd=0` curriculum cell is not yet competitive with scratch
BD3 on `best_eval`. The BD3 continuation improved rapidly immediately after
switching objectives:

| Step | Eval loss |
| ---: | ---: |
| 1200 | 4.914911 |
| 1250 | 4.096647 |
| 1300 | 3.825283 |
| 1450 | 3.638186 |
| 1800 | 3.497221 |
| 2200 | **3.488358** |

After about step 2200 it mostly plateaued above `3.50` and finished at
`3.589082`.

## Interpretation So Far

The `wd=0` result is currently unfavorable for the curriculum claim in this
data-constrained regime. The AR prefix learns quickly but overfits strongly:
its best eval occurs around epoch 4.07, while the final AR eval at the handoff
is much worse. The BD3 continuation can recover a usable BD3 model from the AR
checkpoint, but so far it has not caught the scratch BD3 baseline.

This supports the concern that, in the multi-epoch small-data setting, a long
AR prefix may learn representations that are less useful for the later BD3
objective than starting BD3 directly. This is still preliminary because:

- The `wd=0.1` scratch and curriculum cells have not run yet.
- The pass criterion uses the better of `wd=0` and `wd=0.1`.
- Shorter fixed-epoch AR prefixes have not run yet.

## MoE Health

The scratch BD3 baseline did not fail through router collapse. Its best point
was late, at step 3600. Around late training it showed mild router pressure:

- Step 3600 best eval: `3.207904`
- MoE drop at step 3600 eval row: `0.00676`
- Router entropy at step 3600 eval row: `1.02199`
- Final eval at step 3800: `3.246922`
- MoE drop at step 3800 eval row: `0.02779`

The AR prefix also did not overload experts. It overfit while token drops were
zero and router entropy steadily decreased:

- Best eval at step 500: `3.528117`
- Final eval at step 1150: `3.926234`
- Final AR eval MoE drop: `0.0`
- Final AR eval router entropy: `0.9078`

The BD3 continuation has been MoE-healthy so far:

- Recent eval-row MoE drops are `0.0`
- Router entropy is around `1.08-1.09`
- Grad norms are ordinary after the objective switch

## Route Probe

An opt-in route probe was added to `train_ar.py` and enabled for runs launched
after the scratch baseline:

- `--route-probe-samples 10`
- `--route-probe-source eval`
- Raw route files: `runs/moe_data_limited_curriculum/<run_id>/route_probe/step_*.npz`
- Metrics logged into JSONL/W&B:
  - `route_probe_agree_prev`
  - `route_probe_agree_first`
  - `route_probe_agree_prev_xt` / `route_probe_agree_prev_x0` for BD3
  - `route_probe_dropped_fraction`
  - `route_probe_selected_prob_mean`
  - `route_probe_margin_mean`

For AR, the probe stores expert IDs shaped `(4, 10, 512)`.
For BD3, the probe stores expert IDs shaped `(4, 10, 1024)` for `x_t || x_0`.

Early route-probe observations:

- AR routes rapidly move away from initialization.
- AR agreement with the previous probe rises to about `0.89` by the end.
- AR agreement with the first probe stays near `0.24`, close to a random
  four-expert baseline.
- BD3 continuation routes are stable between adjacent evals, about `0.90-0.92`
  agreement in the later observed window.
- In BD3, clean-stream routes are slightly more stable than noisy-stream
  routes. Around step 1800, previous-probe agreement was about `0.910` on
  `x_t` and `0.924` on `x_0`.

This statistic looks meaningful as a router-stability diagnostic. It should not
be interpreted by itself as proof of semantic expert specialization.

## Follow-Up

Do not use `p_ar=0.50` as the next diagnostic. It is too long to distinguish
"AR helps briefly" from "AR overfits when run for many small-data epochs".

Instead, run fixed-epoch AR prefixes:

| Diagnostic | AR steps | AR epochs | BD3 continuation |
| --- | ---: | ---: | --- |
| `u25mish_ar0p5ep_*_allwd0` | 62 | 0.5 | Through step 3930 |
| `u25mish_ar1ep_*_allwd0` | 123 | 1.0 | Through step 3930 |
| `u25mish_ar4ep_*_allwd0` | 492 | 4.0 | Through step 3930 |

Hypothesis: if the problem is AR overfitting before handoff, shorter AR
prefixes should be less harmful than the current `p_ar=0.30` prefix, which ran
1179 AR steps, about 9.60 data epochs. The 4-epoch diagnostic is especially
interesting because it is close to the observed best AR validation point
(`step=500`, about 4.07 epochs), while still much shorter than the current
handoff.

These six diagnostic process runs have been inserted into the current queue
after `u25mish_p030_bd3_b4_allwd0` and before the `wd=0.1` cells.

Keep for these diagnostics:

- `u25mish`, `wd=0`
- BD3 b=4 continuation from each AR final checkpoint
- Eval cadence at 50
- The 10-sample route probe enabled

Do not expand to `u50mish` or `u100mish` based on the current evidence alone.
Finish the remaining `u25mish` cells first, especially `wd=0.1`.

## 2026-05-08 Update: Split WD and Next Sweep

The all-WD interpretation was too coarse. Applying `wd=0.1` to scratch BD3
was unstable and produced a much worse baseline:

| Run | Best eval | Best step | Final/latest eval | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `u25mish_p000_bd3_b4_allwd0` | **3.207904** | 3600 | 3.246922 @ 3800 | Best scratch baseline |
| `u25mish_p000_bd3_b4_allwd0p1` | 3.361265 | 1950 | 4.414460 @ 2700 | All-WD BD3 is unstable |

The cleaner hypothesis is now:

- BD3 should use no WD for this setting.
- The AR phase needs regularization, but router WD is harmful.
- Small Adam WD in AR helps; `adam_wd=0.01` is too large, while
  `adam_wd=0.001` with `router_adam_wd=0` is the best tested point so far.
- Muon WD around `0.05-0.1` is plausible; the observed difference between
  these two values is small.

Key `u25mish` AR-WD diagnostics:

| Run | AR LR mult | Muon WD | Adam WD | Router WD | Best AR eval | Best step | Final eval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `u25mish_ar1000_muw0p1_adamwd0_ar` | 5 | 0.1 | 0 | 0 | 3.522370 | 900 | 3.551810 |
| `u25mish_ar1000_muw0p1_adamwd0p01_routerwd0_ar` | 5 | 0.1 | 0.01 | 0 | 3.637392 | 550 | 3.660683 |
| `u25mish_ar1000_muw0p1_adamwd0p001_routerwd0_ar` | 5 | 0.1 | 0.001 | 0 | **3.460879** | 850 | 3.504823 |
| `u25mish_ar1000_muw0p05_adamwd0p001_routerwd0_ar` | 5 | 0.05 | 0.001 | 0 | 3.467303 | 850 | 3.488088 |

The corrected full `p_ar=0.30` run used `muon_wd=0.1`, `adam_wd=0.001`,
`router_adam_wd=0` during AR and no WD during BD3. It fixed the catastrophic
late AR handoff and nearly caught scratch, but did not beat it:

| Run | Best eval | Best step | Best epoch | Final eval |
| --- | ---: | ---: | ---: | ---: |
| `u25mish_p030_ar_muw0p1_adamwd0p001_routerwd0` | 3.509549 | 900 | 7.33 | 3.532375 |
| `u25mish_p030_bd3_b4_from_ar_muw0p1_adamwd0p001_routerwd0_bd3wd0` | 3.223083 | 2100 | 17.10 | 3.348229 |

For the next autonomous sweep, the immediate target is the scratch BD3 eval at
matched step 1000: `3.352492`. The best 1k-step AR probe so far is still above
that target. W&B history for the same old-bundle AR MoE recipe suggests
`lr_mult=2` is a better AR LR than the current `lr_mult=5`, so the next probes
should prioritize lower AR LR with router WD off:

| Probe family | AR LR mult | Muon WD | Adam WD | Router WD |
| --- | ---: | ---: | ---: | ---: |
| low-LR no-WD control | 2 | 0 | 0 | 0 |
| low-LR small Adam WD | 2 | 0.1 | 0.001 | 0 |
| low-LR lower Muon WD | 2 | 0.05 | 0.001 | 0 |
| mid-LR small Adam WD | 3 | 0.1 | 0.001 | 0 |

If one of these beats the step-1000 scratch target, promote that setting to a
full `u25mish` `p_ar=0.30` AR phase followed by BD3 with all WD disabled. After
that, move to `u50mish`: run the no-WD baseline/curriculum first, then the
chosen AR-WD curriculum. Do not run `u100mish`.

### U=25M Tuning Stop Point

Additional lower-AR-LR probes did not beat the existing `lr_mult=5` small-WD
probe:

| Run | AR LR mult | Muon WD | Adam WD | Router WD | Best AR eval | Best step | Final/latest eval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `u25mish_ar1000_lr2_muw0_adamwd0_routerwd0_ar` | 2 | 0 | 0 | 0 | 3.491466 | 600 | 3.728109 @ 950 |
| `u25mish_ar1000_lr2_muw0p1_adamwd0p001_routerwd0_ar` | 2 | 0.1 | 0.001 | 0 | 3.496538 | 800 | 3.546918 @ 950 |
| `u25mish_ar1000_lr2_muw0p05_adamwd0p001_routerwd0_ar` | 2 | 0.05 | 0.001 | 0 | 3.484636 | 600 | 3.548296 @ 950 |
| `u25mish_ar1000_lr3_muw0p1_adamwd0p001_routerwd0_ar` | 3 | 0.1 | 0.001 | 0 | 3.7678 | 250 | stopped early |

The `lr_mult=3` probe was stopped after the user decided the U=25M tuning was
sufficient. The best current U=25M AR-WD setting remains:

- AR: `lr_mult=5`, `muon_wd=0.1`, `adam_wd=0.001`, `router_adam_wd=0`
- BD3 continuation: all WD disabled

This setting did not strictly beat the U=25M scratch BD3 best, but it moved the
full curriculum from clearly bad (`3.488358`) to close (`3.223083` versus
scratch `3.207904`). The experiment now moves to `u50mish` with no U=100M runs.

Planned U=50M order:

| Run | Purpose |
| --- | --- |
| `u50mish_p000_bd3_b4_allwd0` | scratch BD3 no-WD baseline |
| `u50mish_p030_ar_allwd0` -> `u50mish_p030_bd3_b4_allwd0` | no-WD curriculum control |
| `u50mish_p030_ar_muw0p1_adamwd0p001_routerwd0` -> `u50mish_p030_bd3_b4_from_ar_muw0p1_adamwd0p001_routerwd0_bd3wd0` | tuned AR-WD curriculum, BD3 no-WD |

## U=50M Results

The U=50M scratch BD3 no-WD baseline is complete:

| Run | Best eval | Best step | Best epoch | Final eval | Final step | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `u50mish_p000_bd3_b4_allwd0` | **2.950898** | 6400 | 26.06 | 2.957172 | 7850 | Strong, stable baseline; intermittent drop spikes but no collapse |

MoE health details:

- Max eval-row drop: `0.139394`, transient.
- Drop at best row: `0.0`.
- Router entropy at best row: `1.114059`.
- Final eval-row drop: `0.0`.
- Final router entropy: `1.044115`.
- Average measured step time: `0.6605s`.

The U=50M no-WD AR prefix is complete:

| Run | Best AR eval | Best step | Best epoch | Final AR eval | Final step | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `u50mish_p030_ar_allwd0` | **3.245539** | 2000 | 8.14 | 3.311268 | 2350 | Best is before handoff; router entropy is low late |

MoE health details for this AR prefix:

- Max eval-row drop: `0.125220`.
- Drop at best row: `0.019080`.
- Router entropy at best row: `0.769534`.
- Final eval-row drop: `0.030147`.
- Final router entropy: `0.746944`.

This strengthens the pattern seen at U=25M: no-WD AR can keep improving for
several epochs on U=50M, but by the fixed 30% handoff the checkpoint is worse
than its best point and the router is relatively low-entropy. The matching BD3
continuation will show how much BD3 can recover from that handoff.

The U=50M no-WD AR -> BD3 curriculum control is complete:

| Run | Best eval | Best step | Best epoch | Final eval | Final step | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `u50mish_p030_bd3_b4_allwd0` | **3.108332** | 6550 | 26.67 | 3.176530 | 7850 | Recovers from AR handoff but remains far behind scratch |

MoE health details for the BD3 continuation:

- Max eval-row drop: `0.073507`.
- Drop at best row: `0.0`.
- Router entropy at best row: `0.976418`.
- Final eval-row drop: `0.011406`.
- Final router entropy: `0.941670`.

Compared with U=50M scratch (`2.950898` best), the no-WD curriculum is behind
by about `0.1574` loss even after a full BD3 continuation. This is the control
that the tuned AR-WD curriculum must beat.

The U=50M tuned split-WD AR prefix is complete:

| Run | Best AR eval | Best step | Best epoch | Final AR eval | Final step | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `u50mish_p030_ar_muw0p1_adamwd0p001_routerwd0` | **3.290012** | 2000 | 8.14 | 3.340705 | 2350 | Worse AR loss than no-WD, but much healthier router at handoff |

MoE health details for this AR prefix:

- Max eval-row drop: `0.125220`.
- Drop at best row: `0.000666`.
- Router entropy at best row: `0.985963`.
- Final eval-row drop: `0.000386`.
- Final router entropy: `1.000642`.

This is a tradeoff rather than a clear AR-phase win. Compared with no-WD AR,
the tuned split-WD run has worse AR eval (`3.340705` final versus `3.311268`)
but much healthier final routing (`1.000642` entropy versus `0.746944`, and
near-zero drops). The BD3 continuation will test whether BD3 benefits more
from the healthier handoff than it loses from the worse AR objective value.

The U=50M tuned split-WD AR -> BD3 no-WD curriculum is complete:

| Run | Best eval | Best step | Best epoch | Final eval | Final step | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `u50mish_p030_bd3_b4_from_ar_muw0p1_adamwd0p001_routerwd0_bd3wd0` | **2.960323** | 6550 | 26.67 | 3.045280 | 7850 | Nearly matches scratch; clearly beats no-WD curriculum |

MoE health details for the tuned BD3 continuation:

- Max eval-row drop: `0.014561`.
- Drop at best row: `0.0`.
- Router entropy at best row: `1.162017`.
- Final eval-row drop: `0.0`.
- Final router entropy: `1.152458`.

U=50M summary:

| Method | Best eval | Gap vs scratch | Final eval | Comment |
| --- | ---: | ---: | ---: | --- |
| Scratch BD3 no-WD | **2.950898** | 0.000000 | 2.957172 | Best overall |
| AR no-WD -> BD3 no-WD | 3.108332 | +0.157434 | 3.176530 | Recovers but remains far behind |
| AR split-WD -> BD3 no-WD | 2.960323 | +0.009425 | 3.045280 | Effectively close to scratch; large improvement over no-WD curriculum |

Interpretation:

- The U=50M result supports the hypothesis that AR-phase regularization can
  make the curriculum competitive with scratch, but the evidence is not a
  strict scratch win.
- The split-WD AR phase had worse AR eval than no-WD AR, but it preserved
  router entropy and almost eliminated drops at the handoff.
- BD3 benefited strongly from that healthier handoff: the tuned continuation
  beat the no-WD curriculum by about `0.1480` best-eval loss and reached within
  `0.0095` of scratch.
- This suggests the useful target is not simply minimizing AR validation loss.
  Handoff health, especially router state and drop behavior, matters for BD3
  adaptation.

No U=100M runs were launched.

## Active U=50M Lighter-WD Tuning

The first U=50M split-WD setting reused the best U=25M probe
(`muon_wd=0.1`, `adam_wd=0.001`, `router_adam_wd=0`). It nearly matched
scratch after BD3 continuation, but its AR validation loss was worse than the
no-WD AR prefix. This suggests the setting may be over-regularized for U=50M.

Current lighter-WD sweep:

| Run | AR WD setting | Status | Interim note |
| --- | --- | --- | --- |
| `u50mish_p030_ar_muw0p05_adamwd0p0003_routerwd0` | `muon_wd=0.05`, `adam_wd=0.0003`, `router_adam_wd=0` | complete | Best `3.316068` at step 2000 / epoch 8.14; final `3.452593` at step 2350 / epoch 9.57; final drop `0.073667`, final entropy `0.689665` |
| `u50mish_p030_ar_muw0p05_adamwd0p0001_routerwd0` | `muon_wd=0.05`, `adam_wd=0.0001`, `router_adam_wd=0` | pending | Tests whether Adam WD should be almost off |
| `u50mish_p030_ar_muw0p025_adamwd0p0003_routerwd0` | `muon_wd=0.025`, `adam_wd=0.0003`, `router_adam_wd=0` | pending | Tests whether Muon WD should be lighter |

The screening criterion is not AR loss alone. A useful candidate should keep AR
loss close to no-WD AR while preserving enough router health for BD3 handoff.
After the AR prefixes finish, continue the best candidate with all BD3 WD
disabled and compare against the U=50M scratch best eval `2.950898`.

The first lighter-WD result confirms the tradeoff, but not in a useful
direction. It did not beat the heavier split-WD AR loss (`3.316068` best vs
`3.290012`), and by the handoff it lost most of the router-health benefit:
entropy fell to `0.689665` and drops rose to `0.073667`. This looks closer to
the no-WD failure mode than the heavier split-WD handoff.
