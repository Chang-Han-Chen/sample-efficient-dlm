# CODEX Overnight Report

## Sources and Hardware

- `PROGRESS.md`: dense AR/MDLM baseline and dense intervention anchors on
  `2xA100`.
- `codex_overnight.md`: AR MoE ablations and full runs on `2xH100`.
- `codex_overnight_v2.md`: 4-GPU timing notes and MDLM MoE full runs on
  `4xA100`.
- W&B was checked for the completed dense baselines:
  - `ar_baseline_lr0p8`, run id `mbbnqirk`
  - `ar_old_bundle`, run id `8fjqg6b9`
  - `mdlm_baseline_lr0p8`, run id `wktduuof`
  - `mdlm_old_bundle_lr0p8`, run id `ha7vvj3w`

## Executive Summary

Best AR config:

- Run: `ar_moe_old_bundle_nonsplit_lr2p0_zloss0p01_ckpt1k`
- Config: `configs/experiments/ar_moe_old_bundle.yaml`
- Key overrides: `--lr-mult 2.0`, `--no-moe-split-router-input`,
  `--moe-router-z-loss-weight 0.01`, local checkpoints every `1000` steps.
- Architecture: AR objective, old-bundle flags on (`attn_qknorm=true`,
  `attn_val_residual=true`, `attn_gating=per-head`,
  `layernorm_scaling=true`), no value embeddings, MoE enabled on alternating
  layers with `4` top-1 experts and capacity factor `1.25`.
- Loss: best `2.752304 @4450`, final/last `2.779193 @5050`.

Best MDLM config:

- Run: `mdlm_moe_old_bundle_arstyle_lr2p0_rz0p01_24x_4a100`
- Config: `configs/experiments/mdlm_moe_old_bundle.yaml`
- Key overrides: `--attn-val-residual`, `--no-moe-split-router-input`,
  `--lr-mult 2.0`, `--moe-router-z-loss-weight 0.01`,
  `--checkpoint-interval 1000`, no W&B checkpoint artifacts.
- Architecture: MDLM objective, AR-style old-bundle flags on
  (`attn_qknorm=true`, `attn_val_residual=true`, `attn_gating=per-head`,
  `layernorm_scaling=true`), no value embeddings, MoE enabled on alternating
  layers with `4` top-1 experts and capacity factor `1.25`.
- Loss: best/final `3.077142 @5050`.

## Baseline Comparisons

Lower eval loss is better. Percent improvement is
`(baseline_loss - intervention_loss) / baseline_loss`.

### AR

| Run | Hardware | Best eval | Final eval | Notes |
|---|---:|---:|---:|---|
| dense baseline `ar_baseline_lr0p8` | `2xA100` | `2.852894 @4450` | `2.884064 @5050` | dense, old flags off |
| dense old-bundle `ar_old_bundle` | `2xA100` | `2.768570 @4450` | `2.799982 @5050` | QK-norm, value residual, per-head gate, LN scaling |
| best AR MoE old-bundle | `2xH100` | `2.752304 @4450` | `2.779193 @5050` | old-bundle MoE, non-split router, z-loss `0.01` |

Best AR MoE versus dense baseline:

- Best eval delta: `-0.100590`, or `3.53%` improvement.
- Final eval delta: `-0.104872`, or `3.64%` improvement.

Best AR MoE versus dense old-bundle:

- Best eval delta: `-0.016266`, or `0.59%` improvement.
- Final eval delta: `-0.020789`, or `0.74%` improvement.

### MDLM

| Run | Hardware | Best eval | Final eval | Notes |
|---|---:|---:|---:|---|
| dense baseline `mdlm_baseline_lr0p8` | `2xA100` | `3.176954 @4450` | `3.195939 @5050` | dense, old flags off |
| dense old-bundle `mdlm_old_bundle_lr0p8` | `2xA100` | `3.084283 @4850` | `3.093443 @5050` | QK-norm, value residual, per-head gate, LN scaling |
| best MDLM MoE old-bundle | `4xA100` | `3.077142 @5050` | `3.077142 @5050` | old-bundle MoE, non-split router, z-loss `0.01` |

Best MDLM MoE versus dense baseline:

- Best eval delta: `-0.099812`, or `3.14%` improvement.
- Final eval delta: `-0.118797`, or `3.72%` improvement.

Best MDLM MoE versus dense old-bundle:

- Best eval delta: `-0.007141`, or `0.23%` improvement.
- Final eval delta: `-0.016301`, or `0.53%` improvement.

## Wall Time by Hardware

These timings are not a single controlled hardware sweep; they are grouped by
where the runs were actually logged. Use within-hardware comparisons for speed
claims.

### `2xA100` Dense Runs from `PROGRESS.md` and W&B

| Run | Objective | Step time | Throughput | Notes |
|---|---|---:|---:|---|
| `ar_baseline_lr0p8` | AR dense baseline | `0.4401s` | `595.6k tok/s` | completed 5.1K W&B run |
| `ar_old_bundle` | AR dense old-bundle | `0.4856s` | `539.9k tok/s` | completed 5.1K W&B run |
| `ar_value_embedding_no_vr_nogain_lr2p0_5k1_from500` | AR dense VE/no-VR | `0.6258s` | `418.9k tok/s` | resumed from stable 500-step checkpoint |
| `mdlm_baseline_lr0p8` | MDLM dense baseline | `1.2120s` | `216.3k tok/s` | completed 5.1K W&B run |
| `mdlm_old_bundle_lr0p8` | MDLM dense old-bundle | `1.0975s` | `238.9k tok/s` | completed 5.1K W&B run |
| `mdlm_value_embedding_no_vr_nogain_lr0p8_maskadam0p005_5k1` | MDLM dense VE split-mask | `1.1579s` | `226.4k tok/s` | completed 5.1K W&B run |

### `2xH100` AR MoE Runs from `codex_overnight.md`

| Run | Objective | Step time | Throughput | Notes |
|---|---|---:|---:|---|
| `ar_baseline_h100_timing_100` | AR dense timing-only | `0.1782s` | `1.47M tok/s` | 100-step timing run, W&B disabled |
| `ar_moe_baseline_lr0p8` | AR plain MoE | `0.2370s` | `1.105M tok/s` | completed 5.1K run |
| `ar_moe_old_bundle_lr2p0_full_nockpt` | AR old-bundle MoE, z-loss `0.001` | `0.2649s` | not logged | completed 5.1K run |
| `ar_moe_old_bundle_nonsplit_lr2p0_zloss0p01_ckpt1k` | AR best old-bundle MoE | `0.2664s` | not logged | checkpoint saves included in mean |
| AR MoE+VE probes | AR MoE+VE | about `0.34s` | not logged | slower and not competitive |

Same-machine AR timing conclusion on `2xH100`:

- Dense AR timing: `0.1782s/step`.
- Plain AR MoE: `0.2370s/step`, about `1.33x` slower.
- Best AR old-bundle MoE: `0.2664s/step`, about `1.50x` slower than dense
  timing and about `1.12x` slower than plain MoE.

### `4xA100` Runs from `codex_overnight_v2.md`

| Run | Objective | Step time | Throughput | Peak HBM | Notes |
|---|---|---:|---:|---:|---|
| `ar_baseline_100_profile_4gpu` | AR dense timing/profile | `0.2417s` | `1.085M tok/s` | `19.03 GB` | 100-step timing/profile run |
| `ar_moe_old_bundle_lr2p0_ckpt_resume_test_part1` | AR old-bundle MoE timing | `0.3260s` | `804k tok/s` | `27.44 GB` | 100-step checkpoint/resume test |
| `mdlm_moe_baseline_lr0p8_24x_4a100` | MDLM plain MoE | `0.7826s` | `334,957 tok/s` | `22.49 GB` | completed 5.1K run |
| `mdlm_moe_old_bundle_arstyle_lr2p0_rz0p01_24x_4a100` | MDLM best old-bundle MoE | `0.7310s` | `358,589 tok/s` | `27.48 GB` | completed 5.1K run |

Same-machine `4xA100` timing conclusions:

- AR old-bundle MoE was `1.35x` slower than dense AR timing
  (`0.3260 / 0.2417`), with throughput ratio `0.74x`.
- MDLM old-bundle MoE was faster than MDLM plain MoE:
  `0.7310s` versus `0.7826s`, about `6.6%` lower step time and about `7.1%`
  higher throughput, while using about `5 GB` more peak HBM.

## Consolidated Ablation Ledger

### Dense AR Baseline and Dense Interventions

- `ar_baseline_lr0p8` finished 5100 steps. Best eval was
  `2.852894 @4450`; final eval was `2.884064 @5050`. This is the tuned dense
  baseline; `lr_mult=1.0` destabilized late, so `lr_mult=0.8` is the stable
  dense baseline.
- `ar_old_bundle` with `lr_mult=5.0` finished 5100 steps. Best eval was
  `2.768570 @4450`; final eval was `2.799982 @5050`. This was the best dense
  AR intervention before MoE.
- `ar_value_embedding_no_vr_nogain_lr2p0_5k1_from500` finished from a stable
  500-step checkpoint. Best eval was `2.8480 @5000`; final eval was
  `2.8895 @5050`. It was competitive with dense baseline by best eval, but not
  better by final eval and was much slower.

### Dense MDLM Baseline and Dense Interventions

- `mdlm_baseline_lr0p8` finished 5100 steps. Best eval was
  `3.176954 @4450`; final eval was `3.195939 @5050`.
- `mdlm_old_bundle_lr0p8` finished 5100 steps. Best eval was
  `3.084283 @4850`; final eval was `3.093443 @5050`. This is the strongest
  dense MDLM intervention anchor.
- `mdlm_value_embedding_no_vr_nogain_lr0p8_maskadam0p005_5k1` finished 5100
  steps. Best eval was `3.131494 @4500`; final eval was `3.131513 @5050`.
- `mdlm_value_embedding_no_vr_nogain_lr0p8_init0p01_nomaskvec_5k1` finished
  5100 steps. Best/final eval was `3.103966 @5050`.

### AR Plain MoE Baseline and Router LR Probes

- `ar_moe_baseline_lr0p8` finished 5100 steps. Best eval was
  `2.806168 @4800`; final eval was `2.838030 @5050`. This beat the dense AR
  baseline but was slower than dense on the same H100 timing comparison.
- Higher global LR probes were stable but worse:
  - `lr_mult=1.0`: matched eval around step `950` was `3.2349` versus baseline
    `3.1526`.
  - `lr_mult=1.25`: last eval `3.4597` by step `600`.
  - `lr_mult=1.5`: clearly worse by step `475`.
- Router LR changes did not clearly improve the plain MoE baseline:
  - At `lr_mult=1.0`, router base LR `0.0005` and `0.002` both reached about
    `3.356` by step `575`.
  - At `lr_mult=0.8`, router base LR `0.0005`, `0.002`, and `0.005` were close
    through `600` steps.
  - The longer router base LR `0.005` probe improved eval by only about
    `0.001` after step `600`, while stressing routing more.

### AR Old-Bundle MoE

- Old-bundle means QK-norm, value residual, per-head attention gating,
  layernorm scaling, no value embeddings.
- Short probes showed the old-bundle MoE recipe wanted a hotter LR than plain
  MoE:
  - `lr_mult=0.6`: `3.2961 @950`, worse than plain MoE, with routing shocks.
  - `lr_mult=0.8`: `3.3672 @950`, worse, with entropy collapse.
  - `lr_mult=1.0`: `3.1894 @950`, close but still worse than plain MoE.
  - `lr_mult=2.0`: `3.0599 @950`, clearly better than plain MoE at matched
    step.
  - `lr_mult=3.0`: unstable and collapsed.
- No-layernorm-scaling old-bundle tied the layernorm-scaling version at
  `lr_mult=2.0` in 1K probes:
  - no-LN-scale `lr_mult=2.0`: `3.0621 @950`
  - LN-scale `lr_mult=2.0`: `3.0599 @950`
- Full old-bundle MoE with z-loss `0.001`:
  - Run: `ar_moe_old_bundle_lr2p0_full_nockpt`
  - Best eval `2.756363 @4800`; final eval `2.800013 @5050`.
  - Better than plain AR MoE by about `0.05` best eval.
- Best full old-bundle MoE with stronger z-loss:
  - Run: `ar_moe_old_bundle_nonsplit_lr2p0_zloss0p01_ckpt1k`
  - Best eval `2.752304 @4450`; final eval `2.779193 @5050`.
  - Tail routing was clean: tail-from-5000 mean drop about `0.00027`, router
    entropy about `0.978`.
  - Verdict: push forward non-split old-bundle MoE with `lr_mult=2.0` and
    router z-loss `0.01`.

### AR Split-Router Experiments

- Split-router architecture routed on unscaled `RMSNorm(x)` while experts used
  the layernorm-scaled input.
- 1K split-router probe looked strong:
  - `ar_moe_old_bundle_splitrouter_lr2p0_probe_1k`: `3.0396 @950`.
- Full split-router with the original z-loss collapsed after step `3200`:
  - best eval before collapse `2.8282 @3200`;
  - collapse region reached eval `4.7616 @3300` and `6.5181 @3500`.
- Lowering router LR alone did not fix split-router:
  - `ar_moe_old_bundle_splitrouter_lr2p0_routerhalf_full_nockpt` was stopped as
    noncompetitive; best eval `2.9983 @2400`.
- Stronger router z-loss stabilized split-router:
  - `ar_moe_old_bundle_splitrouter_lr2p0_zloss0p01_full_nockpt`
  - Best eval `2.762089 @4450`; final eval `2.794618 @5050`.
  - Stable and clean, but still slightly behind the simpler non-split z-loss
    `0.01` run.

### AR MoE Value-Embedding Probes

- VE+MoE used QK-norm, per-head attention gating, layernorm scaling, value
  embeddings on alternating layers, no value residual, no trainable VE gain,
  and the best old-bundle MoE settings available at the time.
- Results:
  - `lr_mult=2.0`: best `3.331774 @550`, final `3.345241 @950`; stressed
    routing and was much worse than plain MoE.
  - `lr_mult=1.0`: best/final `3.261778 @950`; better than `2.0` but still
    worse than plain MoE at matched step.
  - `lr_mult=0.8`: best/final `3.4200 @950`; too slow and had a bad mid-run
    event.
  - VE table LR `0.005`: best `3.218391 @900`, then late shock to
    `3.818626 @950`.
  - Capacity factor `2.0`: removed the late drop shock but was slower and worse
    on loss.
  - Gain ramp from zero did not rescue VE+MoE.
- Verdict: do not push VE+MoE as the main AR path. Treat VE+MoE as a mechanism
  debugging project if revisited.

### MDLM MoE

- Plain MDLM MoE baseline:
  - Run: `mdlm_moe_baseline_lr0p8_24x_4a100`
  - Best/final eval `3.152543 @5050`.
  - Routing remained healthy enough: tail entropy about `1.079`, tail drop mean
    about `1.58%`.
- MDLM old-bundle AR-style MoE:
  - Run: `mdlm_moe_old_bundle_arstyle_lr2p0_rz0p01_24x_4a100`
  - Best/final eval `3.077142 @5050`.
  - Tail routing was very clean: tail drop mean `0.0043%`, entropy about
    `1.200`.
  - It improved over MDLM plain MoE by `0.075402`, about `2.39%`, and also ran
    faster on the same `4xA100` setup.
  - It also beat the best dense MDLM old-bundle anchor narrowly.

### MDLM MoE Value-Embedding Diagnostic

- Run: `mdlm_moe_ve_nomaskvec_nonsplit_z0p01_lr1p0_vetbl0p005_probe_1k`
- Config: MDLM objective, MoE + value embeddings, no value residual, no VE
  gain, no mask-vector value embedding, non-split router input, router z-loss
  `0.01`, VE table LR `0.005`.
- Result: best `3.703264 @800`, final `3.890115 @950`, with recurring
  capacity/drop events.
- Verdict: not promising enough to justify a full MDLM MoE+VE no-mask-vector
  run with this recipe.

## Current Recommendations

- Main AR path: `ar_moe_old_bundle_nonsplit_lr2p0_zloss0p01_ckpt1k`.
- Main MDLM path: `mdlm_moe_old_bundle_arstyle_lr2p0_rz0p01_24x_4a100`.
- Do not spend mainline time on AR or MDLM VE+MoE without a specific mechanism
  hypothesis.
- If tuning MoE further, the highest-signal knobs are router z-loss,
  router-temperature/gating controls, and fixed-architecture ablations to
  separate the effects of old-bundle flags, LR, and router z-loss.
