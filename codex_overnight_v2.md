# CODEX_OVERNIGHT_V2

## 2026-04-26 Timing Comparison Notes

### Dense AR: 4xA100 vs 2xA100

- Current dense AR timing run:
  - Run: `runs/ar_matrix/ar_baseline_100_profile_4gpu/train.jsonl`
  - Hardware: `4xA100`
  - Global batch: `512`
  - Per-GPU batch: `128`
  - Sequence length: `512`
  - Grad accumulation: `1`
  - Average measured step from step 5: `0.2417s`
  - Throughput: `1.085M tokens/s`
  - JAX peak HBM: `19.03 GB`

- Older dense AR timing from `PROGRESS.md`:
  - Run: `runs/ar_matrix/ar_baseline_phase0_300/train.jsonl`
  - Hardware: `2x A100-SXM4-80GB`
  - Global batch: `512`
  - Per-GPU batch: `256`
  - Sequence length: `512`
  - Grad accumulation: `1`
  - Average measured step: `0.4290s`
  - Throughput: `611k tokens/s`
  - JAX peak HBM: `36.53 GB`

- Scaling result:
  - Wall-clock speedup: `0.4290 / 0.2417 = 1.77x`
  - Throughput speedup: `1.085M / 611k = 1.78x`

This is a reasonable result for moving from 2 to 4 A100s at the same global
batch size. It is below ideal 2x scaling because the per-GPU batch drops from
`256` to `128` and data-parallel communication overhead increases.

### MoE Old-Bundle vs Dense on Current 4xA100

- Current dense AR 4xA100:
  - Average measured step: `0.2417s`
  - Throughput: `1.085M tokens/s`
  - JAX peak HBM: `19.03 GB`

- Current AR MoE old-bundle 4xA100:
  - Run: `runs/moe_matrix/ar_moe_old_bundle_lr2p0_ckpt_resume_test_part1/train.jsonl`
  - Config: `configs/experiments/ar_moe_old_bundle.yaml`
  - Override: `--lr-mult 2.0`
  - Global batch: `512`
  - Per-GPU batch: `128`
  - Sequence length: `512`
  - Grad accumulation: `1`
  - Average measured step from step 5: `0.3260s`
  - Throughput: `804k tokens/s`
  - JAX peak HBM: `27.44 GB`

- Dense vs MoE result on the same 4xA100 setup:
  - MoE step-time slowdown: `0.3260 / 0.2417 = 1.35x`
  - MoE throughput ratio: `804k / 1.085M = 0.74x`
  - MoE peak HBM increase: about `8.4 GB`

The MoE slowdown is expected: top-1 routing, token dispatch/scatter, expert
matmuls, auxiliary metrics, and additional communication pressure all add wall
time beyond the dense transformer path.

### Current 4xA100 MoE vs Previous 2xH100 MoE

- `CODEX_OVERNIGHT.md` logged previous H100 MoE timings:
  - Plain AR MoE 2xH100: `0.2370s/step`
  - Old-bundle AR MoE 2xH100: mean `0.2649s`, median `0.2558s`

- Current old-bundle AR MoE 4xA100:
  - Average measured step: `0.3260s`

- Comparison:
  - `0.3260 / 0.2649 = 1.23x` slower than the previous 2xH100 old-bundle mean
  - `0.3260 / 0.2558 = 1.27x` slower than the previous 2xH100 old-bundle median

This is not surprising. The current machine is A100, not H100, and A100 has
lower BF16 tensor throughput and memory bandwidth. The current 4-GPU run also
uses `128` samples/GPU versus `256` samples/GPU in the previous 2-GPU setup,
which can reduce per-GPU efficiency and increase the relative communication
cost.

## 2026-04-26/27 Run Ledger

### Data

- Prepared full local CLIMB mix shard set for these runs:
  - Train tokens: `data/climbmix_24x_newtok_8192/tokens/train`
  - Eval tokens: `data/climbmix_24x_newtok_8192/tokens/val`
  - Train shard count: `24`
  - Eval shard count: `1`
- The folder was renamed to `climbmix_24x_newtok_8192` so configs can be used
  with explicit path overrides instead of editing config files.

### Local and W&B Log Index

- Dense AR 100-step timing/profile run:
  - Local JSONL: `runs/ar_matrix/ar_baseline_100_profile_4gpu/train.jsonl`
  - Local output dir: `runs/ar_matrix/ar_baseline_100_profile_4gpu`
  - W&B: disabled

- AR MoE old-bundle 100-step checkpoint test:
  - Local JSONL:
    `runs/moe_matrix/ar_moe_old_bundle_lr2p0_ckpt_resume_test_part1/train.jsonl`
  - Local output dir:
    `runs/moe_matrix/ar_moe_old_bundle_lr2p0_ckpt_resume_test_part1`
  - Local final checkpoint:
    `runs/moe_matrix/ar_moe_old_bundle_lr2p0_ckpt_resume_test_part1/checkpoints/final`
  - Local resolved config:
    `runs/moe_matrix/ar_moe_old_bundle_lr2p0_ckpt_resume_test_part1/resolved_config.json`
  - W&B run id: `vj4ikbw4`
  - W&B local dir: `wandb/run-20260426_224309-vj4ikbw4`
  - W&B run URL:
    `https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/vj4ikbw4`

- MDLM MoE baseline full run:
  - Local JSONL:
    `runs/moe_matrix/mdlm_moe_baseline_lr0p8_24x_4a100/train.jsonl`
  - Local output dir:
    `runs/moe_matrix/mdlm_moe_baseline_lr0p8_24x_4a100`
  - Local resolved config:
    `runs/moe_matrix/mdlm_moe_baseline_lr0p8_24x_4a100/resolved_config.json`
  - Local best checkpoint:
    `runs/moe_matrix/mdlm_moe_baseline_lr0p8_24x_4a100/checkpoints/best`
  - Local final checkpoint:
    `runs/moe_matrix/mdlm_moe_baseline_lr0p8_24x_4a100/checkpoints/final`
  - W&B run id: `6sh2lavi`
  - W&B local dir: `wandb/run-20260426_225746-6sh2lavi`
  - W&B files:
    `wandb/run-20260426_225746-6sh2lavi/files/{config.yaml,output.log,wandb-summary.json}`
  - W&B run URL:
    `https://wandb.ai/y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm/runs/6sh2lavi`
  - Best and final checkpoints were uploaded to W&B as run artifacts.

## MDLM MoE Baseline Full Run

Command-equivalent config:

```bash
python train_ar.py \
  --config configs/experiments/mdlm_moe_baseline.yaml \
  --train-path data/climbmix_24x_newtok_8192/tokens/train \
  --eval-path data/climbmix_24x_newtok_8192/tokens/val \
  --num-devices 4 \
  --run-name mdlm_moe_baseline_lr0p8_24x_4a100 \
  --output-dir runs/moe_matrix/mdlm_moe_baseline_lr0p8_24x_4a100 \
  --log-jsonl runs/moe_matrix/mdlm_moe_baseline_lr0p8_24x_4a100/train.jsonl \
  --log-every 50 \
  --measure-start-step 105
```

Effective setup:

- Objective: `mdlm`
- Config file: `configs/experiments/mdlm_moe_baseline.yaml`
- Hardware: `4x A100-SXM4-80GB`
- Global batch: `512`
- Per-GPU batch: `128`
- Grad accumulation: `1`
- Context length: `512`
- Tokens per optimizer step: `262144`
- Steps: `5100` rows, train steps `0` through `5099`
- Eval: every `50` steps, `eval_batches=4`, `eval_t_frac=0.6`
- Diffusion: `diffusion_steps=100`, `t_min=0.45`, `t_max=0.95`,
  linear schedule, expected mask rate `0.7`
- Vocab: base `8192`, model `8193`, mask token id `8192`
- Backbone: `d_model=768`, `d_ff=2048`, `n_layers=8`, `n_heads=12`,
  `bfloat16`, `cudnn` attention, weight tying on
- Architecture flags: QK norm off, value residual off, attention gating off,
  layernorm scaling off, value embedding off
- MoE: enabled, token-choice switch, top-1, alternating layers, `4` experts,
  capacity factor `1.25`, router-prob scaling on, token dropping on,
  fp32 router
- MoE losses: load balance weight `0.01`, router z-loss weight `0.001`
- Optimizer: `lr_mult=0.8`; table LR `0.008`, scalar LR `0.004`,
  router LR `0.0008`, Muon LR `0.032`, Muon WD `1e-4`

Final outcome:

- Best eval: step `5050`, `eval_loss=3.152543`, `eval_z_loss=22.0140`
- Last eval: step `5050`, same as best
- Final train row: step `5099`, loss `3.6873`, total loss `3.6994`
- Final checkpoint saved at:
  `runs/moe_matrix/mdlm_moe_baseline_lr0p8_24x_4a100/checkpoints/final`
- Best checkpoint saved at:
  `runs/moe_matrix/mdlm_moe_baseline_lr0p8_24x_4a100/checkpoints/best`
- Timing/memory summary:
  - `compile_plus_first_step=34.103s`
  - `avg_measured_step=0.7826s` from step `105`
  - `tokens_per_sec=334,957`
  - `est_tflops=161.3`
  - `mfu=51.7%`
  - `jax_peak_hbm_gb=22.49`

MoE health at best/final eval step `5050`:

- `moe_aux_loss=0.01009`
- `moe_drop=0.00050`
- `moe_router_entropy=1.07536`
- `moe_load_balance_loss=1.00220`
- `moe_router_z_loss=0.06568`
- Expert assignment min/max: `0.1959` / `0.2918`
- Router probability min/max: `0.2341` / `0.2680`

Tail MoE health from steps `4500-5099`:

- Router entropy mean/min: `1.0793` / `1.0660`
- Drop mean/p95/max: `0.0158` / `0.0452` / `0.0860`
- Load-balance mean: `1.0038`
- Router z-loss mean: `0.0651`
- Grad max: `0.9548`
- Step-time mean: `0.7729s`

Interpretation:

- The run finished cleanly and kept improving late. Best eval and last eval
  matched at step `5050`.
- Routing sharpened gradually, with entropy falling from the early `1.18-1.20`
  band to about `1.08` late. This did not turn into collapse: drop was
  intermittent, load balance stayed close to `1.0`, and router probability
  fractions remained close to uniform.
- Router z-loss stayed small. With `moe_router_z_loss ~= 0.065` and weight
  `0.001`, it contributes only about `0.000065` to the loss. The MoE aux term
  is dominated by load balancing.

### MDLM Comparison Anchors

The dense references below are from `PROGRESS.md` and are the available local
matched-step table, not a complete 5.1K dense rerun on this machine.

| step | plain dense `mdlm_baseline_lr0p8` | dense old-bundle `mdlm_old_bundle_lr0p8` | current MoE baseline | MoE minus plain dense |
|---:|---:|---:|---:|---:|
| 500 | `3.8346` | `3.7159` | `3.8208` | `-0.0138` |
| 1000 | `3.5736` | `3.4652` | `3.5358` | `-0.0378` |
| 1500 | `3.4673` | `3.3582` | `3.4373` | `-0.0300` |
| 2000 | `3.4224` | `3.3133` | `3.3394` | `-0.0830` |
| 2500 | `3.3371` | `3.2119` | `3.3001` | `-0.0370` |
| 3000 | `3.3050` | `3.1899` | `3.2828` | `-0.0222` |

Additional late-run comparison:

- Current MoE best/final eval at step `5050`: `3.1525`
- Versus plain dense step `3000` reference `3.3050`: `-0.1525`
- Versus dense old-bundle step `3000` reference `3.1899`: `-0.0374`
- Caveat: the late comparison uses the latest dense anchors available in
  `PROGRESS.md`; it is not an exact matched-step comparison against a local
  dense 5.1K run.

Working conclusion:

- The MDLM MoE baseline is clearly ahead of the tuned plain dense MDLM baseline
  in matched early/mid training and in the late available comparison.
- It was behind the dense old-bundle reference at matched steps through 3K, but
  the late MoE run crossed below the old-bundle step-3K anchor by the end.
- No urgent LR or router-z tuning is indicated from this run. If future runs show
  sustained drop or entropy collapse, the first targeted knobs should be router
  LR and router z-loss, but this completed baseline does not require intervention.
