# JAX Configs

The experiment YAMLs under `experiments/` are the source of truth for the AR
ablation matrix. `train_ar.py` reads the `train_args` mapping directly:

```bash
python jax/train_ar.py --config jax/configs/experiments/ar_baseline.yaml
```

For LR sweeps, keep the config fixed and override only `--lr-mult` and the log
path:

```bash
python jax/train_ar.py \
  --config jax/configs/experiments/ar_baseline.yaml \
  --lr-mult 0.5 \
  --log-jsonl runs/ar_matrix/ar_baseline/lr0p5/train.jsonl
```

When `output_dir` is set, the trainer writes `resolved_config.json` there before
the first step.
