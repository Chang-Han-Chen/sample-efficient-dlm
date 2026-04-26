# JAX Configs

The experiment YAMLs under `experiments/` are the source of truth for the AR
ablation matrix. `train_ar.py` reads the `train_args` mapping directly:

```bash
python train_ar.py --config configs/experiments/ar_baseline.yaml
```

For LR sweeps, keep the config fixed and override only `--lr-mult` and the log
path:

```bash
python train_ar.py \
  --config configs/experiments/ar_baseline.yaml \
  --lr-mult 0.5 \
  --log-jsonl runs/ar_matrix/ar_baseline/lr0p5/train.jsonl
```

`base_configs` is metadata for humans and validation tests; the trainer does
not currently merge those files into `train_args`.

When `output_dir` is set, the trainer writes `resolved_config.json` there before
the first step.
