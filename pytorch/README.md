# Sample Efficient GPT

Training framework for sample-efficient language model pre-training. This codebase was used to train **IMU-1**, a 430M-parameter model that approaches the benchmark performance of models trained on 56x more data.

**Model:** [thepowerfuldeez/imu1_base](https://huggingface.co/thepowerfuldeez/imu1_base)

## Features

- **Architectural interventions:** QK-norm attention, per-head gating, value residual learning, LayerNorm scaling
- **Optimization:** NorMuon optimizer with cautious weight decay, muP parametrization
- **Custom kernels:** Triton Flash Attention with QK-norm, fused cross-entropy
- **Training:** Multi-stage WSD schedule, checkpoint EMA, gradient checkpointing
- **Tokenization:** Fast Rust-based BPE with HuggingFace conversion

## Requirements

- Python 3.13+
- PyTorch nightly (CUDA 13)
- NVIDIA GPU with CUDA support

## Quick Start

```bash
# Install dependencies (uses PyTorch nightly with CUDA 13)
export UV_TORCH_BACKEND=auto
uv pip install setuptools uv_build maturin
uv sync

# Login to services
uv run wandb login
huggingface-cli login
```

## Reproducing IMU-1

Pre-tokenized datasets and reproduction scripts are provided in `recipes/`.

### 1. Download data

```bash
# Stage 1 & 2 (pre-tokenized)
huggingface-cli download thepowerfuldeez/1218_imu1_base_stable_corpus --repo-type=dataset
huggingface-cli download thepowerfuldeez/1226_imu1_base_decay_corpus --repo-type=dataset

# Stage 3: tokenize with the provided script (see recipes/1_tokenize_stage3.sh)
```

### 2. Train

```bash
# Three-stage training (265k iterations, ~72B tokens)
NUM_GPUS=8 ./recipes/2_train.sh
```

Or run stages individually:

```bash
# Stage 1: Stable (100k iterations)
uv run torchrun --nproc_per_node 8 train.py --config configs/imu1_base.yaml --config-key stable

# Stage 2: Decay (100k iterations)
uv run torchrun --nproc_per_node 8 train.py --config configs/imu1_base.yaml --config-key decay

# Stage 3: Midtrain (65k iterations)
uv run torchrun --nproc_per_node 8 train.py --config configs/imu1_base.yaml --config-key midtrain
```

### 3. Convert to HuggingFace

```bash
# Convert checkpoint to HuggingFace format
uv run scripts/hf/convert_imu1_checkpoint.py \
    --checkpoint checkpoints/imu1_stage3_midtrain/265000.pt \
    --output-dir imu1_hf \
    --tokenizer HuggingFaceTB/SmolLM2-360M
```

## Evaluation

Evaluate models on the CORE benchmark (HellaSwag, ARC, PIQA, Lambada, Winograd, etc.).

### Setup

Download evaluation assets first:

```bash
bash download_eval.sh
```

This downloads the eval bundle to `~/.cache/sample_efficient_gpt/eval_bundle/`.

### Evaluate sample_efficient_gpt checkpoint

```bash
uv run torchrun --nproc_per_node 2 evals/base_eval.py \
    --checkpoint /path/to/checkpoint.pt \
    --type se \
    --tokenizer_path HuggingFaceTB/SmolLM2-360M \
    --batch 1 \
    --base_dir ~/.cache/sample_efficient_gpt \
    --task arc_easy
```

### Evaluate HuggingFace model

```bash
uv run torchrun --nproc_per_node 2 evals/base_eval.py \
    --checkpoint thepowerfuldeez/imu1_base \
    --type hf \
    --batch 1 \
    --base_dir ~/.cache/sample_efficient_gpt \
    --task arc_easy
```

### Evaluation arguments

| Argument | Description |
|----------|-------------|
| `--checkpoint` | Path to `.pt` file (se) or HF model name (hf) |
| `--type` | `se` for sample_efficient_gpt, `hf` for HuggingFace |
| `--tokenizer_path` | Tokenizer (required for `--type se`) |
| `--task` | Specific task (e.g., `arc_easy`, `hellaswag`, `piqa`) or omit for full CORE |
| `--batch` | Per-GPU batch size |
| `--base_dir` | Directory containing eval_bundle |
| `--max_per_task` | Limit examples per task for debugging |

### Available tasks

`hellaswag`, `jeopardy`, `bigbench_qa_wikidata`, `arc_easy`, `arc_challenge`, `copa`, `commonsenseqa`, `piqa`, `openbookqa`, `lambada_openai`, `winograd`, `winogrande`, `bigbench_dyck_languages`, `agieval_lsat_ar`, `bigbench_cs_algorithms`, `bigbench_operators`, `bigbench_repeat_copy_logic`, `squad`, `coqa`, `boolq`, `bigbench_language_id`

## Running HuggingFace Models

### Simple inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("thepowerfuldeez/imu1_base", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("thepowerfuldeez/imu1_base")

text = "The quick brown fox"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### Using the run script

```bash
uv run scripts/hf/run_imu1_hf.py \
    --model-dir thepowerfuldeez/imu1_base \
    --tokenizer thepowerfuldeez/imu1_base \
    --max-new-tokens 50 \
    --prompt "The capital of France is"
```

## Conversion Scripts

### Convert sample_efficient_gpt → HuggingFace

```bash
uv run scripts/hf/convert_imu1_checkpoint.py \
    --checkpoint /path/to/checkpoint.pt \
    --output-dir output_hf \
    --tokenizer HuggingFaceTB/SmolLM2-360M
```

### Convert HuggingFace → sample_efficient_gpt

```bash
uv run scripts/hf/convert_hf_to_imu.py \
    --hf-dir HuggingFaceTB/SmolLM2-360M \
    --output checkpoint.pt
```

### Initialize from SmolLM2 with widening

```bash
uv run scripts/hf/convert_smollm2_to_imu.py \
    --hf-dir HuggingFaceTB/SmolLM2-360M \
    --output checkpoint.pt \
    --width 1152 \
    --n-heads 18 \
    --n-kv-heads 6 \
    --widening-mode preserve-norm
```

## Tokenization

Tokenize datasets for training:

```bash
uv run tokenizer/tokenize_with_fast_tokenizer.py \
    --data-path /path/to/data \
    --tokenized-data-path /path/to/output \
    --tokenizer-name HuggingFaceTB/SmolLM2-360M
```

Input formats: parquet files or text files. Output: `.npy` (text) or `.npz` (parquet) memory-mapped arrays.

## Training

```bash
uv run train.py --config CONFIG --train-path TRAIN.npy --validation-path VAL.npy
```

Override config values:

```bash
uv run train.py --config gpt_small_faster --override '{"model.d_model": 1024, "optim.lr": 1e-3}'
```

## Configuration

Configs use frozen dataclasses (`config_schema.py`):

- **DataConfig:** batch size, context length, tokenizer, data paths
- **ModelConfig:** architecture (d_model, layers, heads, attention variants)
- **OptimConfig:** learning rates, optimizer (NorMuon/AdamW), schedules (cosine/WSD)
- **TrainerConfig:** checkpointing, distributed training, logging

See `configs/imu1_base.yaml` for the full IMU-1 configuration.

## Project Structure

```
sample_efficient_gpt/
├── train.py                 # Main training entry point
├── config_schema.py         # Configuration dataclasses
├── configs/                 # Training configurations
│   ├── imu1_base.yaml       # IMU-1 (430M) config
│   └── gpt_small_faster.py  # Ablation model (70M) config
├── recipes/                 # Reproduction scripts
├── transformer/             # Model architecture
│   ├── transformer.py       # Transformer with KV-cache
│   ├── attention.py         # Multi-head attention (QK-norm, gating, value residual)
│   ├── core.py              # Primitives (Linear, RMSNorm, SwiGLU)
│   └── ops/                 # Triton kernels
├── training/                # Training infrastructure
│   ├── trainer.py           # Training loop
│   ├── data.py              # Memory-mapped dataset
│   └── optimizers/          # NorMuon, Muon implementations
├── tokenizer/               # Tokenization pipeline
├── scripts/hf/              # HuggingFace conversion
│   ├── convert_imu1_checkpoint.py  # SE → HF conversion
│   ├── convert_hf_to_imu.py        # HF → SE conversion
│   ├── convert_smollm2_to_imu.py   # SmolLM2 init with widening
│   └── run_imu1_hf.py              # Run HF model
└── evals/                   # Benchmark evaluation
    ├── base_eval.py         # CORE benchmark (supports SE and HF)
    └── core_eval.py         # Evaluation logic
```

## Data Mix Files

Training uses JSON files specifying dataset paths and token counts:

- `data_stage1_stable.json` - Stage 1 data mix
- `data_stage2_decay.json` - Stage 2 data mix
- `data_stage3_midtrain.json` - Stage 3 data mix

Format: `{"path/to/tokenized/data": num_tokens, ...}`

## References

- [Gated Attention](https://arxiv.org/abs/2505.06708)
- [Value Residual Learning](https://arxiv.org/abs/2410.17897)
- [LayerNorm Scaling](https://arxiv.org/abs/2502.05795)
- [muP Parametrization](https://arxiv.org/abs/2505.02222)
- [Cautious Weight Decay](https://arxiv.org/abs/2510.12402)
- [Z-loss](https://arxiv.org/abs/2204.02311)

## License

MIT
