import subprocess
import sys
from pathlib import Path

import pytest
import torch

from backbone import DiffusionBackbone
from model_bd3lm import Model as BlockMDLM
from prepare import DATA_DIR, TOKENIZER_DIR


ROOT = Path(__file__).resolve().parent.parent


def _climbmix_available():
    tokenizer_path = Path(TOKENIZER_DIR) / "tokenizer.pkl"
    data_dir = Path(DATA_DIR)
    return tokenizer_path.is_file() and len(list(data_dir.glob("*.parquet"))) >= 2


requires_cuda_climbmix = pytest.mark.skipif(
    not torch.cuda.is_available() or not _climbmix_available(),
    reason="Requires CUDA with prepared ClimbMix data",
)


def run_repo_cmd(*args, timeout=120):
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_single_block_forward_train_ignores_clean_stream():
    torch.manual_seed(0)
    model = DiffusionBackbone(
        vocab_size=10,
        n_embd=8,
        n_head=2,
        n_layer=1,
        head_dim=4,
        block_size=4,
        block_len=4,
        dropout=0.0,
    )
    model.eval()

    xt = torch.tensor([[1, 2, 3, 4]])
    x0_a = torch.tensor([[5, 6, 7, 8]])
    x0_b = torch.tensor([[8, 7, 6, 5]])

    logits_a, _ = model.forward_train(xt, x0_a)
    logits_b, _ = model.forward_train(xt, x0_b)

    assert torch.equal(logits_a, logits_b)


@requires_cuda_climbmix
def test_block_train_cli_prints_sample_without_crashing(tmp_path):
    result = run_repo_cmd(
        "train.py",
        "--model",
        "bd3lm",
        "--data",
        "climbmix",
        "--max_iters",
        "1",
        "--eval_interval",
        "1",
        "--eval_iters",
        "1",
        "--batch_size",
        "2",
        "--block_size",
        "16",
        "--block_len",
        "4",
        "--n_embd",
        "16",
        "--n_head",
        "4",
        "--n_layer",
        "1",
        "--save_interval",
        "2",
        "--num_final_samples",
        "0",
        "--gpt2_eval_interval",
        "0",
        "--use_compile",
        "false",
        "--checkpoint_path",
        str(tmp_path / "block_ckpt.pt"),
        "--loss_log_path",
        str(tmp_path / "block_loss.pkl"),
    )

    assert result.returncode == 0, result.stderr
    assert "Generating sample..." in result.stdout
    assert "TypeError" not in result.stdout
    assert "TypeError" not in result.stderr


@requires_cuda_climbmix
def test_train_cli_accepts_save_interval_zero(tmp_path):
    result = run_repo_cmd(
        "train.py",
        "--model",
        "mdlm",
        "--data",
        "climbmix",
        "--max_iters",
        "2",
        "--eval_interval",
        "0",
        "--batch_size",
        "2",
        "--block_size",
        "16",
        "--n_embd",
        "16",
        "--n_head",
        "4",
        "--n_layer",
        "1",
        "--save_interval",
        "0",
        "--num_final_samples",
        "0",
        "--sample_interval",
        "0",
        "--gpt2_eval_interval",
        "0",
        "--use_compile",
        "false",
        "--checkpoint_path",
        str(tmp_path / "mdlm_ckpt.pt"),
        "--loss_log_path",
        str(tmp_path / "mdlm_loss.pkl"),
    )

    assert result.returncode == 0, result.stderr
    assert "ZeroDivisionError" not in result.stdout
    assert "ZeroDivisionError" not in result.stderr


@requires_cuda_climbmix
def test_train_cli_can_warmstart_block_from_ar_checkpoint(tmp_path):
    ar_ckpt = tmp_path / "ar_ckpt.pt"
    ar_loss = tmp_path / "ar_loss.pkl"
    bd_ckpt = tmp_path / "bd_ckpt.pt"
    bd_loss = tmp_path / "bd_loss.pkl"

    ar_result = run_repo_cmd(
        "train.py",
        "--model",
        "ar",
        "--data",
        "climbmix",
        "--max_iters",
        "1",
        "--eval_interval",
        "0",
        "--batch_size",
        "2",
        "--block_size",
        "16",
        "--n_embd",
        "16",
        "--n_head",
        "4",
        "--n_layer",
        "1",
        "--save_interval",
        "0",
        "--num_final_samples",
        "0",
        "--sample_interval",
        "0",
        "--gpt2_eval_interval",
        "0",
        "--use_compile",
        "false",
        "--checkpoint_path",
        str(ar_ckpt),
        "--loss_log_path",
        str(ar_loss),
    )

    assert ar_result.returncode == 0, ar_result.stderr
    assert ar_ckpt.exists()

    bd_result = run_repo_cmd(
        "train.py",
        "--model",
        "bd3lm",
        "--data",
        "climbmix",
        "--max_iters",
        "1",
        "--eval_interval",
        "0",
        "--batch_size",
        "2",
        "--block_size",
        "16",
        "--block_len",
        "4",
        "--n_embd",
        "16",
        "--n_head",
        "4",
        "--n_layer",
        "1",
        "--save_interval",
        "0",
        "--num_final_samples",
        "0",
        "--sample_interval",
        "0",
        "--gpt2_eval_interval",
        "0",
        "--use_compile",
        "false",
        "--resume_from",
        str(ar_ckpt),
        "--checkpoint_path",
        str(bd_ckpt),
        "--loss_log_path",
        str(bd_loss),
    )

    assert bd_result.returncode == 0, bd_result.stderr
    assert "Loaded model weights from" in bd_result.stdout
    assert "optimizer_reset=true" in bd_result.stdout
    assert bd_ckpt.exists()


@requires_cuda_climbmix
def test_warmup_stable_holds_lr_constant(tmp_path):
    """--warmup_stable should keep LR constant after warmup."""
    result = run_repo_cmd(
        "train.py",
        "--model", "mdlm",
        "--data", "climbmix",
        "--max_iters", "101",
        "--warmup_iters", "5",
        "--warmup_stable", "true",
        "--learning_rate", "1e-2",
        "--eval_interval", "0",
        "--batch_size", "2",
        "--block_size", "16",
        "--n_embd", "16",
        "--n_head", "4",
        "--n_layer", "1",
        "--save_interval", "0",
        "--num_final_samples", "0",
        "--sample_interval", "0",
        "--gpt2_eval_interval", "0",
        "--use_compile", "false",
        "--checkpoint_path", str(tmp_path / "ckpt.pt"),
        "--loss_log_path", str(tmp_path / "loss.pkl"),
    )
    assert result.returncode == 0, result.stderr
    # step 100 log line should show lr = learning_rate = 0.010000
    # (well past the 5-step warmup, and constant under warmup_stable)
    found = any("lr 0.010000" in line for line in result.stdout.splitlines()
                if line.startswith("step 100"))
    assert found, f"Expected LR 0.010000 at step 100; output:\n{result.stdout[-500:]}"


@requires_cuda_climbmix
def test_eval_interval_zero_still_uses_train_log_interval(tmp_path):
    """Disabling eval should not force per-step train logging."""
    result = run_repo_cmd(
        "train.py",
        "--model", "mdlm",
        "--data", "climbmix",
        "--max_iters", "21",
        "--warmup_iters", "1",
        "--eval_interval", "0",
        "--train_log_interval", "10",
        "--batch_size", "2",
        "--block_size", "16",
        "--n_embd", "16",
        "--n_head", "4",
        "--n_layer", "1",
        "--save_interval", "0",
        "--num_final_samples", "0",
        "--sample_interval", "0",
        "--gpt2_eval_interval", "0",
        "--gpt2_eval_samples", "0",
        "--skip_final_eval", "true",
        "--use_compile", "false",
        "--checkpoint_path", str(tmp_path / "ckpt.pt"),
        "--loss_log_path", str(tmp_path / "loss.pkl"),
    )
    assert result.returncode == 0, result.stderr

    logged_steps = []
    for line in result.stdout.splitlines():
        if not line.startswith("step ") or "grad_norm" not in line:
            continue
        step_str = line.split(" | ", 1)[0].split()[1]
        logged_steps.append(int(step_str))

    assert logged_steps == [0, 10, 20], (
        "Expected train progress logs at steps 0, 10, and 20 when "
        f"eval is disabled; got {logged_steps}"
    )


@requires_cuda_climbmix
def test_block_len_validated_against_final_block_size(tmp_path):
    """block_len validation must use the block_size set by the data source,
    not the CLI default. With --data climbmix, train.py overrides block_size
    to 2048, so --block_size 32 --block_len 8 must still succeed.
    """
    result = run_repo_cmd(
        "train.py",
        "--model", "bd3lm",
        "--data", "climbmix",
        "--max_iters", "1",
        "--block_size", "32",
        "--block_len", "8",
        "--eval_interval", "0",
        "--batch_size", "2",
        "--n_embd", "16",
        "--n_head", "4",
        "--n_layer", "1",
        "--save_interval", "0",
        "--num_final_samples", "0",
        "--sample_interval", "0",
        "--gpt2_eval_interval", "0",
        "--use_compile", "false",
        "--checkpoint_path", str(tmp_path / "ckpt.pt"),
        "--loss_log_path", str(tmp_path / "loss.pkl"),
    )
    assert result.returncode == 0, result.stderr


def test_climbmix_rejects_non_cuda():
    """--data climbmix on a non-CUDA machine should give a clear error."""
    if torch.cuda.is_available():
        pytest.skip("This test is only meaningful on non-CUDA machines")
    result = run_repo_cmd(
        "train.py",
        "--model", "mdlm",
        "--data", "climbmix",
        "--max_iters", "1",
        "--use_compile", "false",
    )
    assert result.returncode != 0
    assert "requires CUDA" in result.stderr


@pytest.mark.skip(reason="generate_samples.py not yet ported to new repo structure")
def test_generate_samples_auto_discovers_block_checkpoint(tmp_path):
    vocab_size = 64

    model = BlockMDLM(
        vocab_size=vocab_size,
        n_embd=16,
        n_head=4,
        n_layer=1,
        head_dim=4,
        block_size=16,
        block_len=4,
        dropout=0.0,
    )

    ckpt_dir = tmp_path / "block_mdlm_bl4" / "0.1M"
    ckpt_dir.mkdir(parents=True)
    torch.save(
        {
            "iter": 1,
            "model_state_dict": model.state_dict(),
            "loss": 4.2,
            "args": {
                "n_embd": 16,
                "n_layer": 1,
                "n_head": 4,
                "block_len": 4,
            },
        },
        ckpt_dir / "ckpt.pt",
    )

    result = run_repo_cmd(
        "generate_samples.py",
        "--models",
        "bd3lm",
        "--size",
        "0.1M",
        "--num_samples",
        "1",
        "--block_size",
        "16",
        "--prompt_len",
        "8",
        "--ckpt_root",
        str(tmp_path),
    )

    assert result.returncode == 0, result.stderr
    assert "MODEL: bd3lm (0.1M bl=4)" in result.stdout
    assert "Continuation:" in result.stdout
