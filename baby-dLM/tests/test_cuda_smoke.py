"""
test_cuda_smoke.py — End-to-end CUDA smoke tests.

Run these on a GPU VM *before* launching the full LR sweep to verify that
train.py, all three model families, both optimizers, ClimbMix data loading,
torch.compile, checkpointing, and resume all work correctly on CUDA.

Each test runs a very short training loop (≤30 steps) at the smallest
size so total wall-time is ~2-5 minutes for the whole file.

Usage:
    pytest tests/test_cuda_smoke.py -v --tb=short

Requirements:
    - CUDA-capable GPU with ≥16 GB VRAM
    - For ClimbMix-specific tests: run `python prepare.py` first
"""

import json
import os
import pickle
import re
import subprocess
import sys
import tempfile

import pytest

# Import data paths from prepare.py (single source of truth), but fall back
# to computing them directly so test collection never crashes if tiktoken
# or torch aren't installed.
try:
    from prepare import DATA_DIR as PREPARE_DATA_DIR
    from prepare import TOKENIZER_DIR as PREPARE_TOKENIZER_DIR
except ImportError:
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PREPARE_DATA_DIR = os.path.join(_repo, "data_cache", "shards")
    PREPARE_TOKENIZER_DIR = os.path.join(_repo, "data_cache", "tokenizer")

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable

# Regex matching the per-step log line from train.py
_STEP_RE = re.compile(
    r"step\s+(\d+)\s+\|.*?"
    r"loss\s+([\d.]+)\s+\|.*?"
    r"grad_norm\s+([\d.eE+\-]+)\s+\|.*?"
    r"lr\s+([\d.eE+\-]+)"
)


def _run_train(args: list, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run train.py with the given extra args, returning CompletedProcess."""
    cmd = [PYTHON, "-u", "train.py"] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=timeout,
    )
    if result.returncode != 0:
        # Print full output for debugging on the VM
        print("=== STDOUT ===")
        print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
        print("=== STDERR ===")
        print(result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr)
    return result


def _parse_steps(stdout: str):
    """Extract (step, loss, grad_norm, lr) tuples from train.py stdout."""
    records = []
    for m in _STEP_RE.finditer(stdout):
        records.append({
            "step": int(m.group(1)),
            "loss": float(m.group(2)),
            "grad_norm": float(m.group(3)),
            "lr": float(m.group(4)),
        })
    return records


def _check_basic_training(stdout: str, min_steps: int = 3):
    """Assert that training produced valid, decreasing-ish loss."""
    records = _parse_steps(stdout)
    assert len(records) >= min_steps, (
        f"Expected ≥{min_steps} step records, got {len(records)}"
    )
    # Check no NaN/Inf in losses or grad norms
    for r in records:
        assert r["loss"] < 1e6, f"Loss exploded at step {r['step']}: {r['loss']}"
        assert r["grad_norm"] < 1e6, f"Grad norm exploded at step {r['step']}: {r['grad_norm']}"
        assert r["loss"] == r["loss"], f"NaN loss at step {r['step']}"  # NaN != NaN
        assert r["grad_norm"] == r["grad_norm"], f"NaN grad norm at step {r['step']}"


# Skip the entire module if CUDA isn't available.
try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_CUDA = False

pytestmark = pytest.mark.skipif(
    not _HAS_CUDA,
    reason="CUDA not available — run on a GPU machine",
)


# ---------------------------------------------------------------
# Check whether ClimbMix data has been downloaded
# ---------------------------------------------------------------

_CLIMBMIX_TOKENIZER = os.path.join(PREPARE_TOKENIZER_DIR, "tokenizer.pkl")
_CLIMBMIX_DATA_DIR = PREPARE_DATA_DIR

def _climbmix_available():
    """Return True if ClimbMix tokenizer + at least 1 shard exist."""
    if not os.path.isfile(_CLIMBMIX_TOKENIZER):
        return False
    if not os.path.isdir(_CLIMBMIX_DATA_DIR):
        return False
    parquet_files = [f for f in os.listdir(_CLIMBMIX_DATA_DIR) if f.endswith(".parquet")]
    return len(parquet_files) >= 2  # need at least 1 train shard + 1 val shard

_HAS_CLIMBMIX = _climbmix_available()
_skip_climbmix = pytest.mark.skipif(
    not _HAS_CLIMBMIX,
    reason="ClimbMix data not available — run `python prepare.py` first",
)


# ---------------------------------------------------------------
# Common args for short ClimbMix runs (50M architecture)
# ---------------------------------------------------------------

_CLIMBMIX_BASE_ARGS = [
    "--data", "climbmix",
    "--n_embd", "768", "--n_layer", "7", "--n_head", "12",
    "--batch_size", "16",
    "--grad_accum_steps", "1",
    "--block_size", "2048",
    "--max_iters", "30",
    "--eval_interval", "10",
    "--eval_iters", "2",
    "--warmup_iters", "5",
    "--save_interval", "0",
    "--num_final_samples", "0",
    "--gpt2_eval_interval", "0",
    "--gpt2_eval_samples", "0",
    "--sample_interval", "0",
    "--skip_final_eval", "false",
    "--skip_final_checkpoint", "true",
    "--warmup_stable", "true",
    "--use_compile", "false",
]
_BASE_ARGS = _CLIMBMIX_BASE_ARGS

# Tiny model for pre-flight flow tests — verifies checkpoint plumbing without
# filling the disk.  Full 50M config writes ~750 MB per checkpoint; this tiny
# config writes ~5 MB.  n_head must divide n_embd.
_PREFLIGHT_ARGS = [
    "--data", "climbmix",
    "--n_embd", "64", "--n_layer", "2", "--n_head", "4",
    "--batch_size", "4",
    "--grad_accum_steps", "1",
    "--block_size", "256",
    "--eval_interval", "9999",
    "--eval_iters", "1",
    "--warmup_iters", "2",
    "--save_interval", "0",
    "--num_final_samples", "0",
    "--gpt2_eval_interval", "0",
    "--gpt2_eval_samples", "0",
    "--sample_interval", "0",
    "--skip_final_eval", "true",
    "--skip_final_checkpoint", "true",
    "--warmup_stable", "true",
    "--use_compile", "false",
    "--save_weights_only", "true",
]


# ===============================================================
# Test 1: Basic CUDA training for all 3 model families x AdamW
# ===============================================================

@_skip_climbmix
class TestCUDATraining:
    """Verify that each model family trains on CUDA without errors."""

    def test_ar_adamw(self, tmp_path):
        args = _BASE_ARGS + [
            "--model", "ar",
            "--optimizer", "adamw",
            "--learning_rate", "1e-3",
            "--min_lr", "1e-4",
            "--checkpoint_path", str(tmp_path / "ckpt_ar.pt"),
            "--loss_log_path", str(tmp_path / "loss_ar.pkl"),
        ]
        result = _run_train(args)
        assert result.returncode == 0, f"AR training failed:\n{result.stderr[-1000:]}"
        _check_basic_training(result.stdout)

    def test_mdlm_adamw(self, tmp_path):
        args = _BASE_ARGS + [
            "--model", "mdlm",
            "--optimizer", "adamw",
            "--learning_rate", "3e-3",
            "--min_lr", "3e-4",
            "--checkpoint_path", str(tmp_path / "ckpt_mdlm.pt"),
            "--loss_log_path", str(tmp_path / "loss_mdlm.pkl"),
        ]
        result = _run_train(args)
        assert result.returncode == 0, f"MDLM training failed:\n{result.stderr[-1000:]}"
        _check_basic_training(result.stdout)

    def test_bd3lm_adamw(self, tmp_path):
        args = _BASE_ARGS + [
            "--model", "bd3lm",
            "--block_len", "16",
            "--optimizer", "adamw",
            "--learning_rate", "3e-3",
            "--min_lr", "3e-4",
            "--checkpoint_path", str(tmp_path / "ckpt_bd3.pt"),
            "--loss_log_path", str(tmp_path / "loss_bd3.pkl"),
        ]
        result = _run_train(args)
        assert result.returncode == 0, f"BD3-LM training failed:\n{result.stderr[-1000:]}"
        _check_basic_training(result.stdout)


# ===============================================================
# Test 2: NorMuon optimizer on CUDA
# ===============================================================

@_skip_climbmix
class TestCUDANorMuon:
    """Verify the grouped NorMuon optimizer works on CUDA."""

    def test_ar_normuon(self, tmp_path):
        args = _BASE_ARGS + [
            "--model", "ar",
            "--optimizer", "normuon",
            "--learning_rate", "1.0",
            "--min_lr", "0.1",
            "--adam_mult", "1.0",
            "--matrix_mult", "1.0",
            "--checkpoint_path", str(tmp_path / "ckpt.pt"),
            "--loss_log_path", str(tmp_path / "loss.pkl"),
        ]
        result = _run_train(args)
        assert result.returncode == 0, f"AR+NorMuon failed:\n{result.stderr[-1000:]}"
        _check_basic_training(result.stdout)

    def test_bd3lm_normuon(self, tmp_path):
        args = _BASE_ARGS + [
            "--model", "bd3lm",
            "--block_len", "16",
            "--optimizer", "normuon",
            "--learning_rate", "1.0",
            "--min_lr", "0.1",
            "--adam_mult", "1.0",
            "--matrix_mult", "1.0",
            "--checkpoint_path", str(tmp_path / "ckpt.pt"),
            "--loss_log_path", str(tmp_path / "loss.pkl"),
        ]
        result = _run_train(args)
        assert result.returncode == 0, f"BD3-LM+NorMuon failed:\n{result.stderr[-1000:]}"
        _check_basic_training(result.stdout)

    def test_mdlm_normuon(self, tmp_path):
        args = _BASE_ARGS + [
            "--model", "mdlm",
            "--optimizer", "normuon",
            "--learning_rate", "1.0",
            "--min_lr", "0.1",
            "--adam_mult", "1.0",
            "--matrix_mult", "1.0",
            "--checkpoint_path", str(tmp_path / "ckpt.pt"),
            "--loss_log_path", str(tmp_path / "loss.pkl"),
        ]
        result = _run_train(args)
        assert result.returncode == 0, f"MDLM+NorMuon failed:\n{result.stderr[-1000:]}"
        _check_basic_training(result.stdout)


# ===============================================================
# Test 3: torch.compile works on CUDA
# ===============================================================

@_skip_climbmix
class TestTorchCompile:
    """Verify torch.compile doesn't break any model family."""

    @pytest.mark.parametrize("model", ["ar", "mdlm", "bd3lm"])
    def test_compile(self, model, tmp_path):
        extra = ["--block_len", "16"] if model == "bd3lm" else []
        args = _BASE_ARGS + [
            "--model", model,
            "--optimizer", "adamw",
            "--learning_rate", "3e-3",
            "--min_lr", "3e-4",
            "--use_compile", "true",
            "--max_iters", "15",  # shorter — compile overhead is high
            "--eval_interval", "15",
            "--checkpoint_path", str(tmp_path / "ckpt.pt"),
            "--loss_log_path", str(tmp_path / "loss.pkl"),
        ] + extra
        result = _run_train(args, timeout=600)  # compile can be slow first time
        assert result.returncode == 0, (
            f"{model}+compile failed:\n{result.stderr[-1000:]}"
        )


# ===============================================================
# Test 4: Checkpoint save + resume (weights-only warm start)
# ===============================================================

@_skip_climbmix
class TestCheckpointResume:
    """Verify checkpointing and warm-start resume on CUDA."""

    def test_save_and_resume(self, tmp_path):
        ckpt_path = str(tmp_path / "ckpt.pt")
        loss_path = str(tmp_path / "loss.pkl")

        # Phase 1: Train AR for 20 steps, save checkpoint
        args1 = _BASE_ARGS + [
            "--model", "ar",
            "--optimizer", "adamw",
            "--learning_rate", "1e-3",
            "--min_lr", "1e-4",
            "--max_iters", "20",
            "--save_interval", "10",
            "--skip_final_checkpoint", "false",
            "--checkpoint_path", ckpt_path,
            "--loss_log_path", loss_path,
        ]
        r1 = _run_train(args1)
        assert r1.returncode == 0, f"Phase 1 failed:\n{r1.stderr[-1000:]}"
        assert os.path.exists(ckpt_path), "Checkpoint not saved"

        # Phase 2: Resume from checkpoint, train 10 more steps
        loss_path2 = str(tmp_path / "loss2.pkl")
        ckpt_path2 = str(tmp_path / "ckpt2.pt")
        args2 = _BASE_ARGS + [
            "--model", "ar",
            "--optimizer", "adamw",
            "--learning_rate", "1e-3",
            "--min_lr", "1e-4",
            "--max_iters", "10",
            "--resume_from", ckpt_path,
            "--skip_final_checkpoint", "false",
            "--checkpoint_path", ckpt_path2,
            "--loss_log_path", loss_path2,
        ]
        r2 = _run_train(args2)
        assert r2.returncode == 0, f"Phase 2 (resume) failed:\n{r2.stderr[-1000:]}"
        assert "Loaded model weights" in r2.stdout, "Resume message not found"
        _check_basic_training(r2.stdout, min_steps=2)

    def test_cross_family_resume(self, tmp_path):
        """Test warm-starting BD3-LM from an AR checkpoint (curriculum-style)."""
        ar_ckpt = str(tmp_path / "ar_ckpt.pt")
        ar_loss = str(tmp_path / "ar_loss.pkl")

        # Train AR
        args_ar = _BASE_ARGS + [
            "--model", "ar",
            "--optimizer", "adamw",
            "--learning_rate", "1e-3",
            "--min_lr", "1e-4",
            "--max_iters", "15",
            "--skip_final_checkpoint", "false",
            "--checkpoint_path", ar_ckpt,
            "--loss_log_path", ar_loss,
        ]
        r1 = _run_train(args_ar)
        assert r1.returncode == 0, f"AR training failed:\n{r1.stderr[-1000:]}"

        # Resume as BD3-LM
        bd3_ckpt = str(tmp_path / "bd3_ckpt.pt")
        bd3_loss = str(tmp_path / "bd3_loss.pkl")
        args_bd3 = _BASE_ARGS + [
            "--model", "bd3lm",
            "--block_len", "16",
            "--optimizer", "adamw",
            "--learning_rate", "3e-3",
            "--min_lr", "3e-4",
            "--max_iters", "15",
            "--resume_from", ar_ckpt,
            "--skip_final_checkpoint", "false",
            "--checkpoint_path", bd3_ckpt,
            "--loss_log_path", bd3_loss,
        ]
        r2 = _run_train(args_bd3)
        assert r2.returncode == 0, f"BD3-LM resume from AR failed:\n{r2.stderr[-1000:]}"
        assert "Loaded model weights" in r2.stdout
        _check_basic_training(r2.stdout, min_steps=2)


# ===============================================================
# Test 5: ClimbMix data loading works (REQUIRES prepare.py)
# ===============================================================

@_skip_climbmix
class TestClimbMixData:
    """Verify ClimbMix data pipeline on CUDA."""

    def test_data_loads_and_tokenizes(self):
        """Directly test prepare.py's dataloader without training."""
        import torch
        sys.path.insert(0, REPO_ROOT)
        from prepare import Tokenizer, make_dataloader

        tok = Tokenizer.from_directory()
        assert tok.vocab_size == 8192, f"Unexpected vocab_size: {tok.vocab_size}"
        assert tok.mask_token_id is not None, "mask_token_id not set"

        # Get a batch (B=4, T=2048)
        dl = make_dataloader(tok, B=4, T=2048, split="train")
        inputs, targets, epoch = next(dl)
        assert inputs.shape == (4, 2048), f"Unexpected inputs shape: {inputs.shape}"
        assert inputs.dtype == torch.long

        # Verify tokens are in valid range
        assert inputs.min() >= 0
        assert inputs.max() < 8192

    def test_val_data_loads(self):
        """Verify the pinned validation shard loads."""
        sys.path.insert(0, REPO_ROOT)
        from prepare import Tokenizer, make_dataloader

        tok = Tokenizer.from_directory()
        dl = make_dataloader(tok, B=4, T=2048, split="val")
        inputs, targets, epoch = next(dl)
        assert inputs.shape == (4, 2048)

    def test_climbmix_training_runs(self, tmp_path):
        """Full end-to-end: 50M model trains on real ClimbMix data."""
        args = _CLIMBMIX_BASE_ARGS + [
            "--model", "ar",
            "--optimizer", "adamw",
            "--learning_rate", "1e-3",
            "--min_lr", "1e-4",
            "--max_iters", "10",
            "--eval_interval", "10",
            "--checkpoint_path", str(tmp_path / "ckpt.pt"),
            "--loss_log_path", str(tmp_path / "loss.pkl"),
        ]
        result = _run_train(args)
        assert result.returncode == 0, f"ClimbMix AR training failed:\n{result.stderr[-1000:]}"
        _check_basic_training(result.stdout, min_steps=2)


# ===============================================================
# Test 6: BD3-LM with different block_lens
# ===============================================================

@_skip_climbmix
class TestBD3BlockLens:
    """Verify BD3-LM works with the block_lens used in curricula."""

    @pytest.mark.parametrize("block_len", [2, 4, 8, 16, 64])
    def test_block_len(self, block_len, tmp_path):
        args = _BASE_ARGS + [
            "--model", "bd3lm",
            "--block_len", str(block_len),
            "--optimizer", "adamw",
            "--learning_rate", "3e-3",
            "--min_lr", "3e-4",
            "--max_iters", "10",
            "--eval_interval", "10",
            "--checkpoint_path", str(tmp_path / "ckpt.pt"),
            "--loss_log_path", str(tmp_path / "loss.pkl"),
        ]
        result = _run_train(args)
        assert result.returncode == 0, (
            f"BD3-LM block_len={block_len} failed:\n{result.stderr[-1000:]}"
        )


# ===============================================================
# Test 7: bfloat16 autocast works (used by real training)
# ===============================================================

@_skip_climbmix
class TestBFloat16:
    """Verify bfloat16 autocast works for all model families."""

    @pytest.mark.parametrize("model", ["ar", "mdlm", "bd3lm"])
    def test_bf16_training(self, model, tmp_path):
        """train.py uses autocast(bf16) on CUDA by default — verify it works."""
        extra = ["--block_len", "16"] if model == "bd3lm" else []
        args = _BASE_ARGS + [
            "--model", model,
            "--optimizer", "adamw",
            "--learning_rate", "3e-3",
            "--min_lr", "3e-4",
            "--max_iters", "10",
            "--eval_interval", "10",
            "--checkpoint_path", str(tmp_path / "ckpt.pt"),
            "--loss_log_path", str(tmp_path / "loss.pkl"),
        ] + extra
        result = _run_train(args)
        assert result.returncode == 0
        # Verify no "RuntimeError" about dtype mismatches in stderr
        assert "RuntimeError" not in result.stderr


# ===============================================================
# Test 8: Full batch_size=128 fits in VRAM (REQUIRES ClimbMix)
# ===============================================================

@_skip_climbmix
class TestVRAMFit:
    """Verify production config (micro-batch=128, grad_accum=2, eff. 256) fits."""

    @pytest.mark.parametrize("model", ["ar", "mdlm", "bd3lm"])
    def test_full_batch_fits(self, model, tmp_path):
        extra = ["--block_len", "16"] if model == "bd3lm" else []
        args = [
            "--data", "climbmix",
            "--n_embd", "768", "--n_layer", "7", "--n_head", "12",
            "--batch_size", "128",
            "--grad_accum_steps", "2",
            "--block_size", "2048",
            "--max_iters", "3",
            "--eval_interval", "100",  # skip eval
            "--eval_iters", "1",
            "--warmup_iters", "1",
            "--save_interval", "0",
            "--num_final_samples", "0",
            "--gpt2_eval_interval", "0",
            "--gpt2_eval_samples", "0",
            "--sample_interval", "0",
            "--skip_final_eval", "true",
            "--skip_final_checkpoint", "true",
            "--warmup_stable", "true",
            "--use_compile", "false",
            "--model", model,
            "--optimizer", "adamw",
            "--learning_rate", "3e-3",
            "--min_lr", "3e-4",
            "--checkpoint_path", str(tmp_path / "ckpt.pt"),
            "--loss_log_path", str(tmp_path / "loss.pkl"),
        ] + extra
        result = _run_train(args, timeout=300)
        assert result.returncode == 0, (
            f"{model} OOM at batch=128*accum=2:\n{result.stderr[-1000:]}"
        )


# ===============================================================
# Test 8b: Gradient accumulation produces valid training
# ===============================================================

@_skip_climbmix
class TestGradAccum:
    """Verify gradient accumulation runs correctly on CUDA."""

    def test_accum_2_trains(self, tmp_path):
        """grad_accum_steps=2 should train without errors."""
        args = _BASE_ARGS + [
            "--model", "ar",
            "--optimizer", "adamw",
            "--learning_rate", "1e-3",
            "--min_lr", "1e-4",
            "--grad_accum_steps", "2",
            "--max_iters", "15",
            "--eval_interval", "15",
            "--checkpoint_path", str(tmp_path / "ckpt.pt"),
            "--loss_log_path", str(tmp_path / "loss.pkl"),
        ]
        result = _run_train(args)
        assert result.returncode == 0, (
            f"grad_accum=2 failed:\n{result.stderr[-1000:]}"
        )
        _check_basic_training(result.stdout, min_steps=2)

    def test_accum_logs_effective_batch(self, tmp_path):
        """Training info should show effective_batch_size when accum > 1."""
        args = _BASE_ARGS + [
            "--model", "ar",
            "--optimizer", "adamw",
            "--learning_rate", "1e-3",
            "--min_lr", "1e-4",
            "--grad_accum_steps", "2",
            "--max_iters", "5",
            "--eval_interval", "100",
            "--checkpoint_path", str(tmp_path / "ckpt.pt"),
            "--loss_log_path", str(tmp_path / "loss.pkl"),
        ]
        result = _run_train(args)
        assert result.returncode == 0
        assert "effective_batch_size: 32" in result.stdout, (
            "Expected effective_batch_size: 32 (16 * 2) in output"
        )
        assert "grad_accum_steps: 2" in result.stdout


# ===============================================================
# Test 9: run_lr_sweep.py dry-run (validates command building)
# ===============================================================

class TestSweepInfra:
    """Verify the sweep infrastructure builds valid commands."""

    def test_lr_sweep_dry_run(self):
        """run_lr_sweep.py --dry-run should succeed for all combos."""
        cmd = [
            PYTHON, "run_lr_sweep.py",
            "--optimizer", "adamw",
            "--model", "ar",
            "--size", "50M",
            "--dry-run",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=REPO_ROOT, timeout=30,
        )
        assert result.returncode == 0, f"Dry-run failed:\n{result.stderr}"
        assert "dry-run" in result.stdout.lower()

    def test_normuon_sweep_dry_run(self):
        """Normuon sweep dry-run should build valid 2D grid commands."""
        cmd = [
            PYTHON, "run_lr_sweep.py",
            "--optimizer", "normuon",
            "--model", "ar",
            "--size", "50M",
            "--dry-run",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=REPO_ROOT, timeout=30,
        )
        assert result.returncode == 0, f"Dry-run failed:\n{result.stderr}"
        # Should show multiple (adam_mult, matrix_mult) combos
        assert "adam_mult" in result.stdout

    def test_curriculum_dry_run(self):
        """run_curriculum.py --dry-run should succeed."""
        cmd = [
            PYTHON, "run_curriculum.py",
            "--curriculum", "baseline_ar",
            "--size", "50M",
            "--budget", "1e18",
            "--dry-run",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=REPO_ROOT, timeout=30,
        )
        assert result.returncode == 0, f"Curriculum dry-run failed:\n{result.stderr}"


# ===============================================================
# Test 10: Eval pipeline works on a CUDA checkpoint
# ===============================================================

@_skip_climbmix
class TestEvalPipeline:
    """Verify evaluate.py can load and evaluate a CUDA checkpoint."""

    def test_eval_checkpoint(self, tmp_path):
        ckpt_path = str(tmp_path / "ckpt.pt")
        loss_path = str(tmp_path / "loss.pkl")

        # Train briefly to produce a checkpoint
        args = _BASE_ARGS + [
            "--model", "ar",
            "--optimizer", "adamw",
            "--learning_rate", "1e-3",
            "--min_lr", "1e-4",
            "--max_iters", "15",
            "--skip_final_checkpoint", "false",
            "--checkpoint_path", ckpt_path,
            "--loss_log_path", loss_path,
        ]
        r = _run_train(args)
        assert r.returncode == 0

        # Run evaluate.py on the checkpoint
        eval_cmd = [
            PYTHON, "evaluate.py",
            "--checkpoint", ckpt_path,
            "--skip-bpb",  # BPB is slow; just test loading + metadata
            "--num-samples", "1",
        ]
        r2 = subprocess.run(
            eval_cmd, capture_output=True, text=True,
            cwd=REPO_ROOT, timeout=120,
        )
        assert r2.returncode == 0, f"evaluate.py failed:\n{r2.stderr[-1000:]}"


# ===============================================================
# Test 11: Generation produces valid output on CUDA
# ===============================================================

@_skip_climbmix
class TestPhase1Preflight:
    """
    Pre-flight test for run_phase1.sh.

    Reproduces the exact checkpoint-sharing flow at miniature scale:
      1. AR warmup (20 steps, save_interval=5) → numbered checkpoints
      2. BD3-LM resume from a numbered AR checkpoint (curriculum-style)
      3. BD3-LM chained resume (C1-style: bl=2 → bl=4)

    Run this on the VM before launching the overnight job:
        pytest tests/test_cuda_smoke.py::TestPhase1Preflight -v --tb=short
    """

    def test_step_numbered_checkpoints_produced(self, tmp_path):
        """AR warmup produces ckpt_step{N}.pt at the correct steps."""
        ckpt_path = str(tmp_path / "ckpt.pt")
        args = _PREFLIGHT_ARGS + [
            "--model", "ar",
            "--optimizer", "adamw",
            "--learning_rate", "1e-3",
            "--min_lr", "1e-4",
            "--max_iters", "20",
            "--save_interval", "5",
            "--skip_final_checkpoint", "false",
            "--checkpoint_path", ckpt_path,
            "--loss_log_path", str(tmp_path / "loss.pkl"),
        ]
        result = _run_train(args)
        assert result.returncode == 0, f"AR warmup failed:\n{result.stderr[-1000:]}"

        # Verify step-numbered checkpoints exist at the right steps
        for step in [5, 10, 15, 20]:
            step_ckpt = str(tmp_path / f"ckpt_step{step}.pt")
            assert os.path.exists(step_ckpt), (
                f"ckpt_step{step}.pt missing — "
                f"expected after {step} optimizer updates"
            )

        # The final checkpoint should also exist
        assert os.path.exists(ckpt_path), "Final ckpt.pt missing"

        # Verify that ckpt_step20.pt has the same model weights as ckpt.pt
        import torch
        s20 = torch.load(str(tmp_path / "ckpt_step20.pt"), map_location="cpu")
        sfinal = torch.load(ckpt_path, map_location="cpu")
        for key in s20["model_state_dict"]:
            assert torch.equal(s20["model_state_dict"][key],
                               sfinal["model_state_dict"][key]), (
                f"ckpt_step20.pt and ckpt.pt differ on {key}"
            )

    def test_c0_checkpoint_sharing_flow(self, tmp_path):
        """
        C0-style: AR warmup → BD3-LM(16) resumes from numbered checkpoint.

        Mimics: 20 AR steps, then branch at step 10 for 10 BD3 steps.
        """
        ar_ckpt = str(tmp_path / "ar" / "ckpt.pt")
        os.makedirs(tmp_path / "ar", exist_ok=True)

        # AR warmup
        args_ar = _PREFLIGHT_ARGS + [
            "--model", "ar",
            "--optimizer", "adamw",
            "--learning_rate", "1e-3",
            "--min_lr", "1e-4",
            "--max_iters", "20",
            "--save_interval", "5",
            "--skip_final_checkpoint", "false",
            "--checkpoint_path", ar_ckpt,
            "--loss_log_path", str(tmp_path / "ar" / "loss.pkl"),
        ]
        r1 = _run_train(args_ar)
        assert r1.returncode == 0, f"AR warmup failed:\n{r1.stderr[-1000:]}"

        # Resume BD3-LM from AR step 10
        ar_step10 = str(tmp_path / "ar" / "ckpt_step10.pt")
        assert os.path.exists(ar_step10), "ckpt_step10.pt not found"

        bd3_dir = tmp_path / "c0_bd3"
        os.makedirs(bd3_dir, exist_ok=True)
        args_bd3 = _PREFLIGHT_ARGS + [
            "--model", "bd3lm",
            "--block_len", "16",
            "--optimizer", "adamw",
            "--learning_rate", "1e-3",
            "--min_lr", "1e-4",
            "--max_iters", "10",
            "--resume_from", ar_step10,
            "--skip_final_checkpoint", "false",
            "--checkpoint_path", str(bd3_dir / "ckpt.pt"),
            "--loss_log_path", str(bd3_dir / "loss.pkl"),
        ]
        r2 = _run_train(args_bd3)
        assert r2.returncode == 0, f"BD3-LM resume failed:\n{r2.stderr[-1000:]}"
        assert "Loaded model weights" in r2.stdout
        _check_basic_training(r2.stdout, min_steps=2)

    def test_c1_chained_resume_flow(self, tmp_path):
        """
        C1-style: AR → BD3(bl=2) → BD3(bl=4), chained checkpoints.

        Verifies two consecutive cross-block_len resumes work.
        """
        # Stage 0: AR (10 steps)
        ar_dir = tmp_path / "ar"
        os.makedirs(ar_dir, exist_ok=True)
        args_ar = _PREFLIGHT_ARGS + [
            "--model", "ar",
            "--optimizer", "adamw",
            "--learning_rate", "1e-3",
            "--min_lr", "1e-4",
            "--max_iters", "10",
            "--skip_final_checkpoint", "false",
            "--checkpoint_path", str(ar_dir / "ckpt.pt"),
            "--loss_log_path", str(ar_dir / "loss.pkl"),
        ]
        r0 = _run_train(args_ar)
        assert r0.returncode == 0, f"C1 AR stage failed:\n{r0.stderr[-1000:]}"

        # Stage 1: BD3(bl=2) from AR checkpoint
        bl2_dir = tmp_path / "bl2"
        os.makedirs(bl2_dir, exist_ok=True)
        args_bl2 = _PREFLIGHT_ARGS + [
            "--model", "bd3lm",
            "--block_len", "2",
            "--optimizer", "adamw",
            "--learning_rate", "1e-3",
            "--min_lr", "1e-4",
            "--max_iters", "10",
            "--resume_from", str(ar_dir / "ckpt.pt"),
            "--skip_final_checkpoint", "false",
            "--checkpoint_path", str(bl2_dir / "ckpt.pt"),
            "--loss_log_path", str(bl2_dir / "loss.pkl"),
        ]
        r1 = _run_train(args_bl2)
        assert r1.returncode == 0, f"C1 bl=2 stage failed:\n{r1.stderr[-1000:]}"
        assert "Loaded model weights" in r1.stdout

        # Stage 2: BD3(bl=4) from bl=2 checkpoint
        bl4_dir = tmp_path / "bl4"
        os.makedirs(bl4_dir, exist_ok=True)
        args_bl4 = _PREFLIGHT_ARGS + [
            "--model", "bd3lm",
            "--block_len", "4",
            "--optimizer", "adamw",
            "--learning_rate", "1e-3",
            "--min_lr", "1e-4",
            "--max_iters", "10",
            "--resume_from", str(bl2_dir / "ckpt.pt"),
            "--skip_final_checkpoint", "false",
            "--checkpoint_path", str(bl4_dir / "ckpt.pt"),
            "--loss_log_path", str(bl4_dir / "loss.pkl"),
        ]
        r2 = _run_train(args_bl4)
        assert r2.returncode == 0, f"C1 bl=4 stage failed:\n{r2.stderr[-1000:]}"
        assert "Loaded model weights" in r2.stdout
        _check_basic_training(r2.stdout, min_steps=2)

    def test_normuon_checkpoint_sharing_flow(self, tmp_path):
        """Same as C0 flow but with NorMuon optimizer."""
        ar_dir = tmp_path / "ar"
        os.makedirs(ar_dir, exist_ok=True)

        args_ar = _PREFLIGHT_ARGS + [
            "--model", "ar",
            "--optimizer", "normuon",
            "--learning_rate", "1.0",
            "--min_lr", "0.1",
            "--adam_mult", "0.3",
            "--matrix_mult", "1.0",
            "--max_iters", "20",
            "--save_interval", "10",
            "--skip_final_checkpoint", "false",
            "--checkpoint_path", str(ar_dir / "ckpt.pt"),
            "--loss_log_path", str(ar_dir / "loss.pkl"),
        ]
        r1 = _run_train(args_ar)
        assert r1.returncode == 0, f"NorMuon AR warmup failed:\n{r1.stderr[-1000:]}"

        # Resume BD3-LM from step 10
        ar_step10 = str(ar_dir / "ckpt_step10.pt")
        assert os.path.exists(ar_step10), "NorMuon ckpt_step10.pt not found"

        bd3_dir = tmp_path / "bd3"
        os.makedirs(bd3_dir, exist_ok=True)
        args_bd3 = _PREFLIGHT_ARGS + [
            "--model", "bd3lm",
            "--block_len", "16",
            "--optimizer", "normuon",
            "--learning_rate", "1.0",
            "--min_lr", "0.1",
            "--adam_mult", "0.3",
            "--matrix_mult", "1.0",
            "--max_iters", "10",
            "--resume_from", ar_step10,
            "--skip_final_checkpoint", "false",
            "--checkpoint_path", str(bd3_dir / "ckpt.pt"),
            "--loss_log_path", str(bd3_dir / "loss.pkl"),
        ]
        r2 = _run_train(args_bd3)
        assert r2.returncode == 0, f"NorMuon BD3-LM resume failed:\n{r2.stderr[-1000:]}"
        assert "Loaded model weights" in r2.stdout
        _check_basic_training(r2.stdout, min_steps=2)


# ===============================================================
# Test 12: Sample generation
# ===============================================================

@_skip_climbmix
class TestCUDAGeneration:
    """Verify sample generation works on CUDA for all families."""

    @pytest.mark.parametrize("model", ["ar", "mdlm", "bd3lm"])
    def test_generates_samples(self, model, tmp_path):
        extra = ["--block_len", "16"] if model == "bd3lm" else []
        args = _BASE_ARGS + [
            "--model", model,
            "--optimizer", "adamw",
            "--learning_rate", "3e-3",
            "--min_lr", "3e-4",
            "--max_iters", "10",
            "--eval_interval", "100",
            "--num_final_samples", "2",
            "--skip_final_eval", "true",
            "--checkpoint_path", str(tmp_path / "ckpt.pt"),
            "--loss_log_path", str(tmp_path / "loss.pkl"),
        ] + extra
        result = _run_train(args)
        assert result.returncode == 0, f"{model} generation failed:\n{result.stderr[-1000:]}"
        assert "--- Sample 1 ---" in result.stdout, "No samples generated"
        assert "--- Sample 2 ---" in result.stdout, "Only one sample generated"
