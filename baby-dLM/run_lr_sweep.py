#!/usr/bin/env python3
"""
run_lr_sweep.py — LR calibration for ClimbMix scaling experiments (§6.3).

Supports two optimizer families:
  - adamw:   1D sweep over a scalar learning_rate grid (unchanged from original)
  - normuon: 2D sweep over (adam_mult, matrix_mult) around Karpathy reference LRs

For each (optimizer_family, model_family, size) triple, trains for LR_SWEEP_STEPS
steps with a warmup-stable schedule, parses stdout for per-step train loss and
grad norm, selects the best stable config, and persists results.

Parallel execution: Use --jobs N to run N sweep jobs concurrently on the same
GPU.  Independent runs time-share CUDA and overlap data loading with compute.
Auto-detection (--jobs auto) picks N based on model VRAM and 80 GB budget,
but falls back to 1 worker when compile is enabled for sweep runs.

Usage:
    # Full sweep, auto-parallel:
    python run_lr_sweep.py --jobs auto

    # Single optimizer family, 6 concurrent jobs:
    python run_lr_sweep.py --optimizer adamw --jobs 6

    # Single (optimizer, model, size):
    python run_lr_sweep.py --optimizer normuon --model ar --size 50M

    # Dry-run (print commands without executing):
    python run_lr_sweep.py --dry-run
"""

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import experiment_config as ec

# ---------------------------------------------------------------
# Stdout parsing
# ---------------------------------------------------------------

# train.py prints lines like:
#   step 100 | tok_epoch 0.03 | loss 7.1234 | grad_norm 1.2345 | lr 0.001000
_STEP_RE = re.compile(
    r"step\s+(\d+)\s+\|.*?"
    r"loss\s+([\d.]+)\s+\|.*?"
    r"grad_norm\s+([\d.eE+\-]+)\s+\|.*?"
    r"lr\s+([\d.eE+\-]+)"
)


@dataclass
class SweepTrace:
    """Parsed training trace for a single run."""
    model: str
    size: str
    lr: float  # scalar LR for adamw, or dummy for normuon
    optimizer_family: str = "adamw"
    adam_mult: float = 1.0
    matrix_mult: float = 1.0
    steps: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    wall_seconds: float = 0.0
    early_aborted: bool = False

    @property
    def final_loss(self) -> Optional[float]:
        return self.losses[-1] if self.losses else None

    @property
    def grad_norm_stable(self) -> bool:
        """
        Heuristic: grad norm is 'stable' if no single reading in the
        second half of training exceeds 5× the median of the first half.
        """
        if len(self.grad_norms) < 4:
            return True
        mid = len(self.grad_norms) // 2
        first_half = sorted(self.grad_norms[:mid])
        median_first = first_half[len(first_half) // 2]
        if median_first == 0:
            return True
        threshold = 5.0 * median_first
        return all(gn <= threshold for gn in self.grad_norms[mid:])


def parse_stdout(text: str, model: str, size: str, lr: float,
                 optimizer_family: str = "adamw",
                 adam_mult: float = 1.0,
                 matrix_mult: float = 1.0) -> SweepTrace:
    """Extract (step, loss, grad_norm) triples from train.py stdout."""
    trace = SweepTrace(
        model=model, size=size, lr=lr,
        optimizer_family=optimizer_family,
        adam_mult=adam_mult, matrix_mult=matrix_mult,
    )
    for m in _STEP_RE.finditer(text):
        trace.steps.append(int(m.group(1)))
        trace.losses.append(float(m.group(2)))
        trace.grad_norms.append(float(m.group(3)))
    return trace


# ---------------------------------------------------------------
# LR selection
# ---------------------------------------------------------------

def select_best_lr(traces: List[SweepTrace]) -> Tuple[float, SweepTrace]:
    """
    Pick the best LR (adamw): among stable runs, choose the one with the
    lowest final train loss.  If all runs are unstable, fall back to
    the smallest LR.

    Returns (chosen_lr, chosen_trace).
    """
    stable = [t for t in traces if t.grad_norm_stable and t.final_loss is not None]
    if stable:
        best = min(stable, key=lambda t: t.final_loss)
    else:
        print("  WARNING: no stable runs found, falling back to smallest LR")
        best = min(traces, key=lambda t: t.lr)
    return best.lr, best


def select_best_normuon(traces: List[SweepTrace]) -> Tuple[dict, SweepTrace]:
    """
    Pick the best (adam_mult, matrix_mult) for normuon: among stable runs,
    choose the one with the lowest final train loss.  If all unstable,
    fall back to the (1.0, 1.0) reference setting.

    Returns (config_dict, chosen_trace).
    """
    stable = [t for t in traces if t.grad_norm_stable and t.final_loss is not None]
    if stable:
        best = min(stable, key=lambda t: t.final_loss)
    else:
        print("  WARNING: no stable normuon runs, falling back to reference (1.0, 1.0)")
        # Find the reference run
        ref = [t for t in traces if t.adam_mult == 1.0 and t.matrix_mult == 1.0]
        best = ref[0] if ref else min(traces, key=lambda t: t.final_loss or float('inf'))
    config = {"adam_mult": best.adam_mult, "matrix_mult": best.matrix_mult}
    return config, best


# ---------------------------------------------------------------
# VRAM estimation for auto-parallelism
# ---------------------------------------------------------------

# Peak VRAM in GB per model, at batch_size=32, seq_len=2048, bf16, no compile.
# These are conservative heuristics for auto-parallelism on A100-80GB.
# They were originally calibrated on the older, deeper ClimbMix size table and
# may overestimate usage for the current shallower/wider definitions; that is
# intentional until new empirical measurements are collected.
_VRAM_ESTIMATE_GB = {
    # (model, size) -> estimated peak GB at sweep batch_size (32)
    ("ar",    "50M"):  45,
    ("ar",    "98M"):  55,
    ("ar",    "170M"): 75,
    ("mdlm",  "50M"):  45,
    ("mdlm",  "98M"):  55,
    ("mdlm",  "170M"): 75,
    ("bd3lm", "50M"):  55,
    ("bd3lm", "98M"):  65,
    ("bd3lm", "170M"): 80,
}

# Leave this much headroom for CUDA overhead, fragmentation, etc.
_VRAM_HEADROOM_GB = 4
_DEFAULT_GPU_GB = 80  # A100-80GB
# Compile-enabled sweeps match production training, but auto-parallelism stays
# conservative because simultaneous first-compile passes can spike VRAM sharply.
_SWEEP_USE_COMPILE = True


def estimate_max_parallel(
    jobs: list,
    gpu_gb: int = _DEFAULT_GPU_GB,
    batch_size_override: Optional[int] = None,
) -> int:
    """
    Estimate the max number of concurrent jobs that fit in GPU VRAM.

    Strategy: assume worst-case (largest job in the list) and compute
    how many copies fit.  This is conservative but safe.

    When batch_size_override is set, scale VRAM estimates proportionally
    since activations dominate and scale linearly with batch_size.
    """
    if not jobs:
        return 1
    max_vram = max(
        _VRAM_ESTIMATE_GB.get((j["model"], j["size"]), 16)
        for j in jobs
    )
    # Note: VRAM estimates already reflect sweep batch_size (32), not
    # production batch_size.  No scaling needed.

    usable = gpu_gb - _VRAM_HEADROOM_GB
    n = max(1, int(usable / max_vram))
    return n


# ---------------------------------------------------------------
# Early-abort: detect divergence in real time
# ---------------------------------------------------------------

# Effective batch size for LR sweeps.  When --sweep-batch-size reduces the
# micro-batch, grad_accum is increased to keep this constant so that LR
# calibration is representative of production training.
_SWEEP_EFFECTIVE_BATCH = 256

_LOSS_ABORT_THRESHOLD = 100.0   # if loss exceeds this, LR is too high
_GRAD_ABORT_THRESHOLD = 1e4     # grad norm explosion
_ABORT_AFTER_STEP = 200         # only check after this many steps to allow warmup

_LIVE_STEP_RE = re.compile(
    r"step\s+(\d+)\s+\|.*?"
    r"loss\s+([\d.eE+\-]+|nan|inf|-inf)\s+\|.*?"
    r"grad_norm\s+([\d.eE+\-]+|nan|inf|-inf)",
    re.IGNORECASE,
)


def _should_abort_line(line: str) -> bool:
    """Check a single stdout line for signs of divergence."""
    m = _LIVE_STEP_RE.search(line)
    if not m:
        return False
    step = int(m.group(1))
    if step < _ABORT_AFTER_STEP:
        return False
    loss_str = m.group(2).lower()
    grad_str = m.group(3).lower()
    # Immediate abort on nan/inf strings
    if loss_str in ("nan", "inf", "-inf") or grad_str in ("nan", "inf", "-inf"):
        return True
    try:
        loss = float(loss_str)
        grad_norm = float(grad_str)
    except ValueError:
        return True  # unparseable
    if loss != loss or grad_norm != grad_norm:  # NaN from float()
        return True
    if loss > _LOSS_ABORT_THRESHOLD:
        return True
    if grad_norm > _GRAD_ABORT_THRESHOLD:
        return True
    return False


# ---------------------------------------------------------------
# Run one sweep job (with streaming early-abort)
# ---------------------------------------------------------------

# Thread-safe print
_print_lock = threading.Lock()


def _tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def run_single(
    model: str,
    size: str,
    lr: float,
    out_dir: str,
    *,
    dry_run: bool = False,
    optimizer: str = "adamw",
    adam_mult: float = 1.0,
    matrix_mult: float = 1.0,
    batch_size: Optional[int] = None,
    early_abort: bool = True,
) -> Optional[SweepTrace]:
    """Launch train.py for one configuration and return parsed trace."""
    os.makedirs(out_dir, exist_ok=True)

    warmup_iters = max(1, int(ec.LR_SWEEP_STEPS * ec.LR_SWEEP_WARMUP_FRAC))

    build_kwargs = dict(
        model=model,
        size=size,
        out_dir=out_dir,
        max_iters=ec.LR_SWEEP_STEPS,
        lr=lr,
        warmup_iters=warmup_iters,
        warmup_stable=True,
        skip_final_eval=True,
        skip_final_checkpoint=True,
        eval_interval=ec.LR_SWEEP_STEPS + 1,
        train_log_interval=10,
        save_interval=0,
        num_final_samples=0,
        gpt2_eval_interval=0,
        sample_interval=0,
        optimizer=optimizer,
        adam_mult=adam_mult,
        matrix_mult=matrix_mult,
        # Match production training behavior during LR calibration.
        use_compile=_SWEEP_USE_COMPILE,
    )
    # When a smaller sweep batch_size is requested, use grad accumulation
    # to maintain a fixed effective batch size of _SWEEP_EFFECTIVE_BATCH.
    if batch_size is not None:
        build_kwargs["batch_size"] = batch_size
        build_kwargs["grad_accum_steps"] = _SWEEP_EFFECTIVE_BATCH // batch_size
    else:
        build_kwargs["grad_accum_steps"] = _SWEEP_EFFECTIVE_BATCH // ec.CLIMBMIX_BATCH_SIZE

    cmd = ec.build_command(**build_kwargs)

    if optimizer == "normuon":
        label = f"normuon/{model}/{size} am={adam_mult},mm={matrix_mult}"
    else:
        label = f"adamw/{model}/{size} LR={lr:.1e}"

    if dry_run:
        _tprint(f"  [dry-run] {label}")
        _tprint(f"    {' '.join(cmd)}")
        return None

    t0 = time.time()
    _tprint(f"  START  {label}")

    # ---- Launch with streaming stdout for early-abort ----
    if early_abort:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

        stdout_lines = []
        aborted = False
        for line in proc.stdout:
            stdout_lines.append(line)
            if line.startswith("step "):
                _tprint(f"    {label} | {line.rstrip()}")
            if _should_abort_line(line):
                aborted = True
                _tprint(f"  ABORT  {label}  (diverged: {line.strip()})")
                proc.kill()
                break

        # Drain remaining stdout/stderr
        remaining_out, stderr = proc.communicate()
        if remaining_out:
            stdout_lines.append(remaining_out)
        stdout_text = "".join(stdout_lines)
        returncode = proc.returncode

        if aborted:
            trace = parse_stdout(
                stdout_text, model, size, lr,
                optimizer_family=optimizer,
                adam_mult=adam_mult,
                matrix_mult=matrix_mult,
            )
            trace.early_aborted = True
            trace.wall_seconds = time.time() - t0
            _tprint(f"  DONE   {label}  EARLY-ABORT  "
                    f"wall={trace.wall_seconds:.0f}s")
            return trace
    else:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        stdout_text = result.stdout
        stderr = result.stderr
        returncode = result.returncode

    wall = time.time() - t0

    if returncode != 0 and returncode != -9:  # -9 = killed (from abort)
        _tprint(f"  FAIL   {label}  exit={returncode}  wall={wall:.0f}s")
        if stderr:
            for line in stderr.strip().splitlines()[-3:]:
                _tprint(f"    stderr: {line}")
        return None

    trace = parse_stdout(
        stdout_text, model, size, lr,
        optimizer_family=optimizer,
        adam_mult=adam_mult,
        matrix_mult=matrix_mult,
    )
    trace.wall_seconds = wall

    if not trace.steps:
        _tprint(f"  FAIL   {label}  no metrics parsed  "
                f"({len(stdout_text)} chars)")
        return None

    status = "stable" if trace.grad_norm_stable else "UNSTABLE"
    _tprint(f"  DONE   {label}  {status}  "
            f"loss={trace.final_loss:.4f}  wall={wall:.0f}s")
    return trace


# ---------------------------------------------------------------
# Job description (for parallel dispatch)
# ---------------------------------------------------------------

@dataclass
class SweepJob:
    """One sweep run to execute."""
    optimizer: str
    model: str
    size: str
    lr: float
    adam_mult: float = 1.0
    matrix_mult: float = 1.0
    out_dir: str = ""

    @property
    def group_key(self) -> str:
        """Key for grouping runs that share a selection decision."""
        return f"{self.optimizer}|{self.model}|{self.size}"


def collect_jobs(
    optimizers: list,
    models: list,
    sizes: list,
    out_root: str,
) -> List[SweepJob]:
    """Build the flat list of all jobs to run."""
    jobs = []

    for opt in optimizers:
        for model in models:
            for size in sizes:
                if opt == "adamw":
                    grid = ec.LR_SWEEP_GRIDS.get(model, ec.LR_SWEEP_GRID_DEFAULT)
                    for lr in grid:
                        run_dir = os.path.join(
                            out_root, f"adamw_{model}_{size}", f"lr_{lr:.0e}"
                        )
                        jobs.append(SweepJob(
                            optimizer=opt, model=model, size=size,
                            lr=lr, out_dir=run_dir,
                        ))
                elif opt == "normuon":
                    for am in ec.NORMUON_ADAM_MULT_GRID:
                        for mm in ec.NORMUON_MATRIX_MULT_GRID:
                            run_dir = os.path.join(
                                out_root,
                                f"normuon_{model}_{size}",
                                f"am_{am:.1f}_mm_{mm:.1f}",
                            )
                            jobs.append(SweepJob(
                                optimizer=opt, model=model, size=size,
                                lr=1.0, adam_mult=am, matrix_mult=mm,
                                out_dir=run_dir,
                            ))
    return jobs


# ---------------------------------------------------------------
# Parallel sweep executor
# ---------------------------------------------------------------

def run_sweep_parallel(
    jobs: List[SweepJob],
    *,
    max_workers: int = 1,
    dry_run: bool = False,
    batch_size: Optional[int] = None,
    early_abort: bool = True,
) -> Dict[str, List[SweepTrace]]:
    """
    Execute all jobs, up to max_workers concurrently.
    Returns traces grouped by group_key.
    """
    grouped_traces: Dict[str, List[SweepTrace]] = defaultdict(list)

    if dry_run:
        for job in jobs:
            run_single(
                job.model, job.size, job.lr, job.out_dir,
                dry_run=True,
                optimizer=job.optimizer,
                adam_mult=job.adam_mult,
                matrix_mult=job.matrix_mult,
                batch_size=batch_size,
            )
        return grouped_traces

    total = len(jobs)
    completed = [0]  # mutable counter for thread-safe update
    _tprint(f"\nDispatching {total} jobs with {max_workers} workers\n")

    def _run_job(job: SweepJob) -> Tuple[str, Optional[SweepTrace]]:
        trace = run_single(
            job.model, job.size, job.lr, job.out_dir,
            optimizer=job.optimizer,
            adam_mult=job.adam_mult,
            matrix_mult=job.matrix_mult,
            batch_size=batch_size,
            early_abort=early_abort,
        )
        completed[0] += 1
        _tprint(f"  [{completed[0]}/{total}] progress")
        return job.group_key, trace

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_job, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                key, trace = future.result()
                if trace is not None:
                    grouped_traces[key].append(trace)
            except Exception as e:
                job = futures[future]
                _tprint(f"  ERROR  {job.group_key} lr={job.lr}: {e}")

    return grouped_traces


# ---------------------------------------------------------------
# Process results and persist
# ---------------------------------------------------------------

def process_results(
    grouped_traces: Dict[str, List[SweepTrace]],
    out_root: str,
) -> Tuple[dict, dict]:
    """
    For each (optimizer, model, size) group, select the best config
    and persist to calibrated_*.json.

    Returns (adamw_results, normuon_results).
    """
    adamw_results = {}
    normuon_results = {}

    for group_key, traces in sorted(grouped_traces.items()):
        opt, model, size = group_key.split("|")

        # Filter out early-aborted runs from selection (they're unstable by definition)
        viable = [t for t in traces if not t.early_aborted]
        if not viable:
            _tprint(f"\n  WARNING: all runs aborted for {group_key}, "
                    f"using all traces for fallback selection")
            viable = traces

        if not viable:
            _tprint(f"\n  ERROR: no traces for {group_key}")
            continue

        print(f"\n{'='*60}")
        print(f"Selection: {opt} / {model} / {size}  "
              f"({len(viable)} viable of {len(traces)} total)")
        print(f"{'='*60}")

        if opt == "adamw":
            chosen_lr, chosen_trace = select_best_lr(viable)
            print(f"  >> Selected LR = {chosen_lr:.1e}  "
                  f"(final_loss={chosen_trace.final_loss:.4f}, "
                  f"stable={chosen_trace.grad_norm_stable})")
            ec.set_calibrated_lr(model, size, chosen_lr)
            print(f"  >> Saved to calibrated_lrs.json")
            adamw_results[(model, size)] = chosen_lr

        elif opt == "normuon":
            chosen_config, chosen_trace = select_best_normuon(viable)
            print(f"  >> Selected: adam_mult={chosen_config['adam_mult']}, "
                  f"matrix_mult={chosen_config['matrix_mult']}  "
                  f"(final_loss={chosen_trace.final_loss:.4f}, "
                  f"stable={chosen_trace.grad_norm_stable})")
            ec.set_calibrated_normuon(
                model, size,
                chosen_config["adam_mult"],
                chosen_config["matrix_mult"],
            )
            print(f"  >> Saved to calibrated_normuon.json")
            normuon_results[(model, size)] = chosen_config

        # Save raw traces for diagnostics
        traces_dir = os.path.join(out_root, f"{opt}_{model}_{size}")
        os.makedirs(traces_dir, exist_ok=True)
        traces_path = os.path.join(traces_dir, "sweep_traces.pkl")
        with open(traces_path, "wb") as f:
            pickle.dump(traces, f)
        print(f"  >> Traces saved to {traces_path}")

    return adamw_results, normuon_results


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LR calibration sweep for ClimbMix scaling experiments"
    )
    parser.add_argument(
        "--optimizer", type=str, default=None,
        choices=ec.OPTIMIZER_FAMILIES,
        help="Sweep a single optimizer family (default: all)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=ec.ALL_MODELS,
        help="Sweep a single model family (default: all)",
    )
    parser.add_argument(
        "--size", type=str, default=None,
        choices=ec.CLIMBMIX_SIZES,
        help="Sweep a single size (default: all ClimbMix sizes)",
    )
    parser.add_argument(
        "--out-root", type=str, default="runs/lr_sweep",
        help="Root directory for sweep outputs",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--jobs", type=str, default="auto",
        help="Number of parallel jobs (default: auto). "
             "Use 'auto' for VRAM-based auto-detection, "
             "or 1 when compile-enabled sweeps are active, "
             "or an integer (e.g., '6'). Use '1' for sequential.",
    )
    parser.add_argument(
        "--sweep-batch-size", type=int, default=None,
        help="Override batch_size for sweep runs (default: use production "
             "batch_size from experiment_config). Smaller values use less "
             "VRAM and allow more parallelism, at the cost of noisier "
             "loss curves. Tune based on VRAM; 32 is a good starting point "
             "on A100-80GB.",
    )
    parser.add_argument(
        "--no-early-abort", action="store_true",
        help="Disable early-abort of diverged runs (run all configured sweep steps "
             "regardless of loss/grad-norm blow-ups).",
    )
    parser.add_argument(
        "--gpu-gb", type=int, default=_DEFAULT_GPU_GB,
        help="GPU VRAM in GB for auto-parallelism (default: 80 for A100-80GB)",
    )
    args = parser.parse_args()

    optimizers = [args.optimizer] if args.optimizer else ec.OPTIMIZER_FAMILIES
    models = [args.model] if args.model else ec.ALL_MODELS
    sizes = [args.size] if args.size else ec.CLIMBMIX_SIZES

    # ---- Collect all jobs ----
    jobs = collect_jobs(optimizers, models, sizes, args.out_root)

    if not jobs:
        print("No jobs to run.")
        return

    # ---- Determine parallelism ----
    if args.jobs == "auto":
        if _SWEEP_USE_COMPILE:
            max_workers = 1
            print("Auto-detected: 1 parallel worker "
                  "(compile-enabled sweeps use a conservative default)")
        else:
            # Convert jobs to dicts for VRAM estimation
            job_dicts = [{"model": j.model, "size": j.size} for j in jobs]
            max_workers = estimate_max_parallel(
                job_dicts,
                gpu_gb=args.gpu_gb,
                batch_size_override=args.sweep_batch_size,
            )
            print(f"Auto-detected: {max_workers} parallel workers "
                  f"(GPU: {args.gpu_gb}GB)")
    else:
        max_workers = int(args.jobs)
        if _SWEEP_USE_COMPILE and max_workers > 1:
            print("WARNING: compile-enabled sweeps with multiple workers may "
                  "hit large first-compile VRAM spikes")

    print(f"\nTotal jobs: {len(jobs)}")
    print(f"Parallel workers: {max_workers}")
    print(f"torch.compile: {'enabled' if _SWEEP_USE_COMPILE else 'disabled'}")
    if args.sweep_batch_size:
        print(f"Sweep batch_size: {args.sweep_batch_size} "
              f"(production: {ec.CLIMBMIX_BATCH_SIZE})")
    if not args.no_early_abort:
        print(f"Early-abort: enabled (loss>{_LOSS_ABORT_THRESHOLD} or "
              f"grad_norm>{_GRAD_ABORT_THRESHOLD} after step {_ABORT_AFTER_STEP})")

    # ---- Group keys for summary ----
    group_keys = sorted(set(j.group_key for j in jobs))
    print(f"Groups: {len(group_keys)}")
    for k in group_keys:
        n = sum(1 for j in jobs if j.group_key == k)
        print(f"  {k}: {n} runs")

    # ---- Execute ----
    t_start = time.time()
    grouped_traces = run_sweep_parallel(
        jobs,
        max_workers=max_workers,
        dry_run=args.dry_run,
        batch_size=args.sweep_batch_size,
        early_abort=not args.no_early_abort,
    )
    wall_total = time.time() - t_start

    if args.dry_run:
        print(f"\n[Dry-run complete: {len(jobs)} jobs listed]")
        return

    adamw_results, normuon_results = process_results(
        grouped_traces, args.out_root,
    )

    print(f"\nTotal wall time: {wall_total:.0f}s "
          f"({wall_total/60:.1f} min)")
    if len(jobs) > 1:
        sequential_est = sum(
            t.wall_seconds
            for traces in grouped_traces.values()
            for t in traces
        )
        if sequential_est > 0:
            print(f"Estimated sequential time: {sequential_est:.0f}s "
                  f"({sequential_est/60:.1f} min)")
            print(f"Speedup: {sequential_est/wall_total:.1f}×")

    # ---- Print summary tables ----
    if adamw_results:
        print(f"\n{'='*60}")
        print("SUMMARY: Calibrated AdamW Learning Rates")
        print(f"{'='*60}")
        print(f"{'model':>8s}  {'size':>5s}  {'LR':>10s}")
        print(f"{'-'*8:>8s}  {'-'*5:>5s}  {'-'*10:>10s}")
        for (model, size), lr in sorted(adamw_results.items()):
            print(f"{model:>8s}  {size:>5s}  {lr:>10.1e}")

    if normuon_results:
        print(f"\n{'='*60}")
        print("SUMMARY: Calibrated NorMuon Configs")
        print(f"{'='*60}")
        print(f"{'model':>8s}  {'size':>5s}  {'adam_mult':>10s}  {'matrix_mult':>12s}")
        print(f"{'-'*8:>8s}  {'-'*5:>5s}  {'-'*10:>10s}  {'-'*12:>12s}")
        for (model, size), cfg in sorted(normuon_results.items()):
            print(f"{model:>8s}  {size:>5s}  {cfg['adam_mult']:>10.2f}  "
                  f"{cfg['matrix_mult']:>12.2f}")

    # ---- Write combined JSON summary ----
    all_results = {}
    for (m, s), lr in adamw_results.items():
        all_results[f"adamw|{m}|{s}"] = {"lr": lr}
    for (m, s), cfg in normuon_results.items():
        all_results[f"normuon|{m}|{s}"] = cfg

    if all_results:
        summary_path = os.path.join(args.out_root, "lr_summary.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
