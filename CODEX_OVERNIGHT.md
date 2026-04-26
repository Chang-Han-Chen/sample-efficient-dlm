# CODEX_OVERNIGHT

## 2026-04-26 10:57:25 UTC

- Read `MoE_PLAN.md` and inspected the existing MoE-related worktree changes.
- Found that the sparse Switch MoE implementation, transformer wiring, loss integration, optimizer grouping, trainer args, and focused MoE tests were already present in the worktree.
- Ran `pytest -q tests/test_moe.py`; the implementation behaved correctly, but one test expected one MoE layer for `n_layers=3`. Under the plan's `alternating` placement rule, layers `[0, 2]` are selected, so the correct count is two.
- Updated `tests/test_moe.py` to match the plan's placement semantics.
- Added six MoE experiment configs under `configs/experiments/` for the requested AR/MDLM baseline, old-bundle, and value-embedding variants.

## TODO

- Full 5.1K-step W&B runs were not launched from this session. Use the six `configs/experiments/*moe*.yaml` files as the starting configs, tune capacity/router regularization/router LR per `MoE_PLAN.md`, and compare against the named dense W&B anchors.
- A broader stack run (`pytest -q tests/test_moe.py tests/test_training_stack.py tests/test_diffusion_stack.py`) has three non-MoE numerical strictness failures: chunked CE weight-gradient differs by about `9.1e-5`; BD3 blocked attention differs from dense-mask attention by about `7.4e-4`; BD3 blocked model logits differ by about `3.0e-3`. Focused MoE, config, and data-parallel tests pass.

## 2026-04-26 11:06:29 UTC

- Fixed data-parallel train-step compatibility by making MoE loss weights keyword-only for pmapped train steps and restoring pmap `in_axes` to the original positional arity.
- Updated `train_ar.py` to pass MoE loss weights by keyword, preserving existing dense and test call sites.
- Verification:
  - `python -m py_compile transformer/moe.py transformer/transformer.py training/loss.py training/step.py training/optimizer.py train_ar.py` passed.
  - `pytest -q tests/test_moe.py tests/test_configs.py` passed: 10 passed, 1 pytest config warning.
  - `pytest -q tests/test_data_parallel.py` passed: 5 passed, 1 pytest config warning.
  - AR synthetic MoE trainer smoke passed for 2 steps.
  - MDLM synthetic MoE + value embedding + split mask token + no mask vector smoke passed for 2 steps.
