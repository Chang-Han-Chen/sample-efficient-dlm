"""Import-path plumbing so a single test can build both the PyTorch and the
JAX Transformer side-by-side.

The two implementations both call themselves `sample_efficient_gpt.transformer.*`
(the JAX port reuses the PT namespace). To disambiguate, this module lets you
swap `sys.modules["sample_efficient_gpt"]` between two trees on demand.

Paths are computed from __file__, not hard-coded to a particular workspace.
"""

from __future__ import annotations

import os
import sys
import types
from contextlib import contextmanager

# The tests live in <repo>/jax/tests/. Climb up two levels to reach <repo>.
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
PT_REPO = os.path.join(REPO_ROOT, "pytorch")  # <repo>/pytorch/ holds the PT package root
JAX_DIR = os.path.join(REPO_ROOT, "jax")      # <repo>/jax/ holds the JAX transformer tree


def _clear_sample_efficient_gpt():
    for k in list(sys.modules):
        if k == "sample_efficient_gpt" or k.startswith("sample_efficient_gpt."):
            del sys.modules[k]


def _install_profiling_shim():
    """A minimal stub for `sample_efficient_gpt.utils.profiling` so JAX files
    that import nvtx_range don't drag in torch.cuda."""
    utils = types.ModuleType("sample_efficient_gpt.utils")
    utils.__path__ = []
    sys.modules["sample_efficient_gpt.utils"] = utils

    prof = types.ModuleType("sample_efficient_gpt.utils.profiling")

    @contextmanager
    def nvtx_range(_=None):
        yield

    prof.nvtx_range = nvtx_range
    sys.modules["sample_efficient_gpt.utils.profiling"] = prof


def load_jax():
    """Point `sample_efficient_gpt.transformer` at the JAX tree under /jax/."""
    _clear_sample_efficient_gpt()
    pkg = types.ModuleType("sample_efficient_gpt")
    pkg.__path__ = [JAX_DIR]
    sys.modules["sample_efficient_gpt"] = pkg
    _install_profiling_shim()
    import importlib
    return importlib.import_module("sample_efficient_gpt.transformer.transformer")


def _stub_triton_ops():
    """The PyTorch repo imports triton kernels at module load; stub them out
    so we can run on CPU without GPU/triton installed."""
    for name in [
        "sample_efficient_gpt.transformer.ops.rms_norm",
        "sample_efficient_gpt.transformer.ops.silu",
        "sample_efficient_gpt.transformer.ops.triton_flash_attn",
        "sample_efficient_gpt.transformer.ops.triton_flash_attn_qknorm",
        "sample_efficient_gpt.transformer.ops.fused_linear_cross_entropy",
        "sample_efficient_gpt.transformer.ops.cross_entropy",
        "sample_efficient_gpt.transformer.ops.fused_add_rms_norm",
        "sample_efficient_gpt.transformer.ops.utils",
    ]:
        sys.modules[name] = types.ModuleType(name)
    sys.modules["sample_efficient_gpt.transformer.ops.rms_norm"].LigerRMSNormFunction = object
    sys.modules["sample_efficient_gpt.transformer.ops.silu"].LigerSiLUMulFunction = object

    class _Flash:
        @staticmethod
        def apply(*a, **k):
            raise RuntimeError("flash stub - CPU test env")

    sys.modules["sample_efficient_gpt.transformer.ops.triton_flash_attn"].TritonFlashAttnFunc = _Flash
    sys.modules["sample_efficient_gpt.transformer.ops.triton_flash_attn_qknorm"].TritonFlashAttnQKNormFunc = _Flash


def load_pt():
    """Point `sample_efficient_gpt` at the PyTorch tree (the repo root)."""
    _clear_sample_efficient_gpt()
    if PT_REPO not in sys.path:
        sys.path.insert(0, PT_REPO)
    _stub_triton_ops()

    import importlib.metadata as im
    _orig = im.version

    def _version_shim(name: str) -> str:
        try:
            return _orig(name)
        except Exception:
            return "0.0.0"

    im.version = _version_shim
    import importlib
    return importlib.import_module("sample_efficient_gpt.transformer.transformer")
