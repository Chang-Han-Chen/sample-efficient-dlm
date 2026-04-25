"""Smoke tests for the JAX AR training stack and NorMuon+AdamW optimizer."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from transformer.transformer import Transformer
from training.optimizer import (
    NormuonAdamWConfig,
    build_param_specs,
    create_normuon_adamw,
)
from training.step import train_step, train_step_accumulated


def _small_model():
    return Transformer(
        nnx.Rngs(0),
        n_layers=2,
        vocab_size=32,
        d_model=64,
        n_heads=4,
        d_ff=128,
        dtype=jnp.float32,
    )


def test_param_grouping():
    model = _small_model()
    specs = build_param_specs(model)
    assert specs["embedding"]["weight"].kind == "adamw"
    assert specs["lm_head"]["weight"].kind == "adamw"
    assert specs["blocks"][0]["ln1"]["gamma"].kind == "adamw"
    assert specs["blocks"][0]["attn"]["W_q"]["weight"].kind == "muon"
    assert specs["blocks"][0]["attn"]["W_k"]["weight"].kind == "muon"
    assert specs["blocks"][0]["attn"]["W_v"]["weight"].kind == "muon"
    assert specs["blocks"][0]["ffn"]["w_down"]["weight"].kind == "muon"
    assert specs["blocks"][0]["ffn"]["w_up_gate"]["weight"].kind == "muon"


def test_normuon_adamw_train_step_decreases_fixed_batch_loss():
    model = _small_model()
    opt_cfg = NormuonAdamWConfig(
        adam_lr=1e-3,
        muon_lr=3e-3,
        muon_weight_decay=0.0,
        warmup_steps=1,
    )
    optimizer = nnx.Optimizer(model, create_normuon_adamw(model, opt_cfg), wrt=nnx.Param)

    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 32, size=(4, 17), dtype=np.int32)
    inputs = jnp.asarray(tokens[:, :-1], dtype=jnp.int32)
    targets = jnp.asarray(tokens[:, 1:], dtype=jnp.int32)

    losses = []
    for _ in range(12):
        metrics = train_step(model, optimizer, inputs, targets, 1e-4, 1.0)
        losses.append(float(jax.block_until_ready(metrics["loss"])))

    assert losses[-1] < losses[0] * 0.6, losses


def test_gradient_accumulation_train_step_decreases_fixed_batch_loss():
    model = _small_model()
    opt_cfg = NormuonAdamWConfig(
        adam_lr=1e-3,
        muon_lr=3e-3,
        muon_weight_decay=0.0,
        warmup_steps=1,
    )
    optimizer = nnx.Optimizer(model, create_normuon_adamw(model, opt_cfg), wrt=nnx.Param)

    rng = np.random.default_rng(1)
    tokens = rng.integers(0, 32, size=(3, 2, 17), dtype=np.int32)
    inputs = jnp.asarray(tokens[:, :, :-1], dtype=jnp.int32)
    targets = jnp.asarray(tokens[:, :, 1:], dtype=jnp.int32)

    losses = []
    for _ in range(8):
        metrics = train_step_accumulated(model, optimizer, inputs, targets, 1e-4, 1.0)
        losses.append(float(jax.block_until_ready(metrics["loss"])))

    assert losses[-1] < losses[0] * 0.8, losses


if __name__ == "__main__":
    test_param_grouping()
    test_normuon_adamw_train_step_decreases_fixed_batch_loss()
    test_gradient_accumulation_train_step_decreases_fixed_batch_loss()
    print("TRAINING STACK TESTS PASSED")
