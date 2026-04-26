"""Smoke tests for the JAX AR training stack and NorMuon+AdamW optimizer."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from transformer.core import ValueEmbedding
from transformer.transformer import Transformer
from training.optimizer import (
    NormuonAdamWConfig,
    build_param_specs,
    create_normuon_adamw,
)
from training.step import train_step, train_step_accumulated
from training.loss import (
    ar_loss,
    cross_entropy_with_z_loss,
    linear_cross_entropy_with_z_loss_chunked,
)
from train_ar import restore_training_checkpoint, save_training_checkpoint


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
    assert specs["embedding"]["weight"].kind == "adam_table"
    assert specs["lm_head"]["weight"].kind == "adam_table"
    assert specs["blocks"][0]["ln1"]["gamma"].kind == "adam_scalar"
    assert specs["blocks"][0]["attn"]["W_q"]["weight"].kind == "muon"
    assert specs["blocks"][0]["attn"]["W_k"]["weight"].kind == "muon"
    assert specs["blocks"][0]["attn"]["W_v"]["weight"].kind == "muon"
    assert specs["blocks"][0]["ffn"]["w_down"]["weight"].kind == "muon"
    assert specs["blocks"][0]["ffn"]["w_up_gate"]["weight"].kind == "muon"

    ve_model = Transformer(
        nnx.Rngs(1),
        n_layers=2,
        vocab_size=32,
        d_model=64,
        n_heads=4,
        d_ff=128,
        value_embedding=True,
        dtype=jnp.float32,
    )
    ve_specs = build_param_specs(ve_model)
    assert ve_specs["blocks"][1]["attn"]["value_embedding_table"]["weight"].kind == "adam_ve_table"
    assert ve_specs["blocks"][1]["attn"]["value_embedding_gate"]["weight"].kind == "muon"
    assert ve_specs["blocks"][1]["attn"]["value_embedding_gain"].kind == "adam_scalar"

    split_ve_model = Transformer(
        nnx.Rngs(11),
        n_layers=2,
        vocab_size=33,
        d_model=64,
        n_heads=4,
        d_ff=128,
        value_embedding=True,
        value_embedding_split_token_id=32,
        dtype=jnp.float32,
    )
    split_ve_specs = build_param_specs(split_ve_model)
    assert split_ve_specs["blocks"][1]["attn"]["value_embedding_table"]["weight"].kind == "adam_ve_table"
    assert split_ve_specs["blocks"][1]["attn"]["value_embedding_table"]["split_weight"].kind == "adam_ve_mask"


def test_value_embedding_split_token_can_be_zero_without_param():
    ve = ValueEmbedding(
        nnx.Rngs(12),
        vocab_size=8,
        n_kv_heads=2,
        head_dim=4,
        d_model=16,
        split_token_id=7,
        split_token_zero=True,
        dtype=jnp.float32,
    )
    token_ids = jnp.asarray([[1, 7, 2]], dtype=jnp.int32)
    values = ve(token_ids)
    np.testing.assert_allclose(np.asarray(values[:, 1]), 0.0, atol=0.0)
    assert float(jnp.linalg.norm(values[:, 0])) > 0.0

    model = Transformer(
        nnx.Rngs(13),
        n_layers=2,
        vocab_size=33,
        d_model=64,
        n_heads=4,
        d_ff=128,
        value_embedding=True,
        value_embedding_split_token_id=32,
        value_embedding_split_token_zero=True,
        dtype=jnp.float32,
    )
    param_paths = {
        ".".join(str(part) for part in path)
        for path, _ in nnx.to_flat_state(nnx.state(model, nnx.Param))
    }
    assert "blocks.1.attn.value_embedding_table.weight" in param_paths
    assert "blocks.1.attn.value_embedding_table.split_weight" not in param_paths


def test_value_embedding_mask_uses_separate_adam_betas():
    model = Transformer(
        nnx.Rngs(2),
        n_layers=2,
        vocab_size=33,
        d_model=64,
        n_heads=4,
        d_ff=128,
        value_embedding=True,
        value_embedding_split_token_id=32,
        dtype=jnp.float32,
    )
    tx = create_normuon_adamw(
        model,
        NormuonAdamWConfig(
            table_adam_lr=1e-3,
            scalar_adam_lr=1e-3,
            muon_lr=1e-3,
            adam_betas=(0.95, 0.99),
            value_embedding_adam_betas=(0.8, 0.95),
            adam_weight_decay=0.0,
            muon_weight_decay=0.0,
            scheduler="constant",
        ),
    )
    params_state = nnx.state(model, nnx.Param)
    params = nnx.as_pure(params_state)
    opt_state = tx.init(params_state)
    grads = jax.tree_util.tree_map(lambda p: jnp.ones_like(p) * 0.01, params)
    _, new_state = tx.update(grads, opt_state, params)

    embedding_m = new_state.adam_m["embedding"]["weight"]
    ve_m = new_state.adam_m["blocks"][1]["attn"]["value_embedding_table"]["split_weight"]
    np.testing.assert_allclose(np.asarray(embedding_m), 0.0005, atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(ve_m), 0.002, atol=1e-8, rtol=1e-6)


def test_value_embedding_mask_uses_separate_adam_lr():
    model = Transformer(
        nnx.Rngs(3),
        n_layers=2,
        vocab_size=33,
        d_model=64,
        n_heads=4,
        d_ff=128,
        value_embedding=True,
        value_embedding_split_token_id=32,
        dtype=jnp.float32,
    )
    tx = create_normuon_adamw(
        model,
        NormuonAdamWConfig(
            table_adam_lr=1e-3,
            value_embedding_mask_adam_lr=2e-3,
            scalar_adam_lr=1e-3,
            muon_lr=1e-3,
            adam_betas=(0.95, 0.99),
            value_embedding_adam_betas=(0.95, 0.99),
            adam_weight_decay=0.0,
            muon_weight_decay=0.0,
            scheduler="constant",
        ),
    )
    params_state = nnx.state(model, nnx.Param)
    params = nnx.as_pure(params_state)
    opt_state = tx.init(params_state)
    grads = jax.tree_util.tree_map(lambda p: jnp.ones_like(p) * 0.01, params)
    updates, _ = tx.update(grads, opt_state, params)

    embedding_update = updates["embedding"]["weight"]
    ve_update = updates["blocks"][1]["attn"]["value_embedding_table"]["split_weight"]
    np.testing.assert_allclose(np.asarray(embedding_update), -1e-3, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(ve_update), -2e-3, atol=1e-6, rtol=1e-5)


def test_normuon_adamw_update_exercises_both_optimizer_paths():
    model = _small_model()
    tx = create_normuon_adamw(
        model,
        NormuonAdamWConfig(
            table_adam_lr=1e-3,
            scalar_adam_lr=1e-3,
            muon_lr=1e-3,
            adam_weight_decay=0.0,
            muon_weight_decay=0.0,
            scheduler="constant",
        ),
    )
    params_state = nnx.state(model, nnx.Param)
    params = nnx.as_pure(params_state)
    opt_state = tx.init(params_state)
    grads = jax.tree_util.tree_map(lambda p: jnp.ones_like(p) * 0.01, params)
    updates, new_state = tx.update(grads, opt_state, params)
    updates_by_path = {
        ".".join(str(part) for part in path): update
        for path, update in nnx.to_flat_state(nnx.State(updates))
    }

    assert int(new_state.count) == int(opt_state.count) + 1
    assert float(jnp.linalg.norm(updates_by_path["blocks.0.attn.W_q.weight"].astype(jnp.float32))) > 0.0
    assert float(jnp.linalg.norm(updates_by_path["blocks.0.attn.W_k.weight"].astype(jnp.float32))) > 0.0
    assert float(jnp.linalg.norm(updates_by_path["blocks.0.attn.W_v.weight"].astype(jnp.float32))) > 0.0
    assert float(jnp.linalg.norm(updates_by_path["embedding.weight"].astype(jnp.float32))) > 0.0
    assert float(jnp.linalg.norm(updates_by_path["blocks.0.ln1.gamma"].astype(jnp.float32))) > 0.0


def test_normuon_adamw_train_step_decreases_fixed_batch_loss():
    model = _small_model()
    opt_cfg = NormuonAdamWConfig(
        table_adam_lr=1e-3,
        scalar_adam_lr=1e-3,
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
        table_adam_lr=1e-3,
        scalar_adam_lr=1e-3,
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


def test_chunked_linear_ce_matches_full_value_and_grad():
    rng = jax.random.PRNGKey(0)
    hidden = jax.random.normal(rng, (2, 5, 8), dtype=jnp.float32)
    weight = jax.random.normal(jax.random.fold_in(rng, 1), (17, 8), dtype=jnp.float32)
    targets = jax.random.randint(jax.random.fold_in(rng, 2), (2, 5), 0, 17)

    def full_fn(h, w):
        logits = h @ w.T
        loss, z_loss, _ = cross_entropy_with_z_loss(logits, targets)
        return loss + 1e-4 * z_loss

    def chunked_fn(h, w):
        loss, z_loss = linear_cross_entropy_with_z_loss_chunked(
            h,
            w,
            targets,
            chunk_size=3,
        )
        return loss + 1e-4 * z_loss

    full_value, full_grads = jax.value_and_grad(full_fn, argnums=(0, 1))(hidden, weight)
    chunk_value, chunk_grads = jax.value_and_grad(chunked_fn, argnums=(0, 1))(hidden, weight)

    np.testing.assert_allclose(np.asarray(chunk_value), np.asarray(full_value), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(chunk_grads[0]), np.asarray(full_grads[0]), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(chunk_grads[1]), np.asarray(full_grads[1]), atol=1e-6, rtol=1e-6)


def test_chunked_ar_loss_matches_full_loss():
    model = _small_model()
    rng = np.random.default_rng(2)
    tokens = rng.integers(0, 32, size=(4, 17), dtype=np.int32)
    inputs = jnp.asarray(tokens[:, :-1], dtype=jnp.int32)
    targets = jnp.asarray(tokens[:, 1:], dtype=jnp.int32)

    full_total, full_metrics = ar_loss(
        model,
        inputs,
        targets,
        z_loss_weight=1e-4,
        loss_impl="full",
    )
    chunk_total, chunk_metrics = ar_loss(
        model,
        inputs,
        targets,
        z_loss_weight=1e-4,
        loss_impl="chunked",
        logit_chunk_size=7,
    )

    np.testing.assert_allclose(np.asarray(chunk_total), np.asarray(full_total), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(chunk_metrics["loss"]),
        np.asarray(full_metrics["loss"]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_training_checkpoint_roundtrip_restores_model_and_optimizer():
    model = _small_model()
    opt_cfg = NormuonAdamWConfig(
        table_adam_lr=1e-3,
        scalar_adam_lr=1e-3,
        muon_lr=3e-3,
        muon_weight_decay=0.0,
        warmup_steps=1,
    )
    optimizer = nnx.Optimizer(model, create_normuon_adamw(model, opt_cfg), wrt=nnx.Param)

    rng = np.random.default_rng(3)
    tokens = rng.integers(0, 32, size=(4, 17), dtype=np.int32)
    inputs = jnp.asarray(tokens[:, :-1], dtype=jnp.int32)
    targets = jnp.asarray(tokens[:, 1:], dtype=jnp.int32)
    metrics = train_step(model, optimizer, inputs, targets, 1e-4, 1.0)
    metrics = {key: float(jax.block_until_ready(value)) for key, value in metrics.items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "ckpt"
        metadata = save_training_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            step=7,
            kind="test",
            metrics=metrics,
            resolved_config={"test": True},
            rng_state=rng.bit_generator.state,
        )

        restored_model = _small_model()
        restored_optimizer = nnx.Optimizer(
            restored_model,
            create_normuon_adamw(restored_model, opt_cfg),
            wrt=nnx.Param,
        )
        restored_metadata = restore_training_checkpoint(
            checkpoint_path,
            restored_model,
            restored_optimizer,
        )
        assert restored_metadata["step"] == 7
        assert restored_metadata["state_files"]["model"]["sha256"] == metadata["state_files"]["model"]["sha256"]

    for (path_a, value_a), (path_b, value_b) in zip(
        nnx.to_flat_state(nnx.state(model, nnx.Param)),
        nnx.to_flat_state(nnx.state(restored_model, nnx.Param)),
        strict=True,
    ):
        assert path_a == path_b
        np.testing.assert_array_equal(np.asarray(value_a[...]), np.asarray(value_b[...]))

    optimizer_leaves = jax.tree_util.tree_leaves(nnx.to_pure_dict(nnx.state(optimizer)))
    restored_optimizer_leaves = jax.tree_util.tree_leaves(nnx.to_pure_dict(nnx.state(restored_optimizer)))
    assert len(optimizer_leaves) == len(restored_optimizer_leaves)
    for value_a, value_b in zip(optimizer_leaves, restored_optimizer_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(value_a), np.asarray(value_b))


if __name__ == "__main__":
    test_param_grouping()
    test_value_embedding_table_uses_separate_adam_betas()
    test_normuon_adamw_train_step_decreases_fixed_batch_loss()
    test_gradient_accumulation_train_step_decreases_fixed_batch_loss()
    test_chunked_linear_ce_matches_full_value_and_grad()
    test_chunked_ar_loss_matches_full_loss()
    test_training_checkpoint_roundtrip_restores_model_and_optimizer()
    print("TRAINING STACK TESTS PASSED")
