"""Focused tests for sparse Switch MoE routing and integration."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from transformer.moe import SwitchMoE
from transformer.transformer import Transformer, has_moe_layer, has_value_embedding_layer
from training.loss import ar_loss, supervised_lm_loss, supervised_lm_loss_sums
from training.optimizer import NormuonAdamWConfig, build_param_specs, create_normuon_adamw


def _tree_norm(tree) -> float:
    leaves = jax.tree_util.tree_leaves(nnx.as_pure(tree))
    total = sum(jnp.sum(jnp.square(x.astype(jnp.float32))) for x in leaves)
    return float(jnp.sqrt(total))


def _flat_grad_norm(grads, needle: str) -> float:
    total = jnp.zeros((), dtype=jnp.float32)
    for path, value in nnx.to_flat_state(grads):
        path_str = ".".join(str(part) for part in path)
        if needle in path_str:
            total = total + jnp.sum(jnp.square(value[...].astype(jnp.float32)))
    return float(jnp.sqrt(total))


def test_moe_and_value_embedding_placement_match():
    assert [i for i in range(8) if has_value_embedding_layer(i, 8, "alternating")] == [1, 3, 5, 7]
    assert [i for i in range(8) if has_moe_layer(i, 8, "alternating")] == [1, 3, 5, 7]
    assert [i for i in range(8) if has_moe_layer(i, 8, [1, 3, 5, 7])] == [1, 3, 5, 7]
    assert [i for i in range(8) if has_moe_layer(i, 8, "alternating_early")] == [0, 2, 4, 6]


def test_switch_moe_forward_aux_and_drops_are_finite():
    moe = SwitchMoE(
        nnx.Rngs(0),
        d_model=16,
        d_ff=64,
        num_experts=2,
        capacity_factor=0.1,
        dtype=jnp.float32,
    )
    x = jnp.asarray(np.random.default_rng(0).normal(size=(2, 8, 16)).astype(np.float32))
    y, aux = moe(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert float(aux.dropped_fraction) > 0.0
    for value in aux:
        assert bool(jnp.isfinite(value))


def test_switch_moe_router_input_can_be_split_from_expert_input():
    moe = SwitchMoE(
        nnx.Rngs(7),
        d_model=16,
        d_ff=64,
        num_experts=4,
        capacity_factor=2.0,
        dtype=jnp.float32,
    )
    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.normal(size=(2, 5, 16)).astype(np.float32))
    y_default, aux_default = moe(x)
    y_explicit, aux_explicit = moe(x, router_x=x)
    np.testing.assert_allclose(np.asarray(y_default), np.asarray(y_explicit))
    for got, expected in zip(aux_default, aux_explicit, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected))

    expert_x = jnp.zeros_like(x)
    router_x = x
    y_zero_router, aux_zero_router = moe(expert_x)
    y_split_router, aux_split_router = moe(expert_x, router_x=router_x)
    np.testing.assert_allclose(np.asarray(y_zero_router), np.zeros_like(np.asarray(y_zero_router)))
    np.testing.assert_allclose(np.asarray(y_split_router), np.zeros_like(np.asarray(y_split_router)))
    assert not np.isclose(
        float(aux_zero_router.router_z_loss),
        float(aux_split_router.router_z_loss),
    )


def test_moe_layernorm_scaling_is_split_between_router_and_experts():
    model = Transformer(
        nnx.Rngs(8),
        n_layers=2,
        vocab_size=32,
        d_model=32,
        n_heads=4,
        d_ff=64,
        moe=True,
        moe_layers="all",
        moe_num_experts=2,
        layernorm_scaling=True,
        dtype=jnp.float32,
    )
    moe_block = model.blocks[1]
    expected = 1.0 / np.sqrt(2.0)
    np.testing.assert_allclose(moe_block.ln1.depth_scaling, expected, rtol=1e-6)
    np.testing.assert_allclose(moe_block.ln2.depth_scaling, 1.0, rtol=1e-6)
    np.testing.assert_allclose(moe_block.moe_expert_input_scale, expected, rtol=1e-6)

    dense = Transformer(
        nnx.Rngs(8),
        n_layers=2,
        vocab_size=32,
        d_model=32,
        n_heads=4,
        d_ff=64,
        moe=False,
        layernorm_scaling=True,
        dtype=jnp.float32,
    )
    dense_block = dense.blocks[1]
    np.testing.assert_allclose(dense_block.ln1.depth_scaling, expected, rtol=1e-6)
    np.testing.assert_allclose(dense_block.ln2.depth_scaling, expected, rtol=1e-6)
    np.testing.assert_allclose(dense_block.moe_expert_input_scale, 1.0, rtol=1e-6)


def test_moe_layernorm_scaling_can_use_old_shared_router_input():
    model = Transformer(
        nnx.Rngs(9),
        n_layers=2,
        vocab_size=32,
        d_model=32,
        n_heads=4,
        d_ff=64,
        moe=True,
        moe_layers="all",
        moe_num_experts=2,
        moe_split_router_input=False,
        layernorm_scaling=True,
        dtype=jnp.float32,
    )
    moe_block = model.blocks[1]
    expected = 1.0 / np.sqrt(2.0)
    assert not moe_block.moe_split_router_input
    np.testing.assert_allclose(moe_block.ln1.depth_scaling, expected, rtol=1e-6)
    np.testing.assert_allclose(moe_block.ln2.depth_scaling, expected, rtol=1e-6)
    np.testing.assert_allclose(moe_block.moe_expert_input_scale, 1.0, rtol=1e-6)


def test_switch_moe_router_and_expert_gradients_are_nonzero():
    moe = SwitchMoE(
        nnx.Rngs(1),
        d_model=16,
        d_ff=64,
        num_experts=3,
        capacity_factor=2.0,
        use_router_prob=True,
        dtype=jnp.float32,
    )
    x = jnp.asarray(np.random.default_rng(1).normal(size=(2, 6, 16)).astype(np.float32))

    def loss_fn(module, x):
        y, aux = module(x)
        return jnp.mean(y * y) + 0.01 * aux.load_balance_loss + 0.001 * aux.router_z_loss

    _, grads = nnx.value_and_grad(loss_fn)(moe, x)
    assert _flat_grad_norm(grads, "router.weight") > 0.0
    assert _flat_grad_norm(grads, "experts") > 0.0


def test_router_probability_gives_supervised_router_gradient_path():
    gated = SwitchMoE(
        nnx.Rngs(2),
        d_model=16,
        d_ff=64,
        num_experts=2,
        capacity_factor=2.0,
        use_router_prob=True,
        dtype=jnp.float32,
    )
    ungated = SwitchMoE(
        nnx.Rngs(2),
        d_model=16,
        d_ff=64,
        num_experts=2,
        capacity_factor=2.0,
        use_router_prob=False,
        dtype=jnp.float32,
    )
    x = jnp.asarray(np.random.default_rng(2).normal(size=(2, 5, 16)).astype(np.float32))

    def supervised_loss(module, x):
        y, _ = module(x)
        return jnp.mean(y * y)

    _, gated_grads = nnx.value_and_grad(supervised_loss)(gated, x)
    _, ungated_grads = nnx.value_and_grad(supervised_loss)(ungated, x)
    assert _flat_grad_norm(gated_grads, "router.weight") > 0.0
    assert _flat_grad_norm(ungated_grads, "router.weight") == 0.0


def test_transformer_moe_aux_api_and_remat_path():
    cfg = dict(
        n_layers=3,
        vocab_size=32,
        d_model=32,
        n_heads=4,
        d_ff=64,
        moe=True,
        moe_layers="alternating",
        moe_num_experts=2,
        dtype=jnp.float32,
    )
    ids = jnp.asarray(np.random.default_rng(3).integers(0, 32, size=(2, 7)), dtype=jnp.int32)
    dense = Transformer(nnx.Rngs(0), **{k: v for k, v in cfg.items() if not k.startswith("moe")})
    logits = dense(ids)
    assert logits.shape == (2, 7, 32)

    model = Transformer(nnx.Rngs(0), **cfg, num_grad_checkpoint_layers=1)
    logits, aux = model(ids, return_aux=True)
    hidden, aux_hidden = model(ids, return_hidden=True, return_aux=True)
    assert logits.shape == (2, 7, 32)
    assert hidden.shape == (2, 7, 32)
    assert float(aux.num_moe_layers) == 2.0
    np.testing.assert_allclose(np.asarray(aux_hidden.num_moe_layers), np.asarray(aux.num_moe_layers))


def test_moe_losses_are_finite_full_and_chunked():
    model = Transformer(
        nnx.Rngs(4),
        n_layers=2,
        vocab_size=32,
        d_model=32,
        n_heads=4,
        d_ff=64,
        moe=True,
        moe_layers="all",
        moe_num_experts=2,
        dtype=jnp.float32,
    )
    rng = np.random.default_rng(4)
    tokens = rng.integers(0, 32, size=(2, 9), dtype=np.int32)
    inputs = jnp.asarray(tokens[:, :-1], dtype=jnp.int32)
    targets = jnp.asarray(tokens[:, 1:], dtype=jnp.int32)

    for loss_impl in ("full", "chunked"):
        total, metrics = ar_loss(
            model,
            inputs,
            targets,
            z_loss_weight=1e-4,
            loss_impl=loss_impl,
            logit_chunk_size=5,
            moe_load_balance_loss_weight=0.01,
            moe_router_z_loss_weight=0.001,
        )
        assert bool(jnp.isfinite(total))
        assert float(metrics["moe_num_layers"]) == 2.0
        assert float(metrics["moe_aux_loss"]) > 0.0


def test_supervised_moe_aux_denominator_exactness():
    model = Transformer(
        nnx.Rngs(5),
        n_layers=2,
        vocab_size=17,
        d_model=32,
        n_heads=4,
        d_ff=64,
        moe=True,
        moe_layers="all",
        moe_num_experts=2,
        dtype=jnp.float32,
        is_causal=False,
    )
    rng = np.random.default_rng(5)
    inputs = jnp.asarray(rng.integers(0, 17, size=(2, 6)), dtype=jnp.int32)
    targets = jnp.asarray(rng.integers(0, 17, size=(2, 6)), dtype=jnp.int32)
    mask = jnp.asarray([[True, False, True, False, False, False], [False, True, False, False, False, True]])
    denominator = jnp.asarray(9.0, dtype=jnp.float32)
    lb_w = 0.01
    rz_w = 0.001

    total_sum, metrics = supervised_lm_loss_sums(
        model,
        inputs,
        targets,
        mask,
        z_loss_weight=0.0,
        loss_denominator=denominator,
        moe_load_balance_loss_weight=lb_w,
        moe_router_z_loss_weight=rz_w,
    )
    _, mean_metrics = supervised_lm_loss(
        model,
        inputs,
        targets,
        mask,
        z_loss_weight=0.0,
        moe_load_balance_loss_weight=lb_w,
        moe_router_z_loss_weight=rz_w,
    )
    expected_aux_sum = denominator * (
        lb_w * mean_metrics["moe_load_balance_loss"]
        + rz_w * mean_metrics["moe_router_z_loss"]
    )
    np.testing.assert_allclose(
        np.asarray(metrics["moe_aux_loss_sum"]),
        np.asarray(expected_aux_sum),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(total_sum),
        np.asarray(metrics["loss_sum"] + expected_aux_sum),
        atol=1e-6,
        rtol=1e-6,
    )


def test_router_weights_use_router_adam_and_experts_remain_muon():
    model = Transformer(
        nnx.Rngs(6),
        n_layers=2,
        vocab_size=32,
        d_model=32,
        n_heads=4,
        d_ff=64,
        moe=True,
        moe_layers="all",
        moe_num_experts=2,
        dtype=jnp.float32,
    )
    specs = build_param_specs(model)
    assert specs["blocks"][0]["ffn"]["router"]["weight"].kind == "adam_router"
    assert specs["blocks"][0]["ffn"]["experts"][0]["w_up_gate"]["weight"].kind == "muon"

    tx = create_normuon_adamw(
        model,
        NormuonAdamWConfig(
            table_adam_lr=1e-3,
            scalar_adam_lr=1e-3,
            router_adam_lr=2e-3,
            muon_lr=1e-3,
            adam_weight_decay=0.0,
            router_adam_weight_decay=0.0,
            muon_weight_decay=0.0,
            scheduler="constant",
        ),
    )
    params_state = nnx.state(model, nnx.Param)
    params = nnx.as_pure(params_state)
    opt_state = tx.init(params_state)
    grads = jax.tree_util.tree_map(lambda p: jnp.ones_like(p) * 0.01, params)
    updates, _ = tx.update(grads, opt_state, params)
    router_update = updates["blocks"][0]["ffn"]["router"]["weight"]
    np.testing.assert_allclose(np.asarray(router_update), -2e-3, atol=1e-6, rtol=1e-5)
