"""Smoke-test: forward is jittable, grads flow, loss decreases."""
import inspect
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import parity_env
jmod = parity_env.load_jax()
from sample_efficient_gpt.transformer.transformer import Transformer
import jax, jax.numpy as jnp
from flax import nnx
import optax, numpy as np

cfg = dict(n_layers=3, vocab_size=128, d_model=64, n_heads=4, d_ff=256)
model = Transformer(nnx.Rngs(0), **cfg, num_grad_checkpoint_layers=2)
optimizer = nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param)

ids = jnp.asarray(np.random.randint(0, cfg["vocab_size"], (4, 16)), dtype=jnp.int32)


# ---- version-adaptive optimizer.update call --------------------------------
# flax 0.11+/0.12+: nnx.Optimizer.update(model, grads)
# flax 0.10.x    : nnx.Optimizer.update(grads) — model is captured at __init__
#
# Detect once by signature so the jit cache stays clean (a try/except inside
# the jitted function re-traces on every call).
_UPDATE_TAKES_MODEL = "model" in inspect.signature(nnx.Optimizer.update).parameters


def _apply_update(optimizer, model, grads):
    if _UPDATE_TAKES_MODEL:
        optimizer.update(model, grads)
    else:
        optimizer.update(grads)


def loss_fn(model, ids):
    logits = model(ids[:, :-1])
    targets = ids[:, 1:]
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(logp, targets[..., None], axis=-1))


@nnx.jit
def train_step(model, optimizer, ids):
    loss, grads = nnx.value_and_grad(loss_fn)(model, ids)
    _apply_update(optimizer, model, grads)
    return loss


losses = [float(loss_fn(model, ids))]
for _ in range(20):
    losses.append(float(train_step(model, optimizer, ids)))
print("loss trajectory:", [round(l, 4) for l in losses[::2]])
assert losses[-1] < losses[0] * 0.8, "loss didn't decrease enough"
print("TRAINING SMOKE PASSED")
