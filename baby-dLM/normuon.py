"""
normuon.py — MuonAdamW grouped optimizer for the NorMuon training path.

Adapted from Karpathy's autoresearch pretraining script.
Combines Muon (with NorMuon variance reduction) for 2D matrix parameters
and AdamW for embedding / unembedding / scalar parameter groups.

Usage:
    optimizer = setup_normuon_optimizer(
        model, n_embd,
        embedding_lr=0.6, unembedding_lr=0.004,
        matrix_lr=0.04, scalar_lr=0.5,
    )

The four reference LRs follow Karpathy's recipe:
  - embedding_lr, unembedding_lr, scalar_lr are rescaled by 1/sqrt(d_model/768)
  - matrix_lr has no additional width scaling (Muon already normalizes)

External callers (train.py, run_lr_sweep.py) can further multiply these via
adam_mult and matrix_mult sweep factors.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------
# Polar-express coefficients for Newton-Schulz orthogonalization
# ---------------------------------------------------------------

_POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


# ---------------------------------------------------------------
# Step kernels — compiled on CUDA, plain on CPU/MPS
# ---------------------------------------------------------------

def _adamw_step_impl(p, grad, exp_avg, exp_avg_sq,
                     step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


def _muon_step_impl(stacked_grads, stacked_params,
                    momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar express orthogonalization
    # On CPU/MPS bfloat16 may not be supported; use float32 as fallback.
    if stacked_grads.device.type == "cuda":
        X = g.bfloat16()
    else:
        X = g.float()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in _POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in _POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(
        v_mean.to(dtype=second_momentum_buffer.dtype),
        (1 - beta2).to(dtype=second_momentum_buffer.dtype),
    )
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


# Compile only on CUDA; on CPU/MPS use the plain implementations.
_USE_COMPILE = torch.cuda.is_available()
if _USE_COMPILE:
    _adamw_step_fused = torch.compile(
        _adamw_step_impl, dynamic=False, fullgraph=True,
    )
    _muon_step_fused = torch.compile(
        _muon_step_impl, dynamic=False, fullgraph=True,
    )
else:
    _adamw_step_fused = _adamw_step_impl
    _muon_step_fused = _muon_step_impl


# ---------------------------------------------------------------
# MuonAdamW optimizer
# ---------------------------------------------------------------

class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # On CUDA these stay on CPU (torch.compile captures the transfer).
        # On CPU/MPS we lazily move them to the parameter device in _to_dev().
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    @staticmethod
    def _to_dev(t, device):
        """On the compiled path (CUDA), return the CPU tensor as-is.
        On non-compiled paths, move it to the parameter device so that
        MPS / CPU arithmetic doesn't hit a device mismatch."""
        if _USE_COMPILE:
            return t
        return t.to(device, non_blocking=True)

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            dev = p.device
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            _adamw_step_fused(
                p, grad, state['exp_avg'], state['exp_avg_sq'],
                self._to_dev(self._adamw_step_t, dev),
                self._to_dev(self._adamw_lr_t, dev),
                self._to_dev(self._adamw_beta1_t, dev),
                self._to_dev(self._adamw_beta2_t, dev),
                self._to_dev(self._adamw_eps_t, dev),
                self._to_dev(self._adamw_wd_t, dev),
            )

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(
                num_params, *shape, dtype=dtype, device=device
            )
        if "second_momentum_buffer" not in state:
            state_shape = (
                (num_params, shape[-2], 1)
                if shape[-2] >= shape[-1]
                else (num_params, 1, shape[-1])
            )
            state["second_momentum_buffer"] = torch.zeros(
                state_shape, dtype=dtype, device=device
            )

        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(
            group["beta2"] if group["beta2"] is not None else 0.0
        )
        self._muon_lr_t.fill_(
            group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5
        )
        self._muon_wd_t.fill_(group["weight_decay"])

        _muon_step_fused(
            stacked_grads, stacked_params,
            state["momentum_buffer"], state["second_momentum_buffer"],
            self._to_dev(self._muon_momentum_t, device),
            self._to_dev(self._muon_lr_t, device),
            self._to_dev(self._muon_wd_t, device),
            self._to_dev(self._muon_beta2_t, device),
            group["ns_steps"], red_dim,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)


# ---------------------------------------------------------------
# Karpathy-style reference LRs and width scaling
# ---------------------------------------------------------------

# These are the center-of-sweep reference LRs, tuned at d_model=768.
REFERENCE_EMBEDDING_LR = 0.6
REFERENCE_UNEMBEDDING_LR = 0.004
REFERENCE_MATRIX_LR = 0.04
REFERENCE_SCALAR_LR = 0.5

# AdamW betas for embedding / unembedding / scalar groups
ADAM_BETAS = (0.8, 0.95)

# Muon hyperparameters
MUON_MOMENTUM = 0.95
MUON_NS_STEPS = 5
MUON_BETA2 = 0.95


def compute_width_scale(n_embd, reference_dim=768):
    """AdamW LRs scale as 1/sqrt(d_model / reference_dim)."""
    return (n_embd / reference_dim) ** -0.5


def compute_normuon_lrs(n_embd, adam_mult=1.0, matrix_mult=1.0):
    """
    Compute the four per-group learning rates for the NorMuon optimizer.

    Returns a dict with keys: embedding_lr, unembedding_lr, matrix_lr, scalar_lr.

    adam_mult rescales embedding_lr, unembedding_lr, and scalar_lr.
    matrix_mult rescales matrix_lr.
    """
    dmodel_scale = compute_width_scale(n_embd)
    return {
        "embedding_lr":   REFERENCE_EMBEDDING_LR * dmodel_scale * adam_mult,
        "unembedding_lr": REFERENCE_UNEMBEDDING_LR * dmodel_scale * adam_mult,
        "matrix_lr":      REFERENCE_MATRIX_LR * matrix_mult,
        "scalar_lr":      REFERENCE_SCALAR_LR * adam_mult,
    }


# ---------------------------------------------------------------
# Parameter grouping for DiffusionBackbone
# ---------------------------------------------------------------

def setup_normuon_optimizer(
    model,
    n_embd,
    *,
    adam_mult=1.0,
    matrix_mult=1.0,
    weight_decay=0.2,
):
    """
    Build a MuonAdamW optimizer for a DiffusionBackbone model.

    Parameter groups:
      1. Embeddings  (token_emb)     → AdamW with embedding_lr
      2. Unembedding (lm_head)       → AdamW with unembedding_lr
      3. Matrix params (2D in blocks) → Muon with matrix_lr (grouped by shape)
      4. Scalar/vector params (<2D in blocks) → AdamW with scalar_lr

    Returns (optimizer, realized_lrs_dict).
    """
    lrs = compute_normuon_lrs(n_embd, adam_mult=adam_mult, matrix_mult=matrix_mult)

    # Identify parameter groups from the model
    embedding_params = list(model.token_emb.parameters())
    lm_head_params = list(model.lm_head.parameters())

    # Everything in the transformer blocks
    block_params = list(model.blocks.parameters())
    matrix_params = [p for p in block_params if p.ndim >= 2]
    scalar_params = [p for p in block_params if p.ndim < 2]

    # Sanity check: all model params are accounted for
    # (token_emb + lm_head + blocks + buffers/non-params are all there is)
    all_param_ids = {id(p) for p in model.parameters()}
    grouped_ids = (
        {id(p) for p in embedding_params}
        | {id(p) for p in lm_head_params}
        | {id(p) for p in matrix_params}
        | {id(p) for p in scalar_params}
    )
    ungrouped = all_param_ids - grouped_ids
    if ungrouped:
        # Collect any remaining params (e.g. emb_dropout has no params,
        # but future model changes might add new parameter groups).
        extra = [p for p in model.parameters() if id(p) in ungrouped]
        scalar_params.extend(extra)
        print(f"  NorMuon: {len(extra)} extra params added to scalar group")

    # Build param groups
    param_groups = [
        dict(
            kind='adamw', params=lm_head_params,
            lr=lrs["unembedding_lr"], betas=ADAM_BETAS,
            eps=1e-10, weight_decay=0.0,
            group_name='unembedding',
        ),
        dict(
            kind='adamw', params=embedding_params,
            lr=lrs["embedding_lr"], betas=ADAM_BETAS,
            eps=1e-10, weight_decay=0.0,
            group_name='embedding',
        ),
    ]

    if scalar_params:
        param_groups.append(dict(
            kind='adamw', params=scalar_params,
            lr=lrs["scalar_lr"], betas=ADAM_BETAS,
            eps=1e-10, weight_decay=0.0,
            group_name='scalar',
        ))

    # Group matrix params by shape (Muon requires all params in a group
    # to have the same shape for stacking)
    for shape in sorted({p.shape for p in matrix_params}):
        group_params = [p for p in matrix_params if p.shape == shape]
        param_groups.append(dict(
            kind='muon', params=group_params,
            lr=lrs["matrix_lr"],
            momentum=MUON_MOMENTUM, ns_steps=MUON_NS_STEPS,
            beta2=MUON_BETA2, weight_decay=weight_decay,
            group_name=f'muon_{list(shape)}',
        ))

    optimizer = MuonAdamW(param_groups)

    # Store initial LR for schedule scaling
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    # Log the realized LRs
    print(f"  NorMuon optimizer (d_model={n_embd}, "
          f"width_scale={compute_width_scale(n_embd):.4f}):")
    print(f"    embedding_lr   = {lrs['embedding_lr']:.6f}")
    print(f"    unembedding_lr = {lrs['unembedding_lr']:.6f}")
    print(f"    matrix_lr      = {lrs['matrix_lr']:.6f}")
    print(f"    scalar_lr      = {lrs['scalar_lr']:.6f}")
    print(f"    adam_mult={adam_mult}, matrix_mult={matrix_mult}")

    n_embedding = sum(p.numel() for p in embedding_params)
    n_lm_head = sum(p.numel() for p in lm_head_params)
    n_matrix = sum(p.numel() for p in matrix_params)
    n_scalar = sum(p.numel() for p in scalar_params)
    print(f"    params: embedding={n_embedding:,} lm_head={n_lm_head:,} "
          f"matrix={n_matrix:,} scalar={n_scalar:,}")

    return optimizer, lrs


# ---------------------------------------------------------------
# Schedule helpers (matching Karpathy's schedule)
# ---------------------------------------------------------------

def get_muon_momentum(step, warmup_steps=300):
    """Ramp Muon momentum from 0.85 → 0.95 over warmup_steps."""
    frac = min(step / warmup_steps, 1.0)
    return (1 - frac) * 0.85 + frac * MUON_MOMENTUM


def update_normuon_schedule(optimizer, lr_multiplier, step, weight_decay_base=0.2,
                            progress=0.0):
    """
    Update all param group LRs and Muon-specific schedule values.

    - lr_multiplier: from the warmup/warmdown schedule (0→1→0)
    - step: global training step (for momentum warmup)
    - progress: fraction of training done (for weight decay annealing)
    """
    muon_momentum = get_muon_momentum(step)
    muon_wd = weight_decay_base * (1 - progress)

    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lr_multiplier
        if group["kind"] == "muon":
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_wd
