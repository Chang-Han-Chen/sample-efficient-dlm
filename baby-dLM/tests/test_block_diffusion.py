"""
test_block_diffusion.py — Tests for block diffusion (BD3-LM) model.

Covers:
  - DiffusionBackbone forward paths (single-block & multi-block)
  - block_utils mask functions
  - model_bd3lm: get_batch, compute_loss, generate_from
  - The clean-target leakage fix (P1 bug regression test)
"""

import math
import pytest
import torch

import backbone
from backbone import DiffusionBackbone
from block_utils import (
    bd3_train_mask_special_cases_ok,
    block_causal_equals_causal_when_block_len_is_one,
    make_bd3_train_mask,
    make_block_causal_mask,
    num_blocks,
    validate_block_len,
)


# =====================================================================
# Fixtures
# =====================================================================

VOCAB_SIZE = 67
BLOCK_SIZE = 32
BLOCK_LEN = 8  # 4 blocks
BATCH_SIZE = 4
N_EMBD = 64
N_HEAD = 4
N_LAYER = 2
HEAD_DIM = N_EMBD // N_HEAD
T = 10
MASK_TOKEN_ID = 0


@pytest.fixture
def rng():
    torch.manual_seed(42)
    return torch.Generator().manual_seed(42)


@pytest.fixture
def cfg():
    """Minimal cfg dict matching what block models expect."""
    data = torch.randint(1, VOCAB_SIZE, (2000,))
    split = int(0.9 * len(data))

    def survival_prob_tensor(t_steps):
        t_frac = t_steps.float() / T
        return (1.0 - t_frac).clamp(0.0, 1.0)

    return {
        "train_data": data[:split],
        "val_data": data[split:],
        "batch_size": BATCH_SIZE,
        "block_size": BLOCK_SIZE,
        "block_len": BLOCK_LEN,
        "T": T,
        "mask_token_id": MASK_TOKEN_ID,
        "vocab_size": VOCAB_SIZE,
        "device": "cpu",
        "survival_prob_tensor": survival_prob_tensor,
    }


@pytest.fixture
def single_block_model():
    return DiffusionBackbone(
        vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
        n_layer=N_LAYER, head_dim=HEAD_DIM, block_size=BLOCK_SIZE,
        dropout=0.0,
    )


@pytest.fixture
def multi_block_model():
    return DiffusionBackbone(
        vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
        n_layer=N_LAYER, head_dim=HEAD_DIM, block_size=BLOCK_SIZE,
        block_len=BLOCK_LEN, dropout=0.0,
    )


def survival_prob_scalar(t_step):
    t_frac = float(t_step) / T
    return max(0.0, min(1.0, 1.0 - t_frac))


def make_decode():
    itos = {i: chr(65 + (i % 26)) for i in range(VOCAB_SIZE)}
    def decode(l):
        return "".join([itos[i] for i in l])
    return decode


# =====================================================================
# block_utils tests
# =====================================================================


class TestValidateBlockLen:
    def test_valid(self):
        validate_block_len(32, 8)
        validate_block_len(32, 32)
        validate_block_len(32, 1)

    def test_not_divisible(self):
        with pytest.raises(ValueError):
            validate_block_len(32, 7)

    def test_zero(self):
        with pytest.raises(ValueError):
            validate_block_len(32, 0)


class TestNumBlocks:
    def test_basic(self):
        assert num_blocks(32, 8) == 4
        assert num_blocks(32, 32) == 1
        assert num_blocks(32, 1) == 32


class TestMakeBD3TrainMask:
    def test_shape(self):
        mask = make_bd3_train_mask(BLOCK_SIZE, BLOCK_LEN)
        assert mask.shape == (1, 1, 2 * BLOCK_SIZE, 2 * BLOCK_SIZE)
        assert mask.dtype == torch.bool

    def test_x0_never_sees_xt(self):
        """Bottom-left quadrant of the 2L×2L mask must be all-False."""
        mask = make_bd3_train_mask(BLOCK_SIZE, BLOCK_LEN)[0, 0]
        L = BLOCK_SIZE
        assert not mask[L:, :L].any(), "x0 stream should never attend to x_t stream"

    def test_block_diagonal_xt(self):
        """x_t tokens in block b should attend to all x_t tokens in block b."""
        mask = make_bd3_train_mask(BLOCK_SIZE, BLOCK_LEN)[0, 0]
        for b in range(BLOCK_SIZE // BLOCK_LEN):
            start, end = b * BLOCK_LEN, (b + 1) * BLOCK_LEN
            block_mask = mask[start:end, start:end]
            assert block_mask.all(), f"Block {b} x_t should be fully connected"

    def test_xt_does_not_see_own_x0(self):
        """x_t in block b should NOT see x_0 in block b (M_OBC is strict)."""
        mask = make_bd3_train_mask(BLOCK_SIZE, BLOCK_LEN)[0, 0]
        L = BLOCK_SIZE
        for b in range(L // BLOCK_LEN):
            start, end = b * BLOCK_LEN, (b + 1) * BLOCK_LEN
            cross = mask[start:end, L + start:L + end]
            assert not cross.any(), f"Block {b}: x_t should not see own x_0"

    def test_xt_sees_earlier_x0(self):
        """x_t in block b should see x_0 in blocks < b."""
        mask = make_bd3_train_mask(BLOCK_SIZE, BLOCK_LEN)[0, 0]
        L = BLOCK_SIZE
        nblk = L // BLOCK_LEN
        for b in range(1, nblk):
            q_start = b * BLOCK_LEN
            # Should see all x_0 tokens from block 0
            assert mask[q_start, L + 0], f"Block {b} x_t should see block 0 x_0"

    def test_special_cases(self):
        assert bd3_train_mask_special_cases_ok(8, 1)
        assert bd3_train_mask_special_cases_ok(8, 8)
        assert bd3_train_mask_special_cases_ok(16, 4)


class TestMakeBlockCausalMask:
    def test_shape(self):
        mask = make_block_causal_mask(BLOCK_SIZE, BLOCK_LEN)
        assert mask.shape == (1, 1, BLOCK_SIZE, BLOCK_SIZE)

    def test_within_block(self):
        """Tokens within the same block should attend to each other."""
        mask = make_block_causal_mask(BLOCK_SIZE, BLOCK_LEN)[0, 0]
        for b in range(BLOCK_SIZE // BLOCK_LEN):
            start, end = b * BLOCK_LEN, (b + 1) * BLOCK_LEN
            assert mask[start:end, start:end].all()

    def test_later_block_sees_earlier(self):
        """Block 1 tokens should see block 0 tokens."""
        mask = make_block_causal_mask(BLOCK_SIZE, BLOCK_LEN)[0, 0]
        assert mask[BLOCK_LEN, 0]

    def test_earlier_block_not_see_later(self):
        """Block 0 tokens should NOT see block 1 tokens."""
        mask = make_block_causal_mask(BLOCK_SIZE, BLOCK_LEN)[0, 0]
        assert not mask[0, BLOCK_LEN]

    def test_len_one_equals_causal(self):
        assert block_causal_equals_causal_when_block_len_is_one(16)

    def test_matches_x0_stream_of_train_mask(self):
        """The sampling mask should equal the x0-stream (bottom-right) of the training mask."""
        L = 12
        bl = 3
        train = make_bd3_train_mask(L, bl)[0, 0]
        sample = make_block_causal_mask(L, bl)[0, 0]
        assert torch.equal(train[L:, L:], sample)


# =====================================================================
# DiffusionBackbone tests
# =====================================================================


class TestBackboneSingleBlock:
    """Tests for block_len == block_size (standard diffusion mode)."""

    def test_is_single_block_flag(self, single_block_model):
        assert single_block_model._is_single_block

    def test_masks_are_none(self, single_block_model):
        assert single_block_model.train_attn_mask is None
        assert single_block_model.sample_attn_mask is None

    def test_forward_logits_shape(self, single_block_model):
        x = torch.randint(0, VOCAB_SIZE, (2, BLOCK_SIZE))
        logits, loss = single_block_model(x)
        assert logits.shape == (2, BLOCK_SIZE, VOCAB_SIZE)
        assert loss is None

    def test_forward_with_loss(self, single_block_model):
        x = torch.randint(0, VOCAB_SIZE, (2, BLOCK_SIZE))
        mask = torch.rand(2, BLOCK_SIZE) > 0.5
        logits, loss = single_block_model(x, targets=x, supervise_mask=mask)
        assert loss is not None
        assert loss.shape == ()

    def test_forward_requires_mask_with_targets(self, single_block_model):
        x = torch.randint(0, VOCAB_SIZE, (2, BLOCK_SIZE))
        with pytest.raises(ValueError, match="mask or supervise_mask"):
            single_block_model(x, targets=x)

    def test_forward_train_no_leakage(self, single_block_model):
        """Regression test for P1 bug: single-block forward_train must not leak x_0."""
        single_block_model.eval()
        xt = torch.randint(0, VOCAB_SIZE, (1, BLOCK_SIZE))
        x0_a = torch.randint(0, VOCAB_SIZE, (1, BLOCK_SIZE))
        x0_b = torch.randint(0, VOCAB_SIZE, (1, BLOCK_SIZE))

        with torch.no_grad():
            logits_a, _ = single_block_model.forward_train(xt, x0_a)
            logits_b, _ = single_block_model.forward_train(xt, x0_b)

        assert torch.allclose(logits_a, logits_b, atol=1e-6), \
            "Single-block forward_train logits must be independent of x_0"

    def test_forward_sample_shape(self, single_block_model):
        x = torch.randint(0, VOCAB_SIZE, (2, BLOCK_SIZE))
        logits, loss = single_block_model.forward_sample(x)
        assert logits.shape == (2, BLOCK_SIZE, VOCAB_SIZE)
        assert loss is None


class TestBackboneMultiBlock:
    """Tests for block_len < block_size (block diffusion mode)."""

    def test_is_not_single_block(self, multi_block_model):
        assert not multi_block_model._is_single_block

    def test_masks_are_registered(self, multi_block_model):
        assert multi_block_model.train_attn_mask is not None
        assert multi_block_model.sample_attn_mask is not None

    def test_train_mask_shape(self, multi_block_model):
        assert multi_block_model.train_attn_mask.shape == (1, 1, 2 * BLOCK_SIZE, 2 * BLOCK_SIZE)

    def test_sample_mask_shape(self, multi_block_model):
        assert multi_block_model.sample_attn_mask.shape == (1, 1, BLOCK_SIZE, BLOCK_SIZE)

    def test_forward_train_logits_shape(self, multi_block_model):
        xt = torch.randint(0, VOCAB_SIZE, (2, BLOCK_SIZE))
        x0 = torch.randint(0, VOCAB_SIZE, (2, BLOCK_SIZE))
        logits, _ = multi_block_model.forward_train(xt, x0)
        # Logits should be (B, L, V) — only the x_t stream, not 2L
        assert logits.shape == (2, BLOCK_SIZE, VOCAB_SIZE)

    def test_forward_train_with_loss(self, multi_block_model):
        xt = torch.randint(0, VOCAB_SIZE, (2, BLOCK_SIZE))
        x0 = torch.randint(0, VOCAB_SIZE, (2, BLOCK_SIZE))
        mask = torch.rand(2, BLOCK_SIZE) > 0.5
        _, loss = multi_block_model.forward_train(xt, x0, targets=x0, supervise_mask=mask)
        assert loss is not None and loss.item() > 0

    def test_forward_train_block0_independent_of_x0(self, multi_block_model):
        """Block 0 x_t tokens should not see any x_0 (no preceding blocks)."""
        multi_block_model.eval()
        xt = torch.randint(0, VOCAB_SIZE, (1, BLOCK_SIZE))
        x0_a = torch.randint(0, VOCAB_SIZE, (1, BLOCK_SIZE))
        x0_b = torch.randint(0, VOCAB_SIZE, (1, BLOCK_SIZE))

        with torch.no_grad():
            logits_a, _ = multi_block_model.forward_train(xt, x0_a)
            logits_b, _ = multi_block_model.forward_train(xt, x0_b)

        # Block 0 should be identical
        assert torch.allclose(logits_a[:, :BLOCK_LEN], logits_b[:, :BLOCK_LEN], atol=1e-5), \
            "Block 0 logits must be independent of x_0"

    def test_forward_train_block1_depends_on_x0(self, multi_block_model):
        """Block 1 x_t tokens should depend on x_0 from block 0."""
        multi_block_model.eval()
        xt = torch.randint(0, VOCAB_SIZE, (1, BLOCK_SIZE))
        x0_a = torch.randint(0, VOCAB_SIZE, (1, BLOCK_SIZE))
        x0_b = torch.randint(0, VOCAB_SIZE, (1, BLOCK_SIZE))
        # Make block 0 x_0 different
        x0_b[0, :BLOCK_LEN] = (x0_a[0, :BLOCK_LEN] + 1) % VOCAB_SIZE

        with torch.no_grad():
            logits_a, _ = multi_block_model.forward_train(xt, x0_a)
            logits_b, _ = multi_block_model.forward_train(xt, x0_b)

        diff = (logits_a[:, BLOCK_LEN:2*BLOCK_LEN] - logits_b[:, BLOCK_LEN:2*BLOCK_LEN]).abs().max()
        assert diff > 1e-4, \
            f"Block 1 logits should depend on block 0 x_0, but diff={diff}"

    def test_forward_sample_logits_shape(self, multi_block_model):
        x = torch.randint(0, VOCAB_SIZE, (2, BLOCK_SIZE))
        logits, _ = multi_block_model.forward_sample(x)
        assert logits.shape == (2, BLOCK_SIZE, VOCAB_SIZE)

    def test_forward_sample_with_partial_seq(self, multi_block_model):
        """Sampling uses growing x_view = x[:, :block_end]; test with partial length."""
        x = torch.randint(0, VOCAB_SIZE, (2, 2 * BLOCK_LEN))  # 2 blocks worth
        logits, _ = multi_block_model.forward_sample(x)
        assert logits.shape == (2, 2 * BLOCK_LEN, VOCAB_SIZE)

    def test_forward_sample_block_causal(self, multi_block_model):
        """Block 0 should not be affected by tokens in block 1."""
        multi_block_model.eval()
        x = torch.randint(0, VOCAB_SIZE, (1, BLOCK_SIZE))
        x_alt = x.clone()
        x_alt[0, BLOCK_LEN:] = (x[0, BLOCK_LEN:] + 1) % VOCAB_SIZE

        with torch.no_grad():
            logits_orig, _ = multi_block_model.forward_sample(x)
            logits_alt, _ = multi_block_model.forward_sample(x_alt)

        # Block 0 logits should be identical regardless of later blocks
        assert torch.allclose(logits_orig[:, :BLOCK_LEN], logits_alt[:, :BLOCK_LEN], atol=1e-5)

    def test_auto_backend_falls_back_to_sdpa_after_flex_failure(
        self, multi_block_model, monkeypatch
    ):
        class FakeBlockMask:
            pass

        calls = {"count": 0}

        def fail_flex(q, k, v, attn_mask):
            calls["count"] += 1
            raise RuntimeError("flex_attention failed inside compiled wrapper")

        monkeypatch.setattr(backbone, "FLEX_ATTN_AVAILABLE", True)
        monkeypatch.setattr(backbone, "BlockMask", FakeBlockMask)
        monkeypatch.setattr(backbone, "fused_flex_attention", fail_flex)
        monkeypatch.setattr(
            DiffusionBackbone,
            "_should_use_flex_attention",
            lambda self, device: self.bd3_attn_backend != "sdpa",
        )
        monkeypatch.setattr(
            DiffusionBackbone,
            "_get_flex_attn_mask",
            lambda self, Tseq, dual_stream, device: FakeBlockMask(),
        )

        multi_block_model.eval()
        multi_block_model.bd3_attn_backend = "auto"
        xt = torch.randint(0, VOCAB_SIZE, (2, BLOCK_SIZE))
        x0 = torch.randint(0, VOCAB_SIZE, (2, BLOCK_SIZE))

        with torch.no_grad():
            logits, _ = multi_block_model.forward_train(xt, x0)
            logits_2, _ = multi_block_model.forward_train(xt, x0)

        assert logits.shape == (2, BLOCK_SIZE, VOCAB_SIZE)
        assert torch.equal(logits, logits_2)
        assert multi_block_model.bd3_attn_backend == "sdpa"
        assert "flex_attention failed" in multi_block_model._flex_fallback_reason
        assert calls["count"] == 1

    def test_forced_flex_backend_does_not_swallow_flex_failure(
        self, multi_block_model, monkeypatch
    ):
        class FakeBlockMask:
            pass

        def fail_flex(q, k, v, attn_mask):
            raise RuntimeError("flex_attention failed inside compiled wrapper")

        monkeypatch.setattr(backbone, "FLEX_ATTN_AVAILABLE", True)
        monkeypatch.setattr(backbone, "BlockMask", FakeBlockMask)
        monkeypatch.setattr(backbone, "fused_flex_attention", fail_flex)
        monkeypatch.setattr(
            DiffusionBackbone,
            "_should_use_flex_attention",
            lambda self, device: self.bd3_attn_backend != "sdpa",
        )
        monkeypatch.setattr(
            DiffusionBackbone,
            "_get_flex_attn_mask",
            lambda self, Tseq, dual_stream, device: FakeBlockMask(),
        )

        multi_block_model.eval()
        multi_block_model.bd3_attn_backend = "flex"
        xt = torch.randint(0, VOCAB_SIZE, (1, BLOCK_SIZE))
        x0 = torch.randint(0, VOCAB_SIZE, (1, BLOCK_SIZE))

        with pytest.raises(RuntimeError, match="flex_attention failed"):
            with torch.no_grad():
                multi_block_model.forward_train(xt, x0)

        assert multi_block_model.bd3_attn_backend == "flex"
        assert multi_block_model._flex_fallback_reason is None


class TestBackboneRotary:
    def test_dual_stream_rotary_length(self, multi_block_model):
        cos, sin = multi_block_model._select_rotary(2 * BLOCK_SIZE, dual_stream=True)
        assert cos.shape[1] == 2 * BLOCK_SIZE

    def test_single_stream_rotary_length(self, multi_block_model):
        cos, sin = multi_block_model._select_rotary(BLOCK_SIZE, dual_stream=False)
        assert cos.shape[1] == BLOCK_SIZE

    def test_dual_rotary_repeats_positions(self, multi_block_model):
        """In dual-stream, positions 0..L-1 should repeat for both halves."""
        cos_dual, _ = multi_block_model._select_rotary(2 * BLOCK_SIZE, dual_stream=True)
        L = BLOCK_SIZE
        assert torch.equal(cos_dual[:, :L], cos_dual[:, L:])


# =====================================================================
# model_bd3lm tests
# =====================================================================

import model_bd3lm


class TestBD3LMGetBatch:
    def test_shapes(self, cfg):
        xt, x0, mask = model_bd3lm.get_batch("train", cfg)
        assert xt.shape == (BATCH_SIZE, BLOCK_SIZE)
        assert x0.shape == (BATCH_SIZE, BLOCK_SIZE)
        assert mask.shape == (BATCH_SIZE, BLOCK_SIZE)
        assert mask.dtype == torch.bool

    def test_on_device(self, cfg):
        xt, x0, mask = model_bd3lm.get_batch("val", cfg)
        assert str(xt.device) == "cpu"


class TestBD3LMGetEvalBatch:
    def test_has_eval_batch(self):
        assert hasattr(model_bd3lm, "get_eval_batch")

    def test_shapes(self, cfg):
        xt, x0, mask = model_bd3lm.get_eval_batch("val", cfg, fixed_t_step=5)
        assert xt.shape == (BATCH_SIZE, BLOCK_SIZE)
        assert x0.shape == (BATCH_SIZE, BLOCK_SIZE)
        assert mask.shape == (BATCH_SIZE, BLOCK_SIZE)


class TestBD3LMComputeLoss:
    def test_returns_scalar(self, cfg):
        model = model_bd3lm.Model(
            vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
            n_layer=N_LAYER, head_dim=HEAD_DIM, block_size=BLOCK_SIZE,
            block_len=BLOCK_LEN, dropout=0.0,
        )
        batch = model_bd3lm.get_batch("train", cfg)
        loss = model_bd3lm.compute_loss(model, batch, cfg)
        assert loss.shape == ()
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_loss_is_differentiable(self, cfg):
        model = model_bd3lm.Model(
            vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
            n_layer=N_LAYER, head_dim=HEAD_DIM, block_size=BLOCK_SIZE,
            block_len=BLOCK_LEN, dropout=0.0,
        )
        batch = model_bd3lm.get_batch("train", cfg)
        loss = model_bd3lm.compute_loss(model, batch, cfg)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
        assert has_grad, "Loss should produce gradients"


class TestBD3LMComputeEvalLoss:
    def test_returns_scalar(self, cfg):
        model = model_bd3lm.Model(
            vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
            n_layer=N_LAYER, head_dim=HEAD_DIM, block_size=BLOCK_SIZE,
            block_len=BLOCK_LEN, dropout=0.0,
        )
        xt, x0, mask = model_bd3lm.get_eval_batch("val", cfg, fixed_t_step=5)
        loss = model_bd3lm.compute_eval_loss(model, xt, x0, mask)
        assert loss.shape == ()
        assert not torch.isnan(loss)


class TestBD3LMGenerateFrom:
    def test_returns_string(self):
        model = model_bd3lm.Model(
            vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
            n_layer=N_LAYER, head_dim=HEAD_DIM, block_size=BLOCK_SIZE,
            block_len=BLOCK_LEN, dropout=0.0,
        )
        model.eval()
        decode = make_decode()

        x = torch.full((1, BLOCK_SIZE), MASK_TOKEN_ID)
        x[0, :4] = torch.randint(1, VOCAB_SIZE, (4,))
        prompt_mask = torch.zeros(1, BLOCK_SIZE, dtype=torch.bool)
        prompt_mask[0, :4] = True

        text = model_bd3lm.generate_from(
            model, x, prompt_mask,
            T=T, block_size=BLOCK_SIZE, block_len=BLOCK_LEN,
            vocab_size=VOCAB_SIZE, mask_token_id=MASK_TOKEN_ID,
            survival_prob_scalar=survival_prob_scalar,
            decode=decode,
        )
        assert isinstance(text, str)
        assert len(text) == BLOCK_SIZE

    def test_prompt_preserved(self):
        """The prompt tokens should not be overwritten during generation."""
        model = model_bd3lm.Model(
            vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
            n_layer=N_LAYER, head_dim=HEAD_DIM, block_size=BLOCK_SIZE,
            block_len=BLOCK_LEN, dropout=0.0,
        )
        model.eval()
        decode = make_decode()

        prompt_tokens = torch.randint(1, VOCAB_SIZE, (4,))
        x = torch.full((1, BLOCK_SIZE), MASK_TOKEN_ID)
        x[0, :4] = prompt_tokens
        prompt_mask = torch.zeros(1, BLOCK_SIZE, dtype=torch.bool)
        prompt_mask[0, :4] = True

        model_bd3lm.generate_from(
            model, x, prompt_mask,
            T=T, block_size=BLOCK_SIZE, block_len=BLOCK_LEN,
            vocab_size=VOCAB_SIZE, mask_token_id=MASK_TOKEN_ID,
            survival_prob_scalar=survival_prob_scalar,
            decode=decode,
        )
        assert torch.equal(x[0, :4], prompt_tokens)

    def test_no_mask_tokens_in_output(self):
        """After generation, no mask tokens should remain."""
        model = model_bd3lm.Model(
            vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
            n_layer=N_LAYER, head_dim=HEAD_DIM, block_size=BLOCK_SIZE,
            block_len=BLOCK_LEN, dropout=0.0,
        )
        model.eval()
        decode = make_decode()

        x = torch.full((1, BLOCK_SIZE), MASK_TOKEN_ID)
        x[0, :4] = torch.randint(1, VOCAB_SIZE, (4,))
        prompt_mask = torch.zeros(1, BLOCK_SIZE, dtype=torch.bool)
        prompt_mask[0, :4] = True

        model_bd3lm.generate_from(
            model, x, prompt_mask,
            T=T, block_size=BLOCK_SIZE, block_len=BLOCK_LEN,
            vocab_size=VOCAB_SIZE, mask_token_id=MASK_TOKEN_ID,
            survival_prob_scalar=survival_prob_scalar,
            decode=decode,
        )
        assert (x[0] != MASK_TOKEN_ID).all(), \
            f"Found mask tokens after generation: {(x[0] == MASK_TOKEN_ID).sum()} remaining"


class TestModelClassIsBackbone:
    """BD3-LM model should subclass DiffusionBackbone."""
    def test_inherits_backbone(self):
        assert issubclass(model_bd3lm.Model, DiffusionBackbone)


class TestBlockLenVariations:
    """Test that models work with different block_len values."""
    @pytest.mark.parametrize("block_len", [4, 8, 16, 32])
    def test_forward_train_various_block_lens(self, block_len):
        model = DiffusionBackbone(
            vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
            n_layer=N_LAYER, head_dim=HEAD_DIM, block_size=BLOCK_SIZE,
            block_len=block_len, dropout=0.0,
        )
        xt = torch.randint(0, VOCAB_SIZE, (2, BLOCK_SIZE))
        x0 = torch.randint(0, VOCAB_SIZE, (2, BLOCK_SIZE))
        mask = torch.rand(2, BLOCK_SIZE) > 0.5
        logits, loss = model.forward_train(xt, x0, targets=x0, supervise_mask=mask)
        assert logits is None
        assert loss is not None
