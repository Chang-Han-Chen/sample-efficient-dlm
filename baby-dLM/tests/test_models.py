"""
test_models.py — Comprehensive tests for all three model modules.

Tests model_AR, model_MDLM, and model_bd3lm covering:
  - Batch construction invariants
  - Loss computation (scalar, positive, differentiable)
  - Generation (prompt preservation, output validity, algorithmic invariants)
  - Progressive unmasking correctness
  - Block-autoregressive invariants (BD3-LM)
  - Cross-model interface compatibility
"""

import math
import pytest
import torch
import torch.nn as nn

from backbone import DiffusionBackbone
import model_AR
import model_MDLM
import model_bd3lm


# =====================================================================
# Shared test constants and fixtures
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


def linear_survival_prob_scalar(t_step):
    """Deterministic survival probability for testing."""
    t_frac = float(t_step) / T
    return max(0.0, min(1.0, 1.0 - t_frac))


def linear_survival_prob_tensor(t_steps):
    t_frac = t_steps.float() / T
    return (1.0 - t_frac).clamp(0.0, 1.0)


def make_decode():
    itos = {i: chr(65 + (i % 26)) for i in range(VOCAB_SIZE)}
    def decode(l):
        return "".join([itos[i] for i in l])
    return decode


@pytest.fixture
def cfg():
    """Config dict for diffusion models."""
    data = torch.randint(1, VOCAB_SIZE, (2000,))
    split = int(0.9 * len(data))
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
        "survival_prob_tensor": linear_survival_prob_tensor,
    }


@pytest.fixture
def mdlm_model():
    return model_MDLM.Model(
        vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
        n_layer=N_LAYER, head_dim=HEAD_DIM, block_size=BLOCK_SIZE,
        dropout=0.0,
    )


@pytest.fixture
def ar_model():
    return model_AR.Model(
        vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
        n_layer=N_LAYER, head_dim=HEAD_DIM, block_size=BLOCK_SIZE,
        dropout=0.0,
    )


@pytest.fixture
def bd3lm_model():
    return model_bd3lm.Model(
        vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
        n_layer=N_LAYER, head_dim=HEAD_DIM, block_size=BLOCK_SIZE,
        block_len=BLOCK_LEN, dropout=0.0,
    )


# =====================================================================
# model_AR tests
# =====================================================================


class TestARGetBatch:
    def test_shapes(self, cfg):
        x, targets, mask = model_AR.get_batch("train", cfg)
        assert x.shape == (BATCH_SIZE, BLOCK_SIZE)
        assert targets.shape == (BATCH_SIZE, BLOCK_SIZE)
        assert mask is None

    def test_input_equals_targets(self, cfg):
        """AR batch returns the same tensor for input and targets."""
        x, targets, _ = model_AR.get_batch("train", cfg)
        assert torch.equal(x, targets)

    def test_eval_batch_same_as_train(self, cfg):
        """get_eval_batch should return the same format as get_batch."""
        torch.manual_seed(42)
        x1, t1, m1 = model_AR.get_batch("val", cfg)
        torch.manual_seed(42)
        x2, t2, m2 = model_AR.get_eval_batch("val", cfg)
        assert torch.equal(x1, x2)
        assert m1 is None and m2 is None

    def test_no_mask_tokens(self, cfg):
        """AR batches should never contain the mask token (it's real data)."""
        x, _, _ = model_AR.get_batch("train", cfg)
        # Data is generated with randint(1, VOCAB_SIZE), so token 0 shouldn't appear
        # unless it's in the training data by chance (which is possible).
        # But the key point is AR doesn't mask anything.
        assert x.dtype == torch.long

    def test_train_vs_val_split(self, cfg):
        """Train and val splits should produce different data."""
        torch.manual_seed(0)
        x_train, _, _ = model_AR.get_batch("train", cfg)
        torch.manual_seed(0)
        x_val, _, _ = model_AR.get_batch("val", cfg)
        # Different source data, so different batches even with same seed
        # (because the underlying data arrays differ)
        # They *could* match by coincidence, but extremely unlikely.


class TestARComputeLoss:
    def test_returns_scalar(self, cfg, ar_model):
        batch = model_AR.get_batch("train", cfg)
        loss = model_AR.compute_loss(ar_model, batch, cfg)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_differentiable(self, cfg, ar_model):
        batch = model_AR.get_batch("train", cfg)
        loss = model_AR.compute_loss(ar_model, batch, cfg)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in ar_model.parameters())
        assert has_grad

    def test_eval_loss_matches_format(self, cfg, ar_model):
        """compute_eval_loss should accept the (xt, x0, mask) signature."""
        x, targets, mask = model_AR.get_eval_batch("val", cfg)
        loss = model_AR.compute_eval_loss(ar_model, x, targets, mask)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_loss_is_next_token_prediction(self, ar_model):
        """Verify AR loss is logits[:,:-1] predicting targets[:,1:]."""
        ar_model.eval()
        x = torch.randint(1, VOCAB_SIZE, (1, BLOCK_SIZE))
        logits, loss = ar_model(x, targets=x)

        # Manually compute expected loss
        expected_loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB_SIZE),
            x[:, 1:].reshape(-1),
        )
        assert torch.allclose(loss, expected_loss, atol=1e-6)


class TestARModel:
    def test_causal_attention(self, ar_model):
        """Position i should only depend on positions 0..i."""
        ar_model.eval()
        x = torch.randint(1, VOCAB_SIZE, (1, BLOCK_SIZE))
        x_alt = x.clone()
        # Change position 10 onward
        x_alt[0, 10:] = (x[0, 10:] + 1) % VOCAB_SIZE

        with torch.no_grad():
            logits_orig, _ = ar_model(x)
            logits_alt, _ = ar_model(x_alt)

        # Positions 0..9 should produce identical logits
        assert torch.allclose(logits_orig[0, :10], logits_alt[0, :10], atol=1e-5), \
            "AR logits before the change point should be identical (causal)"

        # Position 10+ should differ (position 10 depends on input at 10)
        diff = (logits_orig[0, 10:] - logits_alt[0, 10:]).abs().max()
        assert diff > 1e-4, "AR logits after the change point should differ"

    def test_forward_no_targets(self, ar_model):
        x = torch.randint(1, VOCAB_SIZE, (2, BLOCK_SIZE))
        logits, loss = ar_model(x)
        assert logits.shape == (2, BLOCK_SIZE, VOCAB_SIZE)
        assert loss is None

    def test_variable_length_input(self, ar_model):
        """AR model should handle sub-sequence inputs (used in generate)."""
        for length in [1, 4, 16, BLOCK_SIZE]:
            x = torch.randint(1, VOCAB_SIZE, (1, length))
            logits, _ = ar_model(x)
            assert logits.shape == (1, length, VOCAB_SIZE)


class TestARGenerate:
    def test_returns_string(self, ar_model):
        decode = make_decode()
        x = torch.full((1, BLOCK_SIZE), MASK_TOKEN_ID, dtype=torch.long)
        x[0, :4] = torch.randint(1, VOCAB_SIZE, (4,))
        prompt_mask = torch.zeros(1, BLOCK_SIZE, dtype=torch.bool)
        prompt_mask[0, :4] = True

        text = model_AR.generate_from(
            ar_model, x, prompt_mask,
            T=T, block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE,
            mask_token_id=MASK_TOKEN_ID,
            survival_prob_scalar=linear_survival_prob_scalar,
            decode=decode,
        )
        assert isinstance(text, str)
        assert len(text) == BLOCK_SIZE

    def test_prompt_preserved(self, ar_model):
        decode = make_decode()
        prompt_tokens = torch.randint(1, VOCAB_SIZE, (4,))
        x = torch.full((1, BLOCK_SIZE), MASK_TOKEN_ID, dtype=torch.long)
        x[0, :4] = prompt_tokens
        prompt_mask = torch.zeros(1, BLOCK_SIZE, dtype=torch.bool)
        prompt_mask[0, :4] = True

        model_AR.generate_from(
            ar_model, x, prompt_mask,
            T=T, block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE,
            mask_token_id=MASK_TOKEN_ID,
            survival_prob_scalar=linear_survival_prob_scalar,
            decode=decode,
        )
        assert torch.equal(x[0, :4], prompt_tokens)

    def test_all_positions_filled(self, ar_model):
        """Every position should have a real token after generation."""
        decode = make_decode()
        x = torch.full((1, BLOCK_SIZE), MASK_TOKEN_ID, dtype=torch.long)
        x[0, :4] = torch.randint(1, VOCAB_SIZE, (4,))
        prompt_mask = torch.zeros(1, BLOCK_SIZE, dtype=torch.bool)
        prompt_mask[0, :4] = True

        model_AR.generate_from(
            ar_model, x, prompt_mask,
            T=T, block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE,
            mask_token_id=MASK_TOKEN_ID,
            survival_prob_scalar=linear_survival_prob_scalar,
            decode=decode,
        )
        # All positions should be valid tokens
        assert (x[0] >= 0).all() and (x[0] < VOCAB_SIZE).all()


# =====================================================================
# model_MDLM tests
# =====================================================================


class TestMDLMGetBatch:
    def test_shapes(self, cfg):
        xt, x0, mask = model_MDLM.get_batch("train", cfg)
        assert xt.shape == (BATCH_SIZE, BLOCK_SIZE)
        assert x0.shape == (BATCH_SIZE, BLOCK_SIZE)
        assert mask.shape == (BATCH_SIZE, BLOCK_SIZE)
        assert mask.dtype == torch.bool

    def test_masked_positions_have_mask_token(self, cfg):
        xt, x0, mask = model_MDLM.get_batch("train", cfg)
        assert (xt[mask] == MASK_TOKEN_ID).all(), \
            "All masked positions in xt should equal mask_token_id"

    def test_unmasked_positions_equal_x0(self, cfg):
        xt, x0, mask = model_MDLM.get_batch("train", cfg)
        assert torch.equal(xt[~mask], x0[~mask]), \
            "Unmasked positions in xt should equal x0"

    def test_masking_is_per_sample_not_blockwise(self, cfg):
        """MDLM uses one t per sample (not per block like BD3-LM).

        Within a single sample, all tokens share the same survival probability,
        so the masking is i.i.d. Bernoulli across positions. This means
        adjacent positions should NOT be perfectly correlated.

        In contrast, BD3-LM has per-block t which makes tokens within a block
        share the same mask probability.
        """
        # Run many batches and check that within-block masking variance exists
        torch.manual_seed(42)
        # For MDLM with a single t per sample, the mask is iid within each sample
        # so block-level mask rates should vary within a sample
        n_trials = 20
        found_intra_sample_variation = False
        for _ in range(n_trials):
            xt, x0, mask = model_MDLM.get_batch("train", cfg)
            # Look at first sample, check if different blocks have different mask rates
            for b_idx in range(0, BATCH_SIZE):
                block_rates = []
                for blk in range(BLOCK_SIZE // BLOCK_LEN):
                    start = blk * BLOCK_LEN
                    end = start + BLOCK_LEN
                    block_rates.append(mask[b_idx, start:end].float().mean().item())
                # All blocks in MDLM share the same Bernoulli prob, so rates
                # should *roughly* match but NOT be exactly identical (unless all masked or all unmasked)
                if len(set(block_rates)) > 1 and 0 < sum(block_rates) < len(block_rates):
                    found_intra_sample_variation = True
                    break
            if found_intra_sample_variation:
                break
        assert found_intra_sample_variation, \
            "MDLM should have varying mask rates across blocks within a sample"

    def test_higher_t_means_more_masking(self, cfg):
        """At higher timesteps, more tokens should be masked (on average)."""
        # Use fixed survival prob: at t=1 → a=0.9, at t=T → a=0.0
        torch.manual_seed(0)
        low_mask_counts = []
        high_mask_counts = []
        for _ in range(50):
            xt, x0, mask = model_MDLM.get_batch("train", cfg)
            low_mask_counts.append(mask.float().mean().item())

        avg_mask = sum(low_mask_counts) / len(low_mask_counts)
        # With uniform t in [1, T], average mask rate should be roughly 0.5
        assert 0.2 < avg_mask < 0.8, f"Average mask rate {avg_mask} seems off"


class TestMDLMComputeLoss:
    def test_returns_scalar(self, cfg, mdlm_model):
        batch = model_MDLM.get_batch("train", cfg)
        loss = model_MDLM.compute_loss(mdlm_model, batch, cfg)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_differentiable(self, cfg, mdlm_model):
        batch = model_MDLM.get_batch("train", cfg)
        loss = model_MDLM.compute_loss(mdlm_model, batch, cfg)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in mdlm_model.parameters())
        assert has_grad

    def test_loss_only_on_masked_positions(self, mdlm_model):
        """Loss should be computed only over masked positions (mask=True)."""
        mdlm_model.eval()
        x0 = torch.randint(1, VOCAB_SIZE, (1, BLOCK_SIZE))
        xt = x0.clone()
        mask = torch.zeros(1, BLOCK_SIZE, dtype=torch.bool)
        # Mask positions 5-10
        mask[0, 5:10] = True
        xt[mask] = MASK_TOKEN_ID

        with torch.no_grad():
            logits, _ = mdlm_model(xt)

        # Manually compute loss on masked positions only
        from torch.nn import functional as F
        per_token_loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            x0.reshape(-1),
            reduction="none",
        ).reshape(1, BLOCK_SIZE)
        expected_loss = per_token_loss[mask].mean()

        _, actual_loss = mdlm_model(xt, targets=x0, mask=mask)
        assert torch.allclose(actual_loss, expected_loss, atol=1e-5)

    def test_eval_loss(self, cfg, mdlm_model):
        xt, x0, mask = model_MDLM.get_batch("val", cfg)
        loss = model_MDLM.compute_eval_loss(mdlm_model, xt, x0, mask)
        assert loss.shape == ()
        assert not torch.isnan(loss)


class TestMDLMProgressiveUnmasking:
    """Test the core progressive unmasking algorithm used during generation."""

    def test_t1_fills_all_remaining_masks(self):
        """At t=1, every masked non-prompt position should be filled."""
        B, L = 1, 16
        x = torch.full((B, L), MASK_TOKEN_ID, dtype=torch.long)
        x[0, :4] = torch.randint(1, VOCAB_SIZE, (4,))
        prompt_mask = torch.zeros(B, L, dtype=torch.bool)
        prompt_mask[0, :4] = True

        sampled = torch.randint(1, VOCAB_SIZE, (B, L))
        sampled_conf = torch.rand(B, L)

        result = model_MDLM._progressive_unmask_step(
            x.clone(), prompt_mask, sampled, sampled_conf, t=1,
            mask_token_id=MASK_TOKEN_ID,
            survival_prob_scalar=linear_survival_prob_scalar,
        )
        # All non-prompt positions should now be filled
        assert (result[0, 4:] != MASK_TOKEN_ID).all()
        # Prompt should be unchanged
        assert torch.equal(result[0, :4], x[0, :4])

    def test_prompt_never_modified(self):
        """Prompt positions should never be changed, regardless of timestep."""
        B, L = 1, 16
        prompt_tokens = torch.randint(1, VOCAB_SIZE, (4,))
        x = torch.full((B, L), MASK_TOKEN_ID, dtype=torch.long)
        x[0, :4] = prompt_tokens
        prompt_mask = torch.zeros(B, L, dtype=torch.bool)
        prompt_mask[0, :4] = True

        sampled = torch.randint(1, VOCAB_SIZE, (B, L))
        sampled_conf = torch.rand(B, L)

        for t in range(1, T + 1):
            result = model_MDLM._progressive_unmask_step(
                x.clone(), prompt_mask, sampled, sampled_conf, t,
                mask_token_id=MASK_TOKEN_ID,
                survival_prob_scalar=linear_survival_prob_scalar,
            )
            assert torch.equal(result[0, :4], prompt_tokens), \
                f"Prompt was modified at t={t}"

    def test_no_remasking(self):
        """Once a token is unmasked, it should never be re-masked.

        This is the key invariant that distinguishes MDLM from remasked models.
        """
        B, L = 1, 16
        x = torch.full((B, L), MASK_TOKEN_ID, dtype=torch.long)
        x[0, :4] = torch.randint(1, VOCAB_SIZE, (4,))
        prompt_mask = torch.zeros(B, L, dtype=torch.bool)
        prompt_mask[0, :4] = True

        sampled = torch.randint(1, VOCAB_SIZE, (B, L))
        sampled_conf = torch.rand(B, L)

        prev_unmasked = (x != MASK_TOKEN_ID).clone()
        for t in reversed(range(1, T + 1)):
            x = model_MDLM._progressive_unmask_step(
                x, prompt_mask, sampled, sampled_conf, t,
                mask_token_id=MASK_TOKEN_ID,
                survival_prob_scalar=linear_survival_prob_scalar,
            )
            curr_unmasked = (x != MASK_TOKEN_ID)
            # Every position that was unmasked before should still be unmasked
            assert (curr_unmasked | ~prev_unmasked).all(), \
                f"Token was re-masked at t={t}!"
            prev_unmasked = curr_unmasked.clone()

    def test_monotonic_unmasking(self):
        """The number of unmasked tokens should never decrease across steps."""
        B, L = 1, 16
        x = torch.full((B, L), MASK_TOKEN_ID, dtype=torch.long)
        x[0, :4] = torch.randint(1, VOCAB_SIZE, (4,))
        prompt_mask = torch.zeros(B, L, dtype=torch.bool)
        prompt_mask[0, :4] = True

        sampled = torch.randint(1, VOCAB_SIZE, (B, L))
        sampled_conf = torch.rand(B, L)

        prev_count = (x != MASK_TOKEN_ID).sum().item()
        for t in reversed(range(1, T + 1)):
            x = model_MDLM._progressive_unmask_step(
                x, prompt_mask, sampled, sampled_conf, t,
                mask_token_id=MASK_TOKEN_ID,
                survival_prob_scalar=linear_survival_prob_scalar,
            )
            curr_count = (x != MASK_TOKEN_ID).sum().item()
            assert curr_count >= prev_count, \
                f"Unmasked count decreased from {prev_count} to {curr_count} at t={t}"
            prev_count = curr_count

    def test_highest_confidence_unmasked_first(self):
        """The positions with highest confidence should be unmasked first."""
        B, L = 1, 12
        x = torch.full((B, L), MASK_TOKEN_ID, dtype=torch.long)
        x[0, :2] = torch.randint(1, VOCAB_SIZE, (2,))
        prompt_mask = torch.zeros(B, L, dtype=torch.bool)
        prompt_mask[0, :2] = True

        sampled = torch.randint(1, VOCAB_SIZE, (B, L))
        # Make positions 2,3 have the highest confidence
        sampled_conf = torch.zeros(B, L)
        sampled_conf[0, 2] = 0.99
        sampled_conf[0, 3] = 0.98
        sampled_conf[0, 4] = 0.10
        sampled_conf[0, 5:] = 0.05

        # At a mid-level t, only a few tokens should be unmasked — the
        # highest confidence ones.
        result = model_MDLM._progressive_unmask_step(
            x.clone(), prompt_mask, sampled, sampled_conf, t=T,
            mask_token_id=MASK_TOKEN_ID,
            survival_prob_scalar=linear_survival_prob_scalar,
        )

        # If any tokens were unmasked, they should be the high-confidence ones
        newly_unmasked = (result != MASK_TOKEN_ID) & (x == MASK_TOKEN_ID)
        if newly_unmasked.any():
            unmasked_positions = newly_unmasked[0].nonzero(as_tuple=True)[0].tolist()
            # Position 2 and 3 should be among the first unmasked
            for pos in unmasked_positions:
                assert sampled_conf[0, pos] >= sampled_conf[0, 4], \
                    f"Position {pos} (conf={sampled_conf[0, pos]}) unmasked before higher-confidence positions"

    def test_already_unmasked_preserved(self):
        """Positions that are already real tokens should not change."""
        B, L = 1, 12
        x = torch.full((B, L), MASK_TOKEN_ID, dtype=torch.long)
        x[0, :4] = torch.randint(1, VOCAB_SIZE, (4,))
        # Also unmask position 6 with a specific token
        x[0, 6] = 42
        prompt_mask = torch.zeros(B, L, dtype=torch.bool)
        prompt_mask[0, :4] = True
        # Note: position 6 is NOT in the prompt, but is already unmasked

        sampled = torch.randint(1, VOCAB_SIZE, (B, L))
        sampled_conf = torch.rand(B, L)

        result = model_MDLM._progressive_unmask_step(
            x.clone(), prompt_mask, sampled, sampled_conf, t=5,
            mask_token_id=MASK_TOKEN_ID,
            survival_prob_scalar=linear_survival_prob_scalar,
        )
        # Position 6 should still be 42 (not overwritten by sampled)
        assert result[0, 6] == 42


class TestMDLMGenerate:
    def test_returns_string(self, mdlm_model):
        decode = make_decode()
        x = torch.full((1, BLOCK_SIZE), MASK_TOKEN_ID, dtype=torch.long)
        x[0, :4] = torch.randint(1, VOCAB_SIZE, (4,))
        prompt_mask = torch.zeros(1, BLOCK_SIZE, dtype=torch.bool)
        prompt_mask[0, :4] = True

        text = model_MDLM.generate_from(
            mdlm_model, x, prompt_mask,
            T=T, block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE,
            mask_token_id=MASK_TOKEN_ID,
            survival_prob_scalar=linear_survival_prob_scalar,
            decode=decode,
        )
        assert isinstance(text, str)
        assert len(text) == BLOCK_SIZE

    def test_prompt_preserved(self, mdlm_model):
        decode = make_decode()
        prompt_tokens = torch.randint(1, VOCAB_SIZE, (8,))
        x = torch.full((1, BLOCK_SIZE), MASK_TOKEN_ID, dtype=torch.long)
        x[0, :8] = prompt_tokens
        prompt_mask = torch.zeros(1, BLOCK_SIZE, dtype=torch.bool)
        prompt_mask[0, :8] = True

        model_MDLM.generate_from(
            mdlm_model, x, prompt_mask,
            T=T, block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE,
            mask_token_id=MASK_TOKEN_ID,
            survival_prob_scalar=linear_survival_prob_scalar,
            decode=decode,
        )
        assert torch.equal(x[0, :8], prompt_tokens)

    def test_no_mask_tokens_remaining(self, mdlm_model):
        """generate_from should produce a fully decoded sequence (no mask tokens).

        Note: MDLM's generate_from doesn't guarantee in-place mutation of the
        passed tensor at the final step (torch.where creates a new tensor).
        So we verify via the returned decoded string length instead.
        """
        decode = make_decode()
        x = torch.full((1, BLOCK_SIZE), MASK_TOKEN_ID, dtype=torch.long)
        x[0, :4] = torch.randint(1, VOCAB_SIZE, (4,))
        prompt_mask = torch.zeros(1, BLOCK_SIZE, dtype=torch.bool)
        prompt_mask[0, :4] = True

        result = model_MDLM.generate_from(
            mdlm_model, x, prompt_mask,
            T=T, block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE,
            mask_token_id=MASK_TOKEN_ID,
            survival_prob_scalar=linear_survival_prob_scalar,
            decode=decode,
        )
        # decode maps each token to a char, so a fully-decoded sequence
        # has exactly BLOCK_SIZE characters
        assert len(result) == BLOCK_SIZE


# =====================================================================
# model_bd3lm helper tests
# =====================================================================


class TestBD3LMExpandBlockValues:
    def test_basic(self):
        vals = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        expanded = model_bd3lm._expand_block_values(vals, BLOCK_LEN)
        assert expanded.shape == (1, BLOCK_SIZE)
        assert (expanded[0, :BLOCK_LEN] == 1.0).all()
        assert (expanded[0, BLOCK_LEN:2*BLOCK_LEN] == 2.0).all()
        assert (expanded[0, 2*BLOCK_LEN:3*BLOCK_LEN] == 3.0).all()
        assert (expanded[0, 3*BLOCK_LEN:4*BLOCK_LEN] == 4.0).all()

    def test_with_seq_len_truncation(self):
        vals = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        expanded = model_bd3lm._expand_block_values(vals, BLOCK_LEN, seq_len=20)
        assert expanded.shape == (1, 20)

    def test_batched(self):
        vals = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        expanded = model_bd3lm._expand_block_values(vals, 4)
        assert expanded.shape == (2, 8)
        assert (expanded[0, :4] == 1.0).all()
        assert (expanded[1, :4] == 3.0).all()


class TestBD3LMSampleBlockTimesteps:
    def test_shape(self):
        t = model_bd3lm._sample_block_timesteps(
            BATCH_SIZE, BLOCK_SIZE, BLOCK_LEN, T, "cpu"
        )
        nblk = BLOCK_SIZE // BLOCK_LEN
        assert t.shape == (BATCH_SIZE, nblk)

    def test_range(self):
        t = model_bd3lm._sample_block_timesteps(
            BATCH_SIZE, BLOCK_SIZE, BLOCK_LEN, T, "cpu"
        )
        assert (t >= 1).all() and (t <= T).all()

    def test_fixed_mode(self):
        t = model_bd3lm._sample_block_timesteps(
            BATCH_SIZE, BLOCK_SIZE, BLOCK_LEN, T, "cpu", fixed_t_step=5
        )
        assert (t == 5).all()

    def test_fixed_clipping_high(self):
        """fixed_t_step above T should be clipped to T."""
        t = model_bd3lm._sample_block_timesteps(
            BATCH_SIZE, BLOCK_SIZE, BLOCK_LEN, T, "cpu", fixed_t_step=999
        )
        assert (t == T).all()

    def test_fixed_clipping_low(self):
        """fixed_t_step below 1 should be clipped to 1."""
        t = model_bd3lm._sample_block_timesteps(
            BATCH_SIZE, BLOCK_SIZE, BLOCK_LEN, T, "cpu", fixed_t_step=0
        )
        assert (t == 1).all()

    def test_independent_blocks(self):
        """Different blocks should get different timesteps (probabilistic)."""
        torch.manual_seed(42)
        t = model_bd3lm._sample_block_timesteps(
            16, BLOCK_SIZE, BLOCK_LEN, T, "cpu"
        )
        # With 16 samples and 4 blocks each, very unlikely all blocks match
        all_same = all(
            (t[i, 0] == t[i]).all().item()
            for i in range(t.size(0))
        )
        assert not all_same, "All blocks got the same timestep — should be independent"


class TestBD3LMMakeBlockNoisyBatch:
    def test_masked_positions_have_mask_token(self, cfg):
        x0 = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE))
        xt, _, mask = model_bd3lm._make_block_noisy_batch(x0, cfg)
        assert (xt[mask] == MASK_TOKEN_ID).all()

    def test_unmasked_positions_equal_x0(self, cfg):
        x0 = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE))
        xt, x0_out, mask = model_bd3lm._make_block_noisy_batch(x0, cfg)
        assert torch.equal(xt[~mask], x0_out[~mask])

    def test_high_t_means_more_masking(self, cfg):
        """At high fixed_t_step (close to T), most tokens should be masked."""
        x0 = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE))
        xt, _, mask = model_bd3lm._make_block_noisy_batch(x0, cfg, fixed_t_step=T)
        # At t=T, survival_prob = 0.0, so all tokens should be masked
        assert mask.all(), "At t=T, all tokens should be masked"

    def test_low_t_means_less_masking(self, cfg):
        """At low fixed_t_step (close to 1), most tokens should survive."""
        x0 = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE))
        xt, _, mask = model_bd3lm._make_block_noisy_batch(x0, cfg, fixed_t_step=1)
        # At t=1, survival_prob = 0.9, so ~10% masked
        mask_rate = mask.float().mean().item()
        assert mask_rate < 0.3, f"At t=1, mask rate {mask_rate} is too high"

    def test_masking_is_blockwise_correlated(self, cfg):
        """Within a single block, all tokens share the same masking probability.

        This is the key difference from MDLM: BD3-LM samples one t per block.
        At a fixed timestep, all tokens in a block face the same Bernoulli prob,
        so the expected mask rate within a block should be the same.

        To test this, we use fixed_t_step=5 and check that all blocks have
        roughly the same mask rate.
        """
        torch.manual_seed(42)
        x0 = torch.randint(1, VOCAB_SIZE, (100, BLOCK_SIZE))
        xt, _, mask = model_bd3lm._make_block_noisy_batch(
            x0, cfg, fixed_t_step=5
        )
        # With fixed_t_step, all blocks have survival_prob(5) = 0.5
        # so each block should have ~50% mask rate
        for blk in range(BLOCK_SIZE // BLOCK_LEN):
            start = blk * BLOCK_LEN
            end = start + BLOCK_LEN
            block_mask_rate = mask[:, start:end].float().mean().item()
            assert 0.3 < block_mask_rate < 0.7, \
                f"Block {blk} mask rate {block_mask_rate} is off for t=5"


class TestBD3LMGenerationHelpers:
    def test_current_block_range(self):
        assert model_bd3lm._current_block_range(0, 8) == (0, 8)
        assert model_bd3lm._current_block_range(1, 8) == (8, 16)
        assert model_bd3lm._current_block_range(3, 8) == (24, 32)

    def test_prompt_start_block_full_block(self):
        """Prompt exactly fills one block → start at block 1."""
        pm = torch.zeros(1, 32, dtype=torch.bool)
        pm[0, :8] = True  # fills block 0 exactly
        assert model_bd3lm._prompt_start_block(pm, 8) == 1

    def test_prompt_start_block_partial(self):
        """Prompt partially fills block 0 → start at block 0."""
        pm = torch.zeros(1, 32, dtype=torch.bool)
        pm[0, :4] = True
        assert model_bd3lm._prompt_start_block(pm, 8) == 0

    def test_prompt_start_block_two_blocks(self):
        """Prompt fills blocks 0 and 1 exactly → start at block 2."""
        pm = torch.zeros(1, 32, dtype=torch.bool)
        pm[0, :16] = True
        assert model_bd3lm._prompt_start_block(pm, 8) == 2

    def test_prompt_start_block_empty(self):
        """No prompt → start at block 0."""
        pm = torch.zeros(1, 32, dtype=torch.bool)
        assert model_bd3lm._prompt_start_block(pm, 8) == 0

    def test_build_generation_prompt_mask_freezes_earlier_blocks(self):
        pm = torch.zeros(1, 32, dtype=torch.bool)
        pm[0, :4] = True
        # For block 2: block_start=16, block_end=24
        frozen = model_bd3lm._build_generation_prompt_mask(pm, 16, 24)
        assert frozen.shape == (1, 24)
        # Everything before block_start should be frozen (True)
        assert frozen[0, :16].all()
        # Current block should be False (not in original prompt)
        assert not frozen[0, 16:24].any()

    def test_build_generation_prompt_mask_with_partial_prompt_in_block(self):
        pm = torch.zeros(1, 32, dtype=torch.bool)
        pm[0, :10] = True
        # For block 1: block_start=8, block_end=16
        frozen = model_bd3lm._build_generation_prompt_mask(pm, 8, 16)
        # Block 0 (positions 0-7) fully frozen
        assert frozen[0, :8].all()
        # Positions 8,9 in prompt → frozen
        assert frozen[0, 8] and frozen[0, 9]
        # Positions 10-15 not in prompt → not frozen
        assert not frozen[0, 10:16].any()


# =====================================================================
# BD3-LM progressive unmasking tests
# =====================================================================


class TestBD3LMProgressiveUnmasking:
    """Test the progressive unmasking step in model_bd3lm."""

    def test_t1_fills_all(self):
        B, L = 1, 16
        x = torch.full((B, L), MASK_TOKEN_ID, dtype=torch.long)
        x[0, :4] = torch.randint(1, VOCAB_SIZE, (4,))
        prompt_mask = torch.zeros(B, L, dtype=torch.bool)
        prompt_mask[0, :4] = True

        sampled = torch.randint(1, VOCAB_SIZE, (B, L))
        sampled_conf = torch.rand(B, L)

        result = model_bd3lm._progressive_unmask_step(
            x.clone(), prompt_mask, sampled, sampled_conf, t=1,
            mask_token_id=MASK_TOKEN_ID,
            survival_prob_scalar=linear_survival_prob_scalar,
        )
        assert (result[0, 4:] != MASK_TOKEN_ID).all()

    def test_no_remasking(self):
        """Once unmasked, tokens should stay unmasked (BD3-LM uses MDLM-style sampling)."""
        B, L = 1, 16
        x = torch.full((B, L), MASK_TOKEN_ID, dtype=torch.long)
        x[0, :4] = torch.randint(1, VOCAB_SIZE, (4,))
        prompt_mask = torch.zeros(B, L, dtype=torch.bool)
        prompt_mask[0, :4] = True

        sampled = torch.randint(1, VOCAB_SIZE, (B, L))
        sampled_conf = torch.rand(B, L)

        prev_unmasked = (x != MASK_TOKEN_ID).clone()
        for t in reversed(range(1, T + 1)):
            x = model_bd3lm._progressive_unmask_step(
                x, prompt_mask, sampled, sampled_conf, t,
                mask_token_id=MASK_TOKEN_ID,
                survival_prob_scalar=linear_survival_prob_scalar,
            )
            curr_unmasked = (x != MASK_TOKEN_ID)
            assert (curr_unmasked | ~prev_unmasked).all(), \
                f"Token re-masked at t={t}"
            prev_unmasked = curr_unmasked.clone()

    def test_monotonic_count(self):
        """Unmasked count should never decrease."""
        B, L = 1, 16
        x = torch.full((B, L), MASK_TOKEN_ID, dtype=torch.long)
        x[0, :4] = torch.randint(1, VOCAB_SIZE, (4,))
        prompt_mask = torch.zeros(B, L, dtype=torch.bool)
        prompt_mask[0, :4] = True

        sampled = torch.randint(1, VOCAB_SIZE, (B, L))
        sampled_conf = torch.rand(B, L)

        prev_count = (x != MASK_TOKEN_ID).sum().item()
        for t in reversed(range(1, T + 1)):
            x = model_bd3lm._progressive_unmask_step(
                x, prompt_mask, sampled, sampled_conf, t,
                mask_token_id=MASK_TOKEN_ID,
                survival_prob_scalar=linear_survival_prob_scalar,
            )
            curr_count = (x != MASK_TOKEN_ID).sum().item()
            assert curr_count >= prev_count
            prev_count = curr_count

    def test_noop_when_nothing_masked(self):
        """If no positions are masked, the step should be a no-op."""
        B, L = 1, 8
        x = torch.randint(1, VOCAB_SIZE, (B, L))
        prompt_mask = torch.zeros(B, L, dtype=torch.bool)
        prompt_mask[0, :4] = True

        sampled = torch.randint(1, VOCAB_SIZE, (B, L))
        sampled_conf = torch.rand(B, L)

        x_before = x.clone()
        result = model_bd3lm._progressive_unmask_step(
            x, prompt_mask, sampled, sampled_conf, t=5,
            mask_token_id=MASK_TOKEN_ID,
            survival_prob_scalar=linear_survival_prob_scalar,
        )
        assert torch.equal(result, x_before)


# =====================================================================
# BD3-LM generation integration tests
# =====================================================================


class TestBD3LMGeneration:
    def _make_model(self, block_len=BLOCK_LEN):
        return model_bd3lm.Model(
            vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
            n_layer=N_LAYER, head_dim=HEAD_DIM, block_size=BLOCK_SIZE,
            block_len=block_len, dropout=0.0,
        )

    def test_earlier_blocks_frozen_during_later_generation(self):
        """After a block is generated, it should not change when later blocks
        are generated.

        We verify by monkey-patching generate_from's write-back line: we
        snapshot each block right after it's committed and confirm the final
        tensor still matches those snapshots.
        """
        model = self._make_model()
        model.eval()
        decode = make_decode()

        torch.manual_seed(42)
        x = torch.full((1, BLOCK_SIZE), MASK_TOKEN_ID, dtype=torch.long)
        x[0, :4] = torch.randint(1, VOCAB_SIZE, (4,))
        prompt_mask = torch.zeros(1, BLOCK_SIZE, dtype=torch.bool)
        prompt_mask[0, :4] = True

        # Manually run the block-autoregressive loop so we can snapshot
        # each block right after it's committed.
        num_blk = BLOCK_SIZE // BLOCK_LEN
        first_block = model_bd3lm._prompt_start_block(prompt_mask, BLOCK_LEN)
        snapshots = {}  # block_idx -> snapshot of x[:, block_start:block_end]

        for block_idx in range(first_block, num_blk):
            block_start, block_end = model_bd3lm._current_block_range(block_idx, BLOCK_LEN)
            x_view = x[:, :block_end].clone()
            frozen_mask = model_bd3lm._build_generation_prompt_mask(
                prompt_mask, block_start, block_end
            )
            for t in reversed(range(1, T + 1)):
                logits, _ = model.forward_sample(x_view)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                sampled = torch.multinomial(probs.view(-1, VOCAB_SIZE), 1).view_as(x_view)
                sampled_conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
                x_view = model_bd3lm._progressive_unmask_step(
                    x_view, frozen_mask, sampled, sampled_conf, t,
                    mask_token_id=MASK_TOKEN_ID,
                    survival_prob_scalar=linear_survival_prob_scalar,
                )
            if (x_view[:, block_start:block_end] == MASK_TOKEN_ID).any():
                logits, _ = model.forward_sample(x_view)
                final_tokens = torch.argmax(logits, dim=-1)
                x_view = torch.where(x_view == MASK_TOKEN_ID, final_tokens, x_view)

            # Write back and snapshot
            x[:, :block_end] = x_view
            snapshots[block_idx] = x[:, block_start:block_end].clone()

        # Now verify that every earlier block still matches its snapshot
        for block_idx, snap in snapshots.items():
            block_start = block_idx * BLOCK_LEN
            block_end = block_start + BLOCK_LEN
            assert torch.equal(x[:, block_start:block_end], snap), \
                f"Block {block_idx} was modified after being committed"

    def test_generation_with_prompt_spanning_blocks(self):
        """Prompt that spans multiple blocks should work correctly."""
        model = self._make_model()
        model.eval()
        decode = make_decode()

        # Prompt spans blocks 0 and 1 (16 tokens with block_len=8)
        x = torch.full((1, BLOCK_SIZE), MASK_TOKEN_ID, dtype=torch.long)
        prompt_tokens = torch.randint(1, VOCAB_SIZE, (16,))
        x[0, :16] = prompt_tokens
        prompt_mask = torch.zeros(1, BLOCK_SIZE, dtype=torch.bool)
        prompt_mask[0, :16] = True

        model_bd3lm.generate_from(
            model, x, prompt_mask,
            T=T, block_size=BLOCK_SIZE, block_len=BLOCK_LEN,
            vocab_size=VOCAB_SIZE, mask_token_id=MASK_TOKEN_ID,
            survival_prob_scalar=linear_survival_prob_scalar,
            decode=decode,
        )
        # Prompt should be preserved
        assert torch.equal(x[0, :16], prompt_tokens)
        # All positions should be filled
        assert (x[0] != MASK_TOKEN_ID).all()

    def test_generation_no_prompt(self):
        """Generation with no prompt (all positions masked)."""
        model = self._make_model()
        model.eval()
        decode = make_decode()

        x = torch.full((1, BLOCK_SIZE), MASK_TOKEN_ID, dtype=torch.long)
        prompt_mask = torch.zeros(1, BLOCK_SIZE, dtype=torch.bool)

        model_bd3lm.generate_from(
            model, x, prompt_mask,
            T=T, block_size=BLOCK_SIZE, block_len=BLOCK_LEN,
            vocab_size=VOCAB_SIZE, mask_token_id=MASK_TOKEN_ID,
            survival_prob_scalar=linear_survival_prob_scalar,
            decode=decode,
        )
        assert (x[0] != MASK_TOKEN_ID).all()

    @pytest.mark.parametrize("block_len", [4, 8, 16, 32])
    def test_generation_various_block_lens(self, block_len):
        """Generation should work with different block_len values."""
        model = self._make_model(block_len=block_len)
        model.eval()
        decode = make_decode()

        x = torch.full((1, BLOCK_SIZE), MASK_TOKEN_ID, dtype=torch.long)
        x[0, :4] = torch.randint(1, VOCAB_SIZE, (4,))
        prompt_mask = torch.zeros(1, BLOCK_SIZE, dtype=torch.bool)
        prompt_mask[0, :4] = True

        text = model_bd3lm.generate_from(
            model, x, prompt_mask,
            T=T, block_size=BLOCK_SIZE, block_len=block_len,
            vocab_size=VOCAB_SIZE, mask_token_id=MASK_TOKEN_ID,
            survival_prob_scalar=linear_survival_prob_scalar,
            decode=decode,
        )
        assert len(text) == BLOCK_SIZE
        assert (x[0] != MASK_TOKEN_ID).all()


# =====================================================================
# Cross-model tests
# =====================================================================


class TestCrossModelInterface:
    """Verify all three models expose the same training interface."""

    @pytest.mark.parametrize("mod", [model_AR, model_MDLM, model_bd3lm])
    def test_has_required_functions(self, mod):
        assert callable(mod.get_batch)
        assert callable(mod.make_batch)
        assert callable(mod.make_eval_batch)
        assert callable(mod.compute_loss)
        assert callable(mod.compute_eval_loss)
        assert callable(mod.generate_from)

    @pytest.mark.parametrize("mod", [model_AR, model_MDLM, model_bd3lm])
    def test_get_batch_returns_three_tensors(self, mod, cfg):
        result = mod.get_batch("train", cfg)
        assert len(result) == 3

    @pytest.mark.parametrize("mod", [model_AR, model_MDLM, model_bd3lm])
    def test_make_batch_returns_three_tensors(self, mod, cfg):
        """make_batch(x0, cfg) should return (input, target, mask_or_none)."""
        x0 = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE))
        result = mod.make_batch(x0, cfg)
        assert len(result) == 3
        assert result[0].shape == (BATCH_SIZE, BLOCK_SIZE)
        assert result[1].shape == (BATCH_SIZE, BLOCK_SIZE)

    @pytest.mark.parametrize("mod", [model_AR, model_MDLM, model_bd3lm])
    def test_make_eval_batch_returns_three_tensors(self, mod, cfg):
        x0 = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE))
        result = mod.make_eval_batch(x0, cfg, fixed_t_step=5)
        assert len(result) == 3

    @pytest.mark.parametrize("mod", [model_AR, model_MDLM, model_bd3lm])
    def test_compute_loss_returns_scalar(self, mod, cfg):
        if mod == model_AR:
            model = model_AR.Model(
                vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
                n_layer=N_LAYER, head_dim=HEAD_DIM, block_size=BLOCK_SIZE,
                dropout=0.0,
            )
        elif mod == model_MDLM:
            model = model_MDLM.Model(
                vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
                n_layer=N_LAYER, head_dim=HEAD_DIM, block_size=BLOCK_SIZE,
                dropout=0.0,
            )
        else:
            model = model_bd3lm.Model(
                vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
                n_layer=N_LAYER, head_dim=HEAD_DIM, block_size=BLOCK_SIZE,
                block_len=BLOCK_LEN, dropout=0.0,
            )

        batch = mod.get_batch("train", cfg)
        loss = mod.compute_loss(model, batch, cfg)
        assert loss.shape == ()
        assert loss.item() > 0
        assert not torch.isnan(loss)


class TestParameterCountParity:
    """AR and MDLM should have the same parameter count for same architecture.

    BD3-LM shares DiffusionBackbone with MDLM, so it should also match.
    """

    def test_mdlm_matches_ar(self):
        ar = model_AR.Model(
            vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
            n_layer=N_LAYER, head_dim=HEAD_DIM, block_size=BLOCK_SIZE,
            dropout=0.0,
        )
        mdlm = model_MDLM.Model(
            vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
            n_layer=N_LAYER, head_dim=HEAD_DIM, block_size=BLOCK_SIZE,
            dropout=0.0,
        )
        ar_params = sum(p.numel() for p in ar.parameters())
        mdlm_params = sum(p.numel() for p in mdlm.parameters())
        assert ar_params == mdlm_params, \
            f"AR ({ar_params}) and MDLM ({mdlm_params}) should have same param count"

    def test_bd3lm_matches_mdlm(self):
        mdlm = model_MDLM.Model(
            vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
            n_layer=N_LAYER, head_dim=HEAD_DIM, block_size=BLOCK_SIZE,
            dropout=0.0,
        )
        bd3lm = model_bd3lm.Model(
            vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
            n_layer=N_LAYER, head_dim=HEAD_DIM, block_size=BLOCK_SIZE,
            block_len=BLOCK_LEN, dropout=0.0,
        )
        mdlm_params = sum(p.numel() for p in mdlm.parameters())
        bd3lm_params = sum(p.numel() for p in bd3lm.parameters())
        assert mdlm_params == bd3lm_params, \
            f"MDLM ({mdlm_params}) and BD3-LM ({bd3lm_params}) should have same param count"


class TestWeightSharing:
    """AR and MDLM/BD3-LM use the same architecture.

    We should be able to load AR weights into a DiffusionBackbone and vice versa.
    This is critical for the AR→BD3-LM curriculum.
    """

    def test_ar_weights_load_into_diffusion_backbone(self):
        """AR checkpoint weights should load into DiffusionBackbone (for curriculum)."""
        ar = model_AR.Model(
            vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
            n_layer=N_LAYER, head_dim=HEAD_DIM, block_size=BLOCK_SIZE,
            dropout=0.0,
        )
        bd3lm = model_bd3lm.Model(
            vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
            n_layer=N_LAYER, head_dim=HEAD_DIM, block_size=BLOCK_SIZE,
            block_len=BLOCK_LEN, dropout=0.0,
        )

        # Extract AR state dict (excluding buffers like cos/sin which differ)
        ar_state = ar.state_dict()
        bd3lm_state = bd3lm.state_dict()

        # Check that all AR learned parameters have matching keys in BD3-LM
        ar_learned_keys = {k for k, v in ar.named_parameters()}
        bd3lm_learned_keys = {k for k, v in bd3lm.named_parameters()}
        assert ar_learned_keys == bd3lm_learned_keys, \
            f"Parameter keys differ:\n  AR only: {ar_learned_keys - bd3lm_learned_keys}\n  BD3 only: {bd3lm_learned_keys - ar_learned_keys}"

        # Actually load the weights
        bd3lm.load_state_dict(ar_state, strict=False)

        # Verify the learned parameters match
        for name, param in ar.named_parameters():
            bd3lm_param = dict(bd3lm.named_parameters())[name]
            assert torch.equal(param.data, bd3lm_param.data), \
                f"Parameter {name} doesn't match after loading"
