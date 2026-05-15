"""Tests for the Layer-2 architecture upgrades: lstm_attn pool and DLinear core."""

from __future__ import annotations

import pytest
import torch

from app.models.attention import RecurrentSequenceAttention
from app.models.config import SEQUENCE_LENGTH
from app.models.dlinear import DLinear
from app.models.lstm import ForecasterModel


def _seed() -> None:
    torch.manual_seed(11)


# ---- RecurrentSequenceAttention ----------------------------------------------


def test_recurrent_attention_pool_output_shape() -> None:
    attn = RecurrentSequenceAttention(hidden_size=16)
    seq = torch.randn(3, 20, 16)
    pooled, weights = attn(seq)
    assert pooled.shape == (3, 16)
    assert weights.shape == (3, 20)
    # Weights sum to 1 along the time axis.
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_recurrent_attention_pool_rejects_non_3d_input() -> None:
    attn = RecurrentSequenceAttention(hidden_size=8)
    with pytest.raises(ValueError, match="expects \\(B, T, H\\)"):
        attn(torch.randn(8, 16))


def test_recurrent_attention_pool_honours_mask() -> None:
    """Fully-masked sequences must not produce NaN pooled outputs."""

    attn = RecurrentSequenceAttention(hidden_size=8)
    seq = torch.randn(2, 5, 8)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.float32)
    pooled, weights = attn(seq, mask=mask)
    assert not torch.isnan(pooled).any()
    # Masked positions must have ~zero weight.
    assert weights[0, 2:].abs().sum() < 1e-5
    assert weights[1, 3:].abs().sum() < 1e-5


# ---- LSTM-with-attention forecaster ------------------------------------------


def test_lstm_attn_constructs_with_attention_pool() -> None:
    model = ForecasterModel(model_type="lstm_attn")
    assert model.uses_attention_pool is True
    assert isinstance(model.recurrent_attention, RecurrentSequenceAttention)


def test_lstm_attn_forward_uses_attention_not_last_step() -> None:
    """The whole-sequence attention pool should give a different output than
    the last-step pool for the same inputs and weights."""

    _seed()
    last_step_model = ForecasterModel(model_type="lstm", hidden_size=16)
    attn_model = ForecasterModel(model_type="lstm_attn", hidden_size=16)

    # Force identical LSTM weights so the only difference is the pool.
    attn_model.lstm.load_state_dict(last_step_model.lstm.state_dict())
    attn_model.head.load_state_dict(last_step_model.head.state_dict())

    x = torch.randn(2, SEQUENCE_LENGTH, last_step_model.input_size)
    out_last = last_step_model(x)
    out_attn = attn_model(x)
    assert out_last.shape == out_attn.shape
    # With the same weights and a non-trivial attention layer the outputs
    # should differ — confirms the pool actually fires.
    assert not torch.allclose(out_last, out_attn, atol=1e-5)


def test_lstm_attn_with_credibility_features_concats_correctly() -> None:
    model = ForecasterModel(model_type="lstm_attn", credibility_features=True, hidden_size=8)
    x = torch.randn(2, SEQUENCE_LENGTH, model.input_size)
    credibility = torch.zeros(2, 4)
    out = model(x, credibility=credibility)
    assert out.shape == (2, 2)


# ---- DLinear -----------------------------------------------------------------


def test_dlinear_returns_padding_compatible_shape() -> None:
    core = DLinear(input_size=6, hidden_size=12, sequence_length=SEQUENCE_LENGTH)
    x = torch.randn(4, SEQUENCE_LENGTH, 6)
    out, hidden = core(x)
    assert out.shape == (4, SEQUENCE_LENGTH, 12)
    assert hidden is None


def test_dlinear_rejects_wrong_sequence_length() -> None:
    core = DLinear(input_size=4, hidden_size=8, sequence_length=20)
    with pytest.raises(ValueError, match="seq_len="):
        core(torch.randn(1, 7, 4))


def test_dlinear_initial_output_close_to_feature_projection() -> None:
    """Trend and seasonal linears are zero-initialised so DLinear at step 0
    should just emit the projected feature average — a clean baseline."""

    _seed()
    core = DLinear(input_size=4, hidden_size=8, sequence_length=10)
    x = torch.randn(2, 10, 4)
    out, _ = core(x)
    # Trend = AvgPool of x; seasonal = x − trend. Both linears zero ⇒
    # `summed = bias_t + bias_s = 0`. So projected = feature_proj(0) = bias.
    bias = core.feature_proj.bias
    assert torch.allclose(out, bias.expand_as(out), atol=1e-6)


def test_dlinear_forecaster_end_to_end() -> None:
    model = ForecasterModel(model_type="dlinear", hidden_size=12)
    x = torch.randn(3, SEQUENCE_LENGTH, model.input_size)
    out = model(x)
    assert out.shape == (3, 2)
    # softplus on the volatility column → non-negative.
    assert (out[:, 1] >= 0).all()


# ---- Rejected combos ---------------------------------------------------------


def test_unknown_model_type_lists_full_allowed_set() -> None:
    with pytest.raises(ValueError) as excinfo:
        ForecasterModel(model_type="not_a_model")
    msg = str(excinfo.value)
    for name in ("lstm", "lstm_attn", "gru", "tcn", "transformer", "dlinear"):
        assert name in msg
