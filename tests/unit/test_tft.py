"""Unit tests for the Temporal Fusion Transformer encoder.

The TFT encoder must respect the project's recurrent-core contract:
input ``(B, T, F)`` → output ``(B, T, H)`` plus a ``None`` placeholder so
the ``output, _ = core(x)`` destructuring in ``ForecasterModel.forward``
keeps working. Same-seed determinism is required so that the official
seed set produces reproducible results during the sweep.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.models.config import FEATURE_SIZE, SEQUENCE_LENGTH  # noqa: E402
from app.models.tft import TFTEncoder  # noqa: E402


def _fresh(seed: int = 11) -> None:
    torch.manual_seed(seed)


def test_tft_forward_returns_expected_shape() -> None:
    _fresh()
    encoder = TFTEncoder(input_size=FEATURE_SIZE, hidden_size=64)
    encoder.eval()
    x = torch.randn(4, SEQUENCE_LENGTH, FEATURE_SIZE)
    with torch.no_grad():
        out, placeholder = encoder(x)
    assert out.shape == (4, SEQUENCE_LENGTH, 64)
    assert placeholder is None
    assert torch.all(torch.isfinite(out))


def test_tft_forward_returns_tuple_for_lstm_destructuring() -> None:
    """``output, _ = core(x)`` is the destructuring used by ForecasterModel."""

    _fresh()
    encoder = TFTEncoder(input_size=FEATURE_SIZE, hidden_size=32, n_heads=4)
    x = torch.randn(2, SEQUENCE_LENGTH, FEATURE_SIZE)
    output, _ = encoder(x)
    assert output.shape == (2, SEQUENCE_LENGTH, 32)


def test_tft_gradient_flows_to_parameters() -> None:
    _fresh()
    encoder = TFTEncoder(input_size=FEATURE_SIZE, hidden_size=32)
    x = torch.randn(3, SEQUENCE_LENGTH, FEATURE_SIZE, requires_grad=True)
    out, _ = encoder(x)
    out.sum().backward()
    grad_norms = [
        param.grad.abs().sum().item()
        for param in encoder.parameters()
        if param.requires_grad and param.grad is not None
    ]
    assert grad_norms, "no learnable parameters received gradients"
    assert any(norm > 0.0 for norm in grad_norms)


def test_tft_is_deterministic_at_fixed_seed() -> None:
    """Same input + same seed must yield bit-identical outputs."""

    def _run() -> torch.Tensor:
        torch.manual_seed(11)
        encoder = TFTEncoder(input_size=FEATURE_SIZE, hidden_size=32)
        encoder.eval()
        torch.manual_seed(11)
        x = torch.randn(2, SEQUENCE_LENGTH, FEATURE_SIZE)
        with torch.no_grad():
            out, _ = encoder(x)
        return out

    first = _run()
    second = _run()
    assert torch.equal(first, second), "TFTEncoder is not deterministic at fixed seed"


def test_tft_rejects_indivisible_head_count() -> None:
    with pytest.raises(ValueError):
        TFTEncoder(input_size=FEATURE_SIZE, hidden_size=30, n_heads=4)


def test_tft_rejects_wrong_input_rank() -> None:
    encoder = TFTEncoder(input_size=FEATURE_SIZE, hidden_size=32)
    with pytest.raises(ValueError):
        encoder(torch.randn(SEQUENCE_LENGTH, FEATURE_SIZE))


def test_tft_rejects_feature_dim_mismatch() -> None:
    """VSN is built for a specific num_inputs; mismatched feature dim must error."""

    encoder = TFTEncoder(input_size=FEATURE_SIZE, hidden_size=32)
    with pytest.raises(ValueError):
        encoder(torch.randn(2, SEQUENCE_LENGTH, FEATURE_SIZE + 1))
