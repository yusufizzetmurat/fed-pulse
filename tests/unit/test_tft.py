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


def test_tft_gradient_flows_to_every_parameter() -> None:
    """Widened from the earlier "at least one parameter" assertion.

    Every learnable parameter in the TFT encoder must receive a strictly
    positive gradient norm after one synthetic training step against a
    noise target. The earlier check only proved *some* parameter saw a
    gradient, which would let a dead-weight module slip through.
    """

    _fresh()
    encoder = TFTEncoder(input_size=FEATURE_SIZE, hidden_size=32)
    encoder.train()
    x = torch.randn(3, SEQUENCE_LENGTH, FEATURE_SIZE)
    target = torch.randn(3, SEQUENCE_LENGTH, 32)
    out, _ = encoder(x)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()

    zero_grad_names: list[str] = []
    for name, param in encoder.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            zero_grad_names.append(f"{name}=None")
            continue
        norm = float(param.grad.abs().sum().item())
        if not (norm > 0.0):
            zero_grad_names.append(f"{name}={norm:.3e}")
    assert not zero_grad_names, (
        "TFTEncoder has parameters with zero / missing gradient norm "
        "after one synthetic training step: " + ", ".join(zero_grad_names)
    )


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
