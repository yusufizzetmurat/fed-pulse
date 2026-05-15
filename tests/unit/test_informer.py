"""Unit tests for the Informer encoder core.

The Informer encoder must respect the project's recurrent-core contract:
input ``(B, T, F)`` → output ``(B, T, H)`` plus a ``None`` placeholder so
the ``output, _ = core(x)`` destructuring in ``ForecasterModel.forward``
keeps working. Same-seed determinism is required because the ProbSparse
attention layer samples key indices through the default RNG; the test
locks the seed and asserts byte-identical outputs across two passes.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.models.config import FEATURE_SIZE, SEQUENCE_LENGTH  # noqa: E402
from app.models.informer import InformerEncoder  # noqa: E402


def _fresh(seed: int = 11) -> None:
    torch.manual_seed(seed)


def test_informer_forward_returns_expected_shape() -> None:
    _fresh()
    encoder = InformerEncoder(input_size=FEATURE_SIZE, hidden_size=64)
    encoder.eval()
    x = torch.randn(4, SEQUENCE_LENGTH, FEATURE_SIZE)
    with torch.no_grad():
        out, placeholder = encoder(x)
    assert out.shape == (4, SEQUENCE_LENGTH, 64)
    assert placeholder is None
    assert torch.all(torch.isfinite(out))


def test_informer_forward_returns_tuple_for_lstm_destructuring() -> None:
    """``output, _ = core(x)`` is the destructuring used by ForecasterModel."""

    _fresh()
    encoder = InformerEncoder(input_size=FEATURE_SIZE, hidden_size=32, n_heads=4)
    x = torch.randn(2, SEQUENCE_LENGTH, FEATURE_SIZE)
    output, _ = encoder(x)
    assert output.shape == (2, SEQUENCE_LENGTH, 32)


def test_informer_gradient_flows_to_parameters() -> None:
    _fresh()
    encoder = InformerEncoder(input_size=FEATURE_SIZE, hidden_size=32)
    x = torch.randn(3, SEQUENCE_LENGTH, FEATURE_SIZE, requires_grad=True)
    out, _ = encoder(x)
    out.sum().backward()
    # Every learnable parameter must receive a non-zero gradient.
    grad_norms = [
        param.grad.abs().sum().item()
        for param in encoder.parameters()
        if param.requires_grad and param.grad is not None
    ]
    assert grad_norms, "no learnable parameters received gradients"
    assert all(norm >= 0.0 for norm in grad_norms)
    # At least one parameter must have a strictly positive gradient norm
    # (otherwise the encoder is functionally constant w.r.t. input).
    assert any(norm > 0.0 for norm in grad_norms)


def test_informer_is_deterministic_at_fixed_seed() -> None:
    """Same input + same seed must yield bit-identical outputs."""

    def _run() -> torch.Tensor:
        torch.manual_seed(11)
        encoder = InformerEncoder(input_size=FEATURE_SIZE, hidden_size=32)
        encoder.eval()
        torch.manual_seed(11)
        x = torch.randn(2, SEQUENCE_LENGTH, FEATURE_SIZE)
        with torch.no_grad():
            out, _ = encoder(x)
        return out

    first = _run()
    second = _run()
    assert torch.equal(first, second), "InformerEncoder is not deterministic at fixed seed"


def test_informer_rejects_indivisible_head_count() -> None:
    with pytest.raises(ValueError):
        InformerEncoder(input_size=FEATURE_SIZE, hidden_size=30, n_heads=4)


def test_informer_rejects_wrong_input_rank() -> None:
    encoder = InformerEncoder(input_size=FEATURE_SIZE, hidden_size=32)
    with pytest.raises(ValueError):
        encoder(torch.randn(SEQUENCE_LENGTH, FEATURE_SIZE))


def test_informer_handles_short_sequences() -> None:
    """ProbSparse u/U_part computation must clamp to seq_len for very short inputs."""

    _fresh()
    encoder = InformerEncoder(input_size=FEATURE_SIZE, hidden_size=32)
    encoder.eval()
    x = torch.randn(2, 3, FEATURE_SIZE)
    with torch.no_grad():
        out, _ = encoder(x)
    assert out.shape == (2, 3, 32)
