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


def test_informer_gradient_flows_to_every_parameter() -> None:
    """Widened from the earlier "at least one parameter" assertion.

    Every learnable parameter in the encoder must receive a strictly-
    positive gradient norm after a small synthetic training step against
    a noise target. A parameter that never receives a gradient is a
    latent bug — the module is functionally insensitive to that weight
    and the encoder is paying memory + compute for a dead unit. The
    earlier per-tensor "at least one" check would silently pass when
    that happened.
    """

    _fresh()
    encoder = InformerEncoder(input_size=FEATURE_SIZE, hidden_size=32)
    encoder.train()
    x = torch.randn(3, SEQUENCE_LENGTH, FEATURE_SIZE)
    target = torch.randn(3, SEQUENCE_LENGTH, 32)
    out, _ = encoder(x)
    # MSE against a non-zero target so every output position has a
    # gradient contribution. A bare `out.sum().backward()` lets every
    # output coordinate share the same scalar derivative, which can
    # silently zero-out gradients on parameters that only feed one
    # coordinate.
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
        "InformerEncoder has parameters with zero / missing gradient norm "
        "after one synthetic training step: " + ", ".join(zero_grad_names)
    )


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


def test_informer_forward_does_not_advance_global_rng() -> None:
    """ProbSparse sampling must use a per-instance generator, not the global RNG.

    If forward() calls torch.randint without an explicit generator, the
    global RNG state advances and breaks reproducibility for callers
    that rely on torch.manual_seed for upstream determinism.
    """

    torch.manual_seed(11)
    encoder = InformerEncoder(input_size=FEATURE_SIZE, hidden_size=32)
    encoder.eval()
    x = torch.randn(2, SEQUENCE_LENGTH, FEATURE_SIZE)

    rng_before = torch.random.get_rng_state().clone()
    with torch.no_grad():
        _ = encoder(x)
    rng_after = torch.random.get_rng_state().clone()
    assert torch.equal(rng_before, rng_after), (
        "InformerEncoder advanced the global RNG state during forward(); "
        "ProbSparse sampling must use a per-instance generator."
    )


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
