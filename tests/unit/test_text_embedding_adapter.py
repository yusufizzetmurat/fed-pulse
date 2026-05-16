from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
from torch import nn  # noqa: E402

from app.models.text_embedding_adapter import TextEmbeddingAdapter


def test_adapter_shape_invariant() -> None:
    """Pooled (batch, 1024) projects to (batch, 64) regardless of batch."""

    adapter = TextEmbeddingAdapter(in_dim=1024, out_dim=64, zero_init=False)
    pooled = torch.randn(16, 1024)
    out = adapter(pooled)
    assert out.shape == (16, 64)
    assert out.dtype == pooled.dtype


def test_adapter_layernorm_then_gelu() -> None:
    """Layer order is Linear -> LayerNorm -> GELU.

    Hooks fire on the underlying submodules during the forward pass and
    must be invoked in the documented order.
    """

    adapter = TextEmbeddingAdapter(in_dim=768, out_dim=32, zero_init=False)
    fired: list[str] = []

    def _record(name: str):
        def _hook(_module: nn.Module, _inputs, _output) -> None:
            fired.append(name)

        return _hook

    adapter.linear.register_forward_hook(_record("linear"))
    adapter.norm.register_forward_hook(_record("norm"))
    adapter.activation.register_forward_hook(_record("activation"))

    adapter(torch.randn(4, 768))
    assert fired == ["linear", "norm", "activation"]


def test_adapter_handles_zero_input_without_nan() -> None:
    """All-zero pooled input passes cleanly through the adapter.

    LayerNorm of a zero tensor maps to zero (with eps in the
    denominator); GELU(0) is 0. The adapter must emit a finite tensor
    even on a row whose pooled embedding is all zeros (the
    ``text_embedding_missing == 1.0`` case the loader writes when there
    are fewer than one prior statement).
    """

    adapter = TextEmbeddingAdapter(in_dim=512, out_dim=64, zero_init=True)
    pooled = torch.zeros(2, 512)
    out = adapter(pooled)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all().item()
    # zero_init plus zero input collapses to a zero tensor through the
    # whole stack.
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)


def test_adapter_rejects_wrong_in_dim() -> None:
    adapter = TextEmbeddingAdapter(in_dim=768, out_dim=32, zero_init=False)
    with pytest.raises(ValueError):
        adapter(torch.randn(2, 1024))


def test_adapter_rejects_non_positive_dims() -> None:
    with pytest.raises(ValueError):
        TextEmbeddingAdapter(in_dim=0, out_dim=64)
    with pytest.raises(ValueError):
        TextEmbeddingAdapter(in_dim=768, out_dim=0)


def test_adapter_1d_input_is_unsqueezed() -> None:
    """A single un-batched pooled vector is accepted and projected."""

    adapter = TextEmbeddingAdapter(in_dim=768, out_dim=64, zero_init=False)
    out = adapter(torch.randn(768))
    assert out.shape == (1, 64)
