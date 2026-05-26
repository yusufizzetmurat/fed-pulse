from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from app.models.embedding_adapter import EmbeddingAdapter


def test_adapter_output_shape_and_dtype():
    adapter = EmbeddingAdapter(input_dim=768, output_dim=128)
    pooled = torch.randn(4, 768)
    out = adapter(pooled)
    assert out.shape == (4, 128)
    assert out.dtype == pooled.dtype


def test_zero_init_recovers_zero_activation_pre_norm():
    adapter = EmbeddingAdapter(input_dim=768, output_dim=128, zero_init=True)
    pooled = torch.randn(2, 768)
    # Bias zeroed and weight zeroed → pre-LN activations are zero. LN of zero
    # tensors maps to zero (with eps), then GELU(0) = 0.
    out = adapter(pooled)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)


def test_gradient_flows_through_adapter():
    adapter = EmbeddingAdapter(input_dim=768, output_dim=128, zero_init=False)
    pooled = torch.randn(3, 768, requires_grad=True)
    out = adapter(pooled)
    loss = out.pow(2).mean()
    loss.backward()
    assert adapter.linear.weight.grad is not None
    assert adapter.linear.weight.grad.abs().sum().item() > 0
