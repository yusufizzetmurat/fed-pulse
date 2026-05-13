from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from app.services.forecaster import ForecasterModel


def test_scalar_channel_keeps_legacy_projection_dim():
    model = ForecasterModel(
        use_chunk_attention=True,
        use_llm_embeddings=False,
        text_channel="scalar",
    )
    assert model.chunk_projection_dim == 8
    # Legacy path uses a plain Linear with zero-init weights.
    assert isinstance(model.chunk_projection, torch.nn.Linear)
    assert torch.allclose(model.chunk_projection.weight, torch.zeros_like(model.chunk_projection.weight))


def test_embeddings_channel_uses_adapter():
    model = ForecasterModel(
        use_chunk_attention=False,
        use_llm_embeddings=True,
        text_channel="embeddings",
        embedding_adapter_dim=128,
    )
    assert model.chunk_projection_dim == 128
    # Adapter is the EmbeddingAdapter module, not a plain Linear.
    assert type(model.chunk_projection).__name__ == "EmbeddingAdapter"


def test_invalid_text_channel_raises():
    with pytest.raises(ValueError, match="Unknown text_channel"):
        ForecasterModel(text_channel="hybrid")


def test_forward_shape_with_adapter_active():
    model = ForecasterModel(
        use_llm_embeddings=True,
        text_channel="embeddings",
        embedding_adapter_dim=128,
    )
    batch = 2
    seq_len = 5
    base = torch.randn(batch, seq_len, model.input_size)
    chunk_count = 4
    chunks = torch.randn(batch, chunk_count, model.chunk_embedding_size)
    elapsed = torch.linspace(0.1, 1.0, chunk_count).expand(batch, -1)
    out = model(base, chunks=chunks, elapsed_days=elapsed)
    assert out.shape == (batch, 2)
