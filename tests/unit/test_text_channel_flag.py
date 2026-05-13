from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from app.models.embedding_adapter import EmbeddingAdapter
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
    assert isinstance(model.chunk_projection, EmbeddingAdapter)
    assert model.chunk_projection.out_features == 128


def test_invalid_text_channel_raises():
    with pytest.raises(ValueError, match="Unknown text_channel"):
        ForecasterModel(text_channel="hybrid")


def test_coerce_model_config_preserves_text_channel_from_dict():
    from app.services.forecaster import ModelConfig, _coerce_model_config

    config_dict = {
        "hidden_size": 32,
        "text_channel": "embeddings",
        "embedding_adapter_dim": 64,
    }
    coerced = _coerce_model_config(config_dict)
    assert isinstance(coerced, ModelConfig)
    assert coerced.text_channel == "embeddings"
    assert coerced.embedding_adapter_dim == 64


def test_coerce_model_config_defaults_legacy_dicts_to_scalar():
    from app.services.forecaster import _coerce_model_config

    legacy_dict = {"hidden_size": 32, "num_layers": 2}
    coerced = _coerce_model_config(legacy_dict)
    assert coerced.text_channel == "scalar"
    assert coerced.embedding_adapter_dim == 128


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
