"""The factory must dispatch to MultiModalForecasterModel when
``fusion_mode == 'gated_infonce'`` is set on the config (#235).

This pin guards against a refactor that silently routes the new
fusion mode back through the single-modality wrapper, which would
strip the gated fusion + InfoNCE alignment path entirely.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.models.config import ModelConfig
from app.models.factory import build_forecaster
from app.models.lstm import ForecasterModel
from app.models.multimodal_forecaster import MultiModalForecasterModel


def test_default_config_returns_legacy_forecaster_model() -> None:
    """``fusion_mode`` defaults to ``concat`` so the legacy path is
    the default — the dispatch must NOT silently switch on the new
    multi-modal model when the caller didn't ask for it."""

    cfg = ModelConfig(architecture="lstm")
    model = build_forecaster(cfg)
    assert isinstance(model, ForecasterModel)


def test_gated_infonce_config_returns_multimodal_model() -> None:
    cfg = ModelConfig(
        architecture="lstm",
        output_mode="classification",
        n_classes=3,
        text_embedding_dim=768,
        text_adapter_dim=64,
        fusion_mode="gated_infonce",
        infonce_lambda=0.2,
        infonce_temperature=0.05,
        infonce_latent_dim=32,
    )
    model = build_forecaster(cfg)
    assert isinstance(model, MultiModalForecasterModel)
    assert model.architecture == "lstm"
    assert model.text_embedding_dim == 768
    assert model.fusion.latent_dim == 32
    # Hyperparameters round-trip onto the module for the training loop + from_model.
    assert float(getattr(model, "infonce_lambda")) == 0.2
    assert float(getattr(model, "infonce_temperature")) == 0.05


def test_gated_infonce_requires_classification_mode() -> None:
    cfg = ModelConfig(
        architecture="lstm",
        output_mode="regression",  # invalid combo
        text_embedding_dim=768,
        fusion_mode="gated_infonce",
    )
    with pytest.raises(ValueError, match="output_mode='classification'"):
        build_forecaster(cfg)


def test_gated_infonce_requires_text_embedding_dim() -> None:
    cfg = ModelConfig(
        architecture="lstm",
        output_mode="classification",
        text_embedding_dim=0,  # text path not configured
        fusion_mode="gated_infonce",
    )
    with pytest.raises(ValueError, match="text_embedding_dim"):
        build_forecaster(cfg)


def test_model_config_from_model_round_trips_fusion_fields() -> None:
    cfg = ModelConfig(
        architecture="lstm",
        output_mode="classification",
        n_classes=3,
        text_embedding_dim=768,
        text_adapter_dim=64,
        fusion_mode="gated_infonce",
        infonce_lambda=0.25,
        infonce_temperature=0.08,
        infonce_latent_dim=128,
    )
    model = build_forecaster(cfg)
    restored = ModelConfig.from_model(model)
    assert restored.fusion_mode == "gated_infonce"
    assert restored.infonce_lambda == 0.25
    assert restored.infonce_temperature == 0.08
    assert restored.infonce_latent_dim == 128
