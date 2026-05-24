"""Cover the MultiModalForecasterModel (#235).

The new model wires the recurrent core + gated fusion + classification
head into a single forward path. Tests pin the output shapes, the
text-missing zeroing contract, and round-trip consistency between
``forward`` and ``forward_with_modality_outputs``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.models.multimodal_forecaster import MultiModalForecasterModel


def _build_model(architecture: str = "lstm", **kwargs) -> MultiModalForecasterModel:
    defaults = dict(
        market_input_size=8,
        text_embedding_dim=16,
        latent_dim=4,
        hidden_size=12,
        num_layers=2,
        dropout=0.1,
        head_hidden_size=6,
        architecture=architecture,
        n_classes=3,
    )
    defaults.update(kwargs)
    return MultiModalForecasterModel(**defaults)


def test_forward_returns_classification_logits_with_expected_shape() -> None:
    torch.manual_seed(0)
    model = _build_model()
    x = torch.randn(4, 5, 8)
    text = torch.randn(4, 16)
    logits = model(x, text_embedding=text)
    assert logits.shape == (4, 3)


def test_forward_with_modality_outputs_emits_modality_dict() -> None:
    torch.manual_seed(1)
    model = _build_model()
    x = torch.randn(4, 5, 8)
    text = torch.randn(4, 16)
    out = model.forward_with_modality_outputs(x, text_embedding=text)
    assert set(out.keys()) == {"logits", "r_t", "t_t", "fused", "gate"}
    assert out["logits"].shape == (4, 3)
    assert out["r_t"].shape == (4, 4)
    assert out["t_t"].shape == (4, 4)
    assert out["fused"].shape == (4, 4)


def test_text_missing_flag_zeros_the_text_input_before_fusion() -> None:
    """When ``text_embedding_missing=1`` the fusion must see a zero
    text vector regardless of what the loader actually emitted."""

    torch.manual_seed(2)
    model = _build_model()
    model.eval()  # disable dropout so the two calls compare cleanly
    x = torch.randn(2, 5, 8)
    text_nonzero = torch.randn(2, 16)
    missing_flag = torch.ones(2, 1)  # both rows missing
    out_missing = model.forward_with_modality_outputs(
        x, text_embedding=text_nonzero, text_embedding_missing=missing_flag
    )
    # Same model, same x, but text manually zeroed; outputs must match
    text_zero = torch.zeros(2, 16)
    out_zero = model.forward_with_modality_outputs(x, text_embedding=text_zero)
    assert torch.allclose(out_missing["logits"], out_zero["logits"], atol=1e-6)
    assert torch.allclose(out_missing["t_t"], out_zero["t_t"], atol=1e-6)


def test_forward_and_forward_with_modality_outputs_agree_on_logits() -> None:
    torch.manual_seed(3)
    model = _build_model()
    model.eval()  # avoid dropout randomness across the two calls
    x = torch.randn(3, 5, 8)
    text = torch.randn(3, 16)
    logits_a = model(x, text_embedding=text)
    out_b = model.forward_with_modality_outputs(x, text_embedding=text)
    assert torch.allclose(logits_a, out_b["logits"], atol=1e-6)


@pytest.mark.parametrize("architecture", ["lstm", "gru", "transformer", "tcn"])
def test_forward_works_across_supported_architectures(architecture: str) -> None:
    """The fusion + classification head should work regardless of which
    recurrent core feeds the market projection."""

    torch.manual_seed(4)
    model = _build_model(architecture=architecture)
    x = torch.randn(2, 5, 8)
    text = torch.randn(2, 16)
    out = model.forward_with_modality_outputs(x, text_embedding=text)
    assert out["logits"].shape == (2, 3)


def test_constructor_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="Unknown architecture"):
        _build_model(architecture="bogus")


def test_constructor_rejects_single_class() -> None:
    with pytest.raises(ValueError, match="n_classes"):
        _build_model(n_classes=1)
