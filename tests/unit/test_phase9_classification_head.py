from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.models.config import ModelConfig
from app.models.factory import build_forecaster


# ---------------------------------------------------------------------------
# Phase 9 V2 (#195) classification head wiring
# ---------------------------------------------------------------------------


def test_default_modelconfig_stays_regression() -> None:
    """Regression byte-identity contract: default ModelConfig keeps
    the existing 2-output head shape."""
    mc = ModelConfig()
    assert mc.output_mode == "regression"
    assert mc.n_classes == 3   # tracked but unused in regression mode


def test_regression_head_emits_two_output_features() -> None:
    model = build_forecaster(ModelConfig(architecture="lstm"))
    assert model.output_mode == "regression"
    assert model.head[-1].out_features == 2
    x = torch.randn(4, 5, 6)
    out = model(x)
    assert out.shape == (4, 2)


def test_regression_volatility_stays_non_negative() -> None:
    """Softplus on the vol column keeps the regression contract."""
    model = build_forecaster(ModelConfig(architecture="lstm"))
    x = torch.randn(8, 5, 6) * 10  # push some columns into negative territory
    out = model(x)
    assert (out[:, 1] >= 0).all()


@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_classification_head_emits_n_class_logits(n_classes: int) -> None:
    """The multi-task head (#78) replaced the single classification
    Sequential. The stance branch carries the canonical 3-class
    (configurable) target the training loop reads via the model's
    primary forward output."""

    cfg = ModelConfig(
        architecture="lstm", output_mode="classification", n_classes=n_classes
    )
    model = build_forecaster(cfg)
    assert model.output_mode == "classification"
    assert model.n_classes == n_classes
    assert model.head.stance.out_features == n_classes
    x = torch.randn(4, 5, 6)
    out = model(x)
    assert out.shape == (4, n_classes)


def test_classification_head_does_not_clamp_negative_logits() -> None:
    """Logits stay raw -- no softplus. The training-loop CrossEntropy
    path will apply log_softmax internally."""
    cfg = ModelConfig(
        architecture="lstm", output_mode="classification", n_classes=3
    )
    model = build_forecaster(cfg)
    x = torch.randn(16, 5, 6) * 10
    out = model(x)
    # At least one logit somewhere should be negative -- if every
    # output is >= 0, something is clamping that shouldn't.
    assert (out < 0).any(), "classification logits unexpectedly all non-negative"


def test_classification_logits_route_through_softmax_cleanly() -> None:
    """Forward output is a valid CrossEntropy input (no NaN / Inf)."""
    cfg = ModelConfig(
        architecture="lstm", output_mode="classification", n_classes=3
    )
    model = build_forecaster(cfg)
    x = torch.randn(4, 5, 6)
    out = model(x)
    targets = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(out, targets)
    assert torch.isfinite(loss)


def test_unknown_output_mode_raises() -> None:
    with pytest.raises(ValueError, match="output_mode"):
        ModelConfig(architecture="lstm", output_mode="bogus")
        # ModelConfig is frozen; the validation lives in
        # ForecasterModel.__init__, so we construct via the factory:
        build_forecaster(ModelConfig(architecture="lstm").__class__(  # type: ignore[arg-type]
            architecture="lstm",
            output_mode="bogus",   # bad value
        ))


def test_classification_requires_n_classes_at_least_two() -> None:
    """Single-class classification has no decision boundary."""
    with pytest.raises(ValueError, match="n_classes"):
        build_forecaster(
            ModelConfig(architecture="lstm", output_mode="classification", n_classes=1)
        )


def test_modelconfig_round_trips_classification_fields() -> None:
    cfg = ModelConfig(
        architecture="lstm",
        output_mode="classification",
        n_classes=3,
        vol_regime_quantiles=(0.0015, 0.0042),
        vol_regime_target="forward_realized_vol_10d",
    )
    payload = cfg.to_dict()
    assert payload["output_mode"] == "classification"
    assert payload["n_classes"] == 3
    assert payload["vol_regime_quantiles"] == (0.0015, 0.0042)
    assert payload["vol_regime_target"] == "forward_realized_vol_10d"


def test_factory_persists_vol_regime_quantiles_onto_model() -> None:
    """Per-fold quantile cutoffs must ride on the built module so the
    checkpoint round-trip via ``ModelConfig.from_model`` recovers them."""

    cutoffs = (0.0015, 0.0042)
    model = build_forecaster(
        ModelConfig(
            architecture="lstm",
            output_mode="classification",
            n_classes=3,
            vol_regime_quantiles=cutoffs,
        )
    )
    assert model.vol_regime_quantiles == cutoffs
    assert model.vol_regime_target == "forward_realized_vol_10d"
    # The reverse trip must give back the same tuple bit-for-bit.
    restored = ModelConfig.from_model(model)
    assert restored.vol_regime_quantiles == cutoffs
    assert restored.output_mode == "classification"
    assert restored.n_classes == 3
