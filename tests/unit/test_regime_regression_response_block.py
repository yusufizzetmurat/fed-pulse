"""Cover the /analyze ``regime_regression`` sibling block (#304).

The dual-head retrofit keeps the classification card as the headline
on the response and adds a sibling :class:`RegimeRegressionCard` so a
downstream consumer can read the regression head's point estimate +
90% conformal interval as a standalone surface. The block is
populated only when the active checkpoint mounts the regression head
(``head_mode`` in ``regression`` / ``dual``) AND the classification
card carries a non-null ``log_rv_point``; otherwise the field stays
``None`` and the legacy classification-only payload shape is byte-
identical to pre-#304.

These tests pin:

- the schema field exists and validates against the new
  :class:`RegimeRegressionCard`;
- ``_build_regime_regression_block`` derives the block correctly from
  a dual-head classification card;
- the helper short-circuits to ``None`` on a classification-only card
  (or one whose regression head did not run);
- the checkpoint round-trip preserves the four new
  ``EvaluationMetrics`` regression fields (RMSE / MAE / loss / R^2)
  so an aggregator reading per-trial JSONs gets the full surface
  back.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Schema surface


def test_analyze_response_carries_regime_regression_field() -> None:
    """The :class:`AnalyzeResponse` schema must declare the new sibling field."""

    from app.schemas import AnalyzeResponse

    assert "regime_regression" in AnalyzeResponse.model_fields
    field = AnalyzeResponse.model_fields["regime_regression"]
    # Optional + defaults to None so the classification-only payload
    # stays byte-identical to the pre-#304 shape on every legacy run.
    assert field.default is None


def test_regime_regression_card_validates_point_only() -> None:
    """A point-only card (no conformal sidecar on disk) must validate."""

    from app.schemas import RegimeRegressionCard

    card = RegimeRegressionCard(log_rv_point=-3.5)
    assert card.log_rv_point == pytest.approx(-3.5)
    assert card.log_rv_lower is None
    assert card.log_rv_upper is None
    assert card.coverage is None


def test_regime_regression_card_validates_full_interval() -> None:
    """A populated conformal interval round-trips through the schema."""

    from app.schemas import RegimeRegressionCard

    card = RegimeRegressionCard(
        log_rv_point=-3.5,
        log_rv_lower=-3.8,
        log_rv_upper=-3.2,
        coverage=0.9,
    )
    assert card.log_rv_lower == pytest.approx(-3.8)
    assert card.log_rv_upper == pytest.approx(-3.2)
    assert card.coverage == pytest.approx(0.9)


def test_regime_regression_card_rejects_missing_point() -> None:
    """``log_rv_point`` is required — the block is meaningless without it."""

    from app.schemas import RegimeRegressionCard

    with pytest.raises(ValidationError):
        RegimeRegressionCard()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# /analyze block derivation


def test_build_regime_regression_block_returns_none_for_classification_only() -> None:
    """Classification-only card (no log_rv_point) yields ``None``."""

    from app.main import _build_regime_regression_block

    card = {
        "predicted_set": ["normal", "high"],
        "set_label": "{normal, high}",
        "set_size": 2,
        "coverage": 0.8,
        "distribution": {"calm": 0.18, "normal": 0.52, "high": 0.30},
        "argmax_class": "normal",
        "log_rv_point": None,
        "log_rv_lower": None,
        "log_rv_upper": None,
        "bucket_source": "classification",
    }
    assert _build_regime_regression_block(card) is None


def test_build_regime_regression_block_returns_none_for_none_input() -> None:
    """A degraded /analyze (no classification card) emits no sibling block."""

    from app.main import _build_regime_regression_block

    assert _build_regime_regression_block(None) is None


def test_build_regime_regression_block_passes_through_dual_head_card() -> None:
    """Dual-head card carries log_rv_point + conformal bounds onto the block."""

    from app.main import _build_regime_regression_block

    card = {
        "predicted_set": ["normal"],
        "set_label": "{normal}",
        "set_size": 1,
        "coverage": 0.9,
        "distribution": {"calm": 0.10, "normal": 0.70, "high": 0.20},
        "argmax_class": "normal",
        "log_rv_point": -3.42,
        "log_rv_lower": -3.71,
        "log_rv_upper": -3.13,
        "bucket_source": "regression",
    }
    block = _build_regime_regression_block(card)
    assert block is not None
    assert block["log_rv_point"] == pytest.approx(-3.42)
    assert block["log_rv_lower"] == pytest.approx(-3.71)
    assert block["log_rv_upper"] == pytest.approx(-3.13)
    # Coverage rides off the classification card's nominal coverage
    # because the regression interval re-uses the same conformal
    # manifest (residual_quantile_volatility) today.
    assert block["coverage"] == pytest.approx(0.9)


def test_build_regime_regression_block_point_only_drops_coverage() -> None:
    """A regression card without a conformal sidecar emits no coverage value."""

    from app.main import _build_regime_regression_block

    card = {
        "predicted_set": ["normal"],
        "set_label": "{normal}",
        "set_size": 1,
        "coverage": 0.0,  # cold-start checkpoint, no manifest
        "distribution": {"calm": 0.10, "normal": 0.70, "high": 0.20},
        "argmax_class": "normal",
        "log_rv_point": -3.42,
        "log_rv_lower": None,
        "log_rv_upper": None,
        "bucket_source": "regression",
    }
    block = _build_regime_regression_block(card)
    assert block is not None
    assert block["log_rv_point"] == pytest.approx(-3.42)
    assert block["log_rv_lower"] is None
    assert block["log_rv_upper"] is None
    assert "coverage" not in block


def test_build_regime_regression_block_card_round_trips_through_schema() -> None:
    """The derived dict must hydrate cleanly into the pydantic card."""

    from app.main import _build_regime_regression_block
    from app.schemas import RegimeRegressionCard

    card = {
        "log_rv_point": -3.42,
        "log_rv_lower": -3.71,
        "log_rv_upper": -3.13,
        "coverage": 0.9,
        "argmax_class": "normal",
        "predicted_set": ["normal"],
        "set_label": "{normal}",
        "set_size": 1,
        "distribution": {"calm": 0.1, "normal": 0.7, "high": 0.2},
        "bucket_source": "regression",
    }
    block = _build_regime_regression_block(card)
    assert block is not None
    hydrated = RegimeRegressionCard(**block)
    assert hydrated.log_rv_point == pytest.approx(-3.42)
    assert hydrated.coverage == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Checkpoint round-trip on the dual-head metrics surface


def test_evaluation_metrics_carries_regression_r2_log_rv_field() -> None:
    """Per acceptance: R^2 on log_rv joins the existing RMSE / MAE pair."""

    from app.evaluation.metrics import EvaluationMetrics

    metrics = EvaluationMetrics(
        loss=0.5,
        close_rmse=float("inf"),
        volatility_rmse=float("inf"),
        combined_rmse=float("inf"),
        regression_rmse_log_rv=0.42,
        regression_mae_log_rv=0.31,
        regression_loss=0.18,
        regression_r2_log_rv=0.27,
    )
    assert metrics.regression_r2_log_rv == pytest.approx(0.27)
    # Default stays None so the legacy contract holds on every pre-
    # #304 caller that does not pass the kwarg.
    default = EvaluationMetrics(
        loss=0.0,
        close_rmse=0.0,
        volatility_rmse=0.0,
        combined_rmse=0.0,
    )
    assert default.regression_r2_log_rv is None


def test_metrics_from_payload_round_trips_regression_fields() -> None:
    """The four #304 regression fields must rehydrate from a per-trial JSON."""

    from app.training.checkpoint import _metrics_from_payload

    payload = {
        "metrics": {
            "loss": 0.5,
            "close_rmse": float("inf"),
            "volatility_rmse": float("inf"),
            "combined_rmse": float("inf"),
            "regression_rmse_log_rv": 0.42,
            "regression_mae_log_rv": 0.31,
            "regression_loss": 0.18,
            "regression_r2_log_rv": 0.27,
        }
    }
    metrics = _metrics_from_payload(payload)
    assert metrics is not None
    assert metrics.regression_rmse_log_rv == pytest.approx(0.42)
    assert metrics.regression_mae_log_rv == pytest.approx(0.31)
    assert metrics.regression_loss == pytest.approx(0.18)
    assert metrics.regression_r2_log_rv == pytest.approx(0.27)


def test_metrics_from_payload_legacy_pre_304_returns_none_regression_fields() -> None:
    """Pre-#304 per-trial JSONs (no regression keys) keep the contract."""

    from app.training.checkpoint import _metrics_from_payload

    payload = {
        "metrics": {
            "loss": 0.5,
            "close_rmse": 0.1,
            "volatility_rmse": 0.05,
            "combined_rmse": 0.15,
        }
    }
    metrics = _metrics_from_payload(payload)
    assert metrics is not None
    assert metrics.regression_rmse_log_rv is None
    assert metrics.regression_mae_log_rv is None
    assert metrics.regression_loss is None
    assert metrics.regression_r2_log_rv is None


def test_coerce_payload_config_round_trips_head_mode_and_alpha() -> None:
    """A dual-head checkpoint must rehydrate ``head_mode`` + ``regression_alpha``."""

    from app.training.checkpoint import _coerce_payload_config

    payload = {
        "model_config": {
            "output_mode": "classification",
            "head_mode": "dual",
            "regression_alpha": 0.7,
            "n_classes": 3,
        }
    }
    config = _coerce_payload_config(payload)
    assert config.head_mode == "dual"
    assert config.regression_alpha == pytest.approx(0.7)


def test_coerce_payload_config_legacy_pre_304_defaults_match_dataclass() -> None:
    """Pre-#304 checkpoint (no head_mode key) inherits the dataclass default."""

    from app.models.config import ModelConfig
    from app.training.checkpoint import _coerce_payload_config

    payload = {"model_config": {"output_mode": "regression"}}
    config = _coerce_payload_config(payload)
    # ModelConfig's dataclass default is 'dual' (ADR 0015 / #322); the
    # coercion path must match so a legacy checkpoint rebuilds the
    # same config the dataclass would emit cold.
    assert config.head_mode == ModelConfig().head_mode
    assert config.regression_alpha == pytest.approx(ModelConfig().regression_alpha)
