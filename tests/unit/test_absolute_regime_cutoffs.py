"""Unit tests for the #472 absolute vol-regime labelling path.

Covers the alternative-target row that swaps the per-fold quantile
cutoffs for fixed (calm_max, high_min) thresholds expressed in the
same per-period unit as ``forward_realized_vol_10d``.
"""

from __future__ import annotations

import math

import pytest


# ---------------------------------------------------------------------------
# vol_regime_absolute_class_for: math
# ---------------------------------------------------------------------------


def test_absolute_class_for_below_calm_max() -> None:
    """Values strictly below ``calm_max`` map to class 0 (calm)."""

    from app.training.loaders import vol_regime_absolute_class_for

    thresholds = (0.02, 0.04)
    assert vol_regime_absolute_class_for(0.0, thresholds) == 0
    assert vol_regime_absolute_class_for(0.005, thresholds) == 0
    assert vol_regime_absolute_class_for(0.019999, thresholds) == 0


def test_absolute_class_for_between_thresholds() -> None:
    """Values in ``[calm_max, high_min)`` map to class 1 (normal)."""

    from app.training.loaders import vol_regime_absolute_class_for

    thresholds = (0.02, 0.04)
    assert vol_regime_absolute_class_for(0.02, thresholds) == 1
    assert vol_regime_absolute_class_for(0.03, thresholds) == 1
    assert vol_regime_absolute_class_for(0.039999, thresholds) == 1


def test_absolute_class_for_at_or_above_high_min() -> None:
    """Values >= ``high_min`` map to class 2 (high)."""

    from app.training.loaders import vol_regime_absolute_class_for

    thresholds = (0.02, 0.04)
    assert vol_regime_absolute_class_for(0.04, thresholds) == 2
    assert vol_regime_absolute_class_for(0.1, thresholds) == 2
    assert vol_regime_absolute_class_for(1.0, thresholds) == 2


def test_absolute_class_for_missing_returns_minus_one() -> None:
    """Missing (None / NaN) targets return -1, matching the quantile contract."""

    from app.training.loaders import (
        vol_regime_absolute_class_for,
        vol_regime_class_for,
    )

    thresholds = (0.02, 0.04)
    assert vol_regime_absolute_class_for(None, thresholds) == -1
    assert vol_regime_absolute_class_for(float("nan"), thresholds) == -1
    # Same -1 contract as the per-fold quantile path so the
    # row-drop predicate is identical on both branches.
    assert vol_regime_class_for(None, thresholds) == -1
    assert vol_regime_class_for(float("nan"), thresholds) == -1


def test_absolute_class_for_matches_quantile_class_for_on_shared_inputs() -> None:
    """Absolute + quantile paths agree on every cell for the same ordered cutoffs.

    Both helpers implement the same less-than-cutoffs algorithm; the
    only difference is the alternative function fixes the tuple length
    at 2 and documents the economic semantics. The cell-by-cell match
    is what lets the loop dispatch route the absolute thresholds
    through the existing ``vol_regime_quantiles`` slot without any
    downstream codepath changes.
    """

    from app.training.loaders import (
        vol_regime_absolute_class_for,
        vol_regime_class_for,
    )

    thresholds = (0.025, 0.05)
    for value in [0.0, 0.01, 0.025, 0.03, 0.05, 0.07]:
        assert vol_regime_absolute_class_for(
            value, thresholds
        ) == vol_regime_class_for(value, thresholds)


# ---------------------------------------------------------------------------
# Annualized -> per-period conversion: ModelConfig defaults
# ---------------------------------------------------------------------------


def test_default_absolute_thresholds_match_12_and_22_pct_annualized() -> None:
    """ModelConfig defaults reflect the documented 12% / 22% annualized cutoffs.

    The conversion formula is
    ``vol_per_period = vol_annualized / sqrt(252 / 10)``; the test
    re-derives the cutoffs and asserts the dataclass default is within
    float tolerance of the recomputed pair. Guards against a future
    edit that touches the constants without recomputing the per-period
    values.
    """

    from app.models.config import (
        DEFAULT_ABSOLUTE_VOL_CALM_MAX_ANNUALIZED,
        DEFAULT_ABSOLUTE_VOL_HIGH_MIN_ANNUALIZED,
        DEFAULT_ABSOLUTE_VOL_THRESHOLDS,
    )

    expected_calm = DEFAULT_ABSOLUTE_VOL_CALM_MAX_ANNUALIZED / math.sqrt(25.2)
    expected_high = DEFAULT_ABSOLUTE_VOL_HIGH_MIN_ANNUALIZED / math.sqrt(25.2)
    assert DEFAULT_ABSOLUTE_VOL_CALM_MAX_ANNUALIZED == pytest.approx(0.12)
    assert DEFAULT_ABSOLUTE_VOL_HIGH_MIN_ANNUALIZED == pytest.approx(0.22)
    assert DEFAULT_ABSOLUTE_VOL_THRESHOLDS[0] == pytest.approx(expected_calm)
    assert DEFAULT_ABSOLUTE_VOL_THRESHOLDS[1] == pytest.approx(expected_high)
    # Sanity check: round-trip per-period -> annualized -> per-period.
    calm_back = DEFAULT_ABSOLUTE_VOL_THRESHOLDS[0] * math.sqrt(25.2)
    high_back = DEFAULT_ABSOLUTE_VOL_THRESHOLDS[1] * math.sqrt(25.2)
    assert calm_back == pytest.approx(0.12)
    assert high_back == pytest.approx(0.22)


def test_modelconfig_carries_default_label_mode_and_thresholds() -> None:
    """Default ModelConfig opts into the byte-identical quantile path."""

    from app.models.config import (
        DEFAULT_ABSOLUTE_VOL_THRESHOLDS,
        DEFAULT_VOL_REGIME_LABEL_MODE,
        ModelConfig,
    )

    cfg = ModelConfig()
    assert cfg.vol_regime_label_mode == DEFAULT_VOL_REGIME_LABEL_MODE
    assert cfg.vol_regime_label_mode == "per_fold_quantile"
    assert cfg.absolute_vol_thresholds == DEFAULT_ABSOLUTE_VOL_THRESHOLDS


# ---------------------------------------------------------------------------
# ModelConfig round-trip via from_model
# ---------------------------------------------------------------------------


class _StubModel:
    """Minimal stand-in for the model object ``ModelConfig.from_model`` reads.

    Carries only the attributes the round-trip needs to assert the new
    fields survive checkpoint serialise / deserialise. All other slots
    fall back to ``ModelConfig`` defaults via ``getattr(default)``.
    """

    def __init__(
        self,
        *,
        vol_regime_label_mode: str,
        absolute_vol_thresholds: tuple[float, float],
    ) -> None:
        self.model_type = "lstm"
        self.input_size = 6
        self.hidden_size = 64
        self.num_layers = 1
        self.dropout = 0.1
        self.head_hidden_size = 32
        self.initial_decay_rate = 0.0
        self.vol_regime_label_mode = vol_regime_label_mode
        self.absolute_vol_thresholds = absolute_vol_thresholds


def test_modelconfig_round_trips_absolute_label_mode() -> None:
    """``vol_regime_label_mode='absolute'`` survives ``from_model``."""

    from app.models.config import ModelConfig

    thresholds = (0.03, 0.05)
    stub = _StubModel(
        vol_regime_label_mode="absolute",
        absolute_vol_thresholds=thresholds,
    )
    cfg = ModelConfig.from_model(stub)
    assert cfg.vol_regime_label_mode == "absolute"
    assert cfg.absolute_vol_thresholds == thresholds


def test_modelconfig_round_trips_per_fold_label_mode() -> None:
    """Default ``per_fold_quantile`` survives ``from_model`` unchanged."""

    from app.models.config import DEFAULT_ABSOLUTE_VOL_THRESHOLDS, ModelConfig

    stub = _StubModel(
        vol_regime_label_mode="per_fold_quantile",
        absolute_vol_thresholds=DEFAULT_ABSOLUTE_VOL_THRESHOLDS,
    )
    cfg = ModelConfig.from_model(stub)
    assert cfg.vol_regime_label_mode == "per_fold_quantile"
    assert cfg.absolute_vol_thresholds == DEFAULT_ABSOLUTE_VOL_THRESHOLDS


def test_legacy_model_without_label_mode_falls_back_to_default() -> None:
    """Pre-#472 checkpoints (no field) restore the default labelling mode."""

    from app.models.config import (
        DEFAULT_ABSOLUTE_VOL_THRESHOLDS,
        DEFAULT_VOL_REGIME_LABEL_MODE,
        ModelConfig,
    )

    class _LegacyModel:
        model_type = "lstm"
        input_size = 6
        hidden_size = 64
        num_layers = 1
        dropout = 0.1
        head_hidden_size = 32
        initial_decay_rate = 0.0

    cfg = ModelConfig.from_model(_LegacyModel())
    assert cfg.vol_regime_label_mode == DEFAULT_VOL_REGIME_LABEL_MODE
    assert cfg.absolute_vol_thresholds == DEFAULT_ABSOLUTE_VOL_THRESHOLDS
