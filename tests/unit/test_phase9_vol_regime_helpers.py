from __future__ import annotations

import pytest

from app.training.loaders import fit_vol_regime_quantiles, vol_regime_class_for


# ---------------------------------------------------------------------------
# Phase 9 V2 (#195) per-fold quantile fit
# ---------------------------------------------------------------------------


def test_fit_returns_two_cutoffs_for_3_class() -> None:
    vols = [0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012, 0.015, 0.020, 0.025, 0.030]
    cutoffs = fit_vol_regime_quantiles(vols, n_classes=3)
    assert len(cutoffs) == 2
    # cutoffs are ascending (no class can overlap a neighbour)
    assert cutoffs[0] < cutoffs[1]


@pytest.mark.parametrize("n_classes,expected_cutoff_count", [(2, 1), (3, 2), (4, 3), (5, 4)])
def test_fit_emits_n_minus_one_cutoffs(n_classes: int, expected_cutoff_count: int) -> None:
    vols = list(range(1, 101))
    cutoffs = fit_vol_regime_quantiles(vols, n_classes=n_classes)
    assert len(cutoffs) == expected_cutoff_count


def test_fit_empty_input_returns_empty_tuple() -> None:
    assert fit_vol_regime_quantiles([], n_classes=3) == ()


def test_fit_drops_nan_and_none_before_quantile() -> None:
    vols = [0.01, 0.02, None, float("nan"), 0.03, 0.04]
    cutoffs = fit_vol_regime_quantiles(vols, n_classes=3)
    # 4 valid values -> still emits 2 cutoffs
    assert len(cutoffs) == 2
    assert cutoffs[0] < cutoffs[1]


def test_fit_returns_empty_when_too_few_observations_for_split() -> None:
    """Need at least n_classes non-NaN observations to fit a defensible split."""
    assert fit_vol_regime_quantiles([0.01, 0.02], n_classes=3) == ()


def test_fit_rejects_n_classes_below_two() -> None:
    with pytest.raises(ValueError, match="n_classes"):
        fit_vol_regime_quantiles([0.01, 0.02, 0.03], n_classes=1)


# ---------------------------------------------------------------------------
# Phase 9 V2 (#195) class assignment from fitted quantiles
# ---------------------------------------------------------------------------


def test_class_assignment_uses_quantile_boundaries() -> None:
    q = (0.01, 0.02)  # calm < 0.01 ; normal in [0.01, 0.02) ; high >= 0.02
    assert vol_regime_class_for(0.005, q) == 0
    assert vol_regime_class_for(0.01, q) == 1   # right-open lower bound -> class 1
    assert vol_regime_class_for(0.015, q) == 1
    assert vol_regime_class_for(0.02, q) == 2   # right-open lower bound -> class 2
    assert vol_regime_class_for(0.10, q) == 2


def test_class_assignment_returns_minus_one_for_missing() -> None:
    q = (0.01, 0.02)
    assert vol_regime_class_for(None, q) == -1
    assert vol_regime_class_for(float("nan"), q) == -1


def test_class_assignment_handles_arbitrary_n_classes() -> None:
    q = (0.01, 0.02, 0.03, 0.04)   # 5-class
    assert vol_regime_class_for(0.005, q) == 0
    assert vol_regime_class_for(0.015, q) == 1
    assert vol_regime_class_for(0.025, q) == 2
    assert vol_regime_class_for(0.035, q) == 3
    assert vol_regime_class_for(0.10, q) == 4


def test_train_quantile_roundtrip_yields_balanced_classes() -> None:
    """Fit on the train slice + apply to the same slice should give
    approximately even per-class population."""
    import random

    rng = random.Random(11)
    vols = [rng.uniform(0.005, 0.05) for _ in range(900)]
    cutoffs = fit_vol_regime_quantiles(vols, n_classes=3)
    labels = [vol_regime_class_for(v, cutoffs) for v in vols]
    # Each class should have ~300 +- a margin for the random sample.
    counts = [labels.count(c) for c in range(3)]
    for c in counts:
        assert 270 <= c <= 330, f"class counts {counts} should be near 300 each"
