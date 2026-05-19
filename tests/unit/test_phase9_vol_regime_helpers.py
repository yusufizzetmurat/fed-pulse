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


# ---------------------------------------------------------------------------
# A1 (#206) per-fold class weighting
# ---------------------------------------------------------------------------


def test_class_weights_invert_frequency_on_balanced_train_slice() -> None:
    """Perfectly-balanced train slice -> weights all close to 1.0."""

    from app.training.loaders import fit_class_weights

    quantiles = (0.01, 0.02)
    # 30 events evenly across the 3 classes
    vols = [0.005] * 10 + [0.015] * 10 + [0.025] * 10
    weights = fit_class_weights(vols, quantiles, n_classes=3)
    assert len(weights) == 3
    # All near 1.0 (perfect balance); sum should equal n_classes
    assert sum(weights) == pytest.approx(3.0)
    for w in weights:
        assert 0.9 < w < 1.1


def test_class_weights_upweight_minority_class() -> None:
    """A rare class gets the largest weight; the dominant class the
    smallest. Inverse-frequency invariant."""

    from app.training.loaders import fit_class_weights

    quantiles = (0.01, 0.02)
    # 90% calm, 5% normal, 5% high
    vols = [0.005] * 90 + [0.015] * 5 + [0.025] * 5
    weights = fit_class_weights(vols, quantiles, n_classes=3)
    # calm has lowest weight; normal + high tied at the top
    assert weights[0] < weights[1]
    assert weights[0] < weights[2]
    assert weights[1] == pytest.approx(weights[2], abs=0.05)


def test_class_weights_handle_missing_class_via_smoothing() -> None:
    """A class with zero training rows must still get a finite weight
    -- the smoothing constant prevents 1/0 blow-up."""

    from app.training.loaders import fit_class_weights

    quantiles = (0.01, 0.02)
    # No "high" class events
    vols = [0.005] * 20 + [0.015] * 20
    weights = fit_class_weights(vols, quantiles, n_classes=3)
    assert len(weights) == 3
    assert all(w > 0 for w in weights)
    # The empty class should have the highest weight
    assert weights[2] >= weights[0]
    assert weights[2] >= weights[1]


def test_class_weights_empty_quantiles_returns_empty() -> None:
    from app.training.loaders import fit_class_weights

    weights = fit_class_weights([0.01, 0.02, 0.03], (), n_classes=3)
    assert weights == ()


def test_class_weights_empty_vols_returns_empty() -> None:
    from app.training.loaders import fit_class_weights

    weights = fit_class_weights([], (0.01, 0.02), n_classes=3)
    assert weights == ()


def test_class_weights_normalisation_sums_to_n_classes() -> None:
    """The normalisation contract: weights sum to ``n_classes`` so a
    class at the dataset's average frequency gets weight ~1.0."""

    from app.training.loaders import fit_class_weights

    quantiles = (0.01, 0.02)
    vols = [0.005] * 50 + [0.015] * 30 + [0.025] * 20
    weights = fit_class_weights(vols, quantiles, n_classes=3)
    assert sum(weights) == pytest.approx(3.0, abs=1e-6)


def test_class_weights_skip_nan_and_none() -> None:
    from app.training.loaders import fit_class_weights

    quantiles = (0.01, 0.02)
    vols = [0.005, 0.015, 0.025, None, float("nan")]
    weights = fit_class_weights(vols, quantiles, n_classes=3)
    # 3 valid events, one per class -> balanced
    assert sum(weights) == pytest.approx(3.0)
    for w in weights:
        assert 0.9 < w < 1.1
