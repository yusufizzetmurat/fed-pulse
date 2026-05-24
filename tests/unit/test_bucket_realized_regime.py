from __future__ import annotations

from unittest.mock import patch

from app.services import forecaster


def test_returns_none_when_no_cutoffs_loaded():
    with patch.object(forecaster, "get_vol_regime_quantiles", return_value=()):
        assert forecaster.bucket_realized_regime(0.18) is None


def test_returns_none_when_input_is_missing_or_nan():
    with patch.object(forecaster, "get_vol_regime_quantiles", return_value=(0.10, 0.20)):
        assert forecaster.bucket_realized_regime(None) is None
        assert forecaster.bucket_realized_regime(float("nan")) is None


def test_buckets_match_cutoff_order():
    # Cutoffs taken at the 33% / 67% quantiles, matching the trained
    # classifier convention: v < q33 -> calm, q33 <= v < q67 -> normal,
    # v >= q67 -> high.
    cutoffs = (0.10, 0.20)
    with patch.object(forecaster, "get_vol_regime_quantiles", return_value=cutoffs):
        assert forecaster.bucket_realized_regime(0.05) == "calm"
        assert forecaster.bucket_realized_regime(0.15) == "normal"
        assert forecaster.bucket_realized_regime(0.25) == "high"
        # Exact-boundary case lands in the higher bucket because the
        # comparison is strict-less-than on the lower cutoff.
        assert forecaster.bucket_realized_regime(0.10) == "normal"


def test_returns_none_when_cutoff_count_disagrees_with_class_labels():
    # Defensive: a 4-class checkpoint with 3 cutoffs would land here.
    with patch.object(forecaster, "get_vol_regime_quantiles", return_value=(0.05, 0.15, 0.25)):
        assert forecaster.bucket_realized_regime(0.10) is None
