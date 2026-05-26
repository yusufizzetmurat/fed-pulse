"""Unit tests for ``app.services.regime_bucketing`` (#322).

The module exposes two pure functions used by ``/analyze`` to recover
3-class regime labels from the regression head's log-realised-vol point
estimate and residual std:

- ``bucket_log_rv(log_rv_point, raw_vol_cutoffs)`` -> hard label.
- ``derive_distribution(log_rv_point, log_rv_std, raw_vol_cutoffs)`` ->
  Gaussian-CDF mass into each of the three bins.

The raw-volatility cutoffs are the per-fold tertile edges persisted in
the fold manifest (in raw RV units, not log units); the bucketing path
compares ``log_rv_point`` against ``log(cutoff)`` so the caller can pass
the manifest values directly.

Cutoff pairs used below ((0.5, 1.2), (0.3, 0.9)) are plausible 33/67
tertile splits for forward 10-day realised vol on the canonical training
package; the regression target is O(1) so the log-space boundaries land
in (-1.2, 0.2).
"""

from __future__ import annotations

import math

import pytest

from app.services.regime_bucketing import (
    REGIME_LABELS,
    bucket_log_rv,
    derive_distribution,
)


CUTOFF_PAIRS: tuple[tuple[float, float], ...] = (
    (0.5, 1.2),
    (0.3, 0.9),
)


def test_regime_labels_are_canonical_three_class_tuple() -> None:
    assert REGIME_LABELS == ("calm", "normal", "high")


@pytest.mark.parametrize("cutoffs", CUTOFF_PAIRS)
def test_bucket_log_rv_assigns_each_region_to_correct_label(
    cutoffs: tuple[float, float],
) -> None:
    """Point just below the lower edge -> calm; just above -> normal;
    just above the upper edge -> high. ``eps`` is in log space (the
    comparison axis) so the test is robust to the exact convention
    (strict-less-than vs. less-than-or-equal) at the boundaries."""

    low, high = cutoffs
    log_low = math.log(low)
    log_high = math.log(high)
    eps = 1e-3

    assert bucket_log_rv(log_low - eps, cutoffs) == "calm"
    assert bucket_log_rv(log_low + eps, cutoffs) == "normal"
    assert bucket_log_rv(log_high + eps, cutoffs) == "high"


def test_bucket_log_rv_returns_none_when_cutoffs_empty() -> None:
    assert bucket_log_rv(0.0, ()) is None


@pytest.mark.parametrize(
    "bad_cutoffs",
    [
        (0.5,),
        (0.3, 0.6, 0.9),
        (0.1, 0.2, 0.3, 0.4),
    ],
)
def test_bucket_log_rv_returns_none_when_cutoffs_wrong_length(
    bad_cutoffs: tuple[float, ...],
) -> None:
    """The vol-regime head is 3-class; anything other than two cutoffs
    is a contract violation, so the bucketing function refuses to guess."""

    assert bucket_log_rv(0.0, bad_cutoffs) is None


@pytest.mark.parametrize("cutoffs", CUTOFF_PAIRS)
@pytest.mark.parametrize("log_rv_point", [-2.0, -0.7, 0.0, 0.18, 0.7, 1.5])
@pytest.mark.parametrize("log_rv_std", [0.05, 0.25, 0.75])
def test_derive_distribution_sums_to_one(
    cutoffs: tuple[float, float],
    log_rv_point: float,
    log_rv_std: float,
) -> None:
    dist = derive_distribution(log_rv_point, log_rv_std, cutoffs)
    assert dist is not None
    assert set(dist.keys()) == set(REGIME_LABELS)
    total = sum(dist.values())
    assert abs(total - 1.0) < 1e-9


@pytest.mark.parametrize("cutoffs", CUTOFF_PAIRS)
def test_derive_distribution_concentrates_in_correct_bin_when_std_is_small(
    cutoffs: tuple[float, float],
) -> None:
    """When the residual std collapses, mass should pile up almost
    entirely in the single bin that contains the point estimate."""

    low, high = cutoffs
    log_low = math.log(low)
    log_high = math.log(high)
    tiny_std = 1e-4

    # calm region: point well below the lower edge.
    calm_point = log_low - 1.0
    dist_calm = derive_distribution(calm_point, tiny_std, cutoffs)
    assert dist_calm is not None
    assert dist_calm["calm"] > 0.99

    # normal region: point centred between the two edges.
    normal_point = 0.5 * (log_low + log_high)
    dist_normal = derive_distribution(normal_point, tiny_std, cutoffs)
    assert dist_normal is not None
    assert dist_normal["normal"] > 0.99

    # high region: point well above the upper edge.
    high_point = log_high + 1.0
    dist_high = derive_distribution(high_point, tiny_std, cutoffs)
    assert dist_high is not None
    assert dist_high["high"] > 0.99


@pytest.mark.parametrize("cutoffs", CUTOFF_PAIRS)
def test_derive_distribution_returns_none_when_std_non_positive(
    cutoffs: tuple[float, float],
) -> None:
    assert derive_distribution(0.0, 0.0, cutoffs) is None
    assert derive_distribution(0.0, -0.1, cutoffs) is None


def test_derive_distribution_returns_none_when_cutoffs_missing() -> None:
    assert derive_distribution(0.0, 0.25, ()) is None
    assert derive_distribution(0.0, 0.25, (0.5,)) is None
    assert derive_distribution(0.0, 0.25, (0.1, 0.2, 0.3)) is None
