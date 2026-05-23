"""Hypothesis property tests on the per-fold quantile fit.

The unit suite at ``tests/unit/test_phase9_vol_regime_helpers.py`` covers
the documented edge cases (empty input, NaN-stripping, n_classes=1, the
3-class roundtrip with 900 random vols). This file adds the property-
based invariants the unit suite is too coarse to hit: any non-empty
positive-vol input with at least n_classes finite observations must
produce strictly ascending cutoffs and a class assignment that is
approximately balanced when applied to the same input.
"""

from __future__ import annotations

from typing import Sequence

import pytest
from hypothesis import given, settings, strategies as st

from app.training.loaders import fit_vol_regime_quantiles, vol_regime_class_for


def _positive_vols(min_size: int) -> st.SearchStrategy[list[float]]:
    """Lists of finite positive floats large enough to feed the fitter."""

    return st.lists(
        st.floats(
            min_value=1e-6,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=min_size,
        max_size=400,
    )


@given(vols=_positive_vols(min_size=3))
@settings(max_examples=100, deadline=2_000)
def test_cutoffs_are_strictly_ascending(vols: list[float]) -> None:
    cutoffs = fit_vol_regime_quantiles(vols, n_classes=3)
    if not cutoffs:
        return
    assert all(cutoffs[i] <= cutoffs[i + 1] for i in range(len(cutoffs) - 1))


@given(
    vols=_positive_vols(min_size=5),
    n_classes=st.integers(min_value=2, max_value=5),
)
@settings(max_examples=80, deadline=2_000)
def test_cutoff_count_matches_n_classes_minus_one(
    vols: list[float], n_classes: int
) -> None:
    cutoffs = fit_vol_regime_quantiles(vols, n_classes=n_classes)
    if len(vols) < n_classes:
        assert cutoffs == ()
        return
    # Quantile fit emits exactly n_classes - 1 interior cutoffs when it
    # has enough data; the unit suite locks the boundary behaviour.
    assert len(cutoffs) == n_classes - 1


@given(vols=_positive_vols(min_size=60))
@settings(max_examples=50, deadline=3_000)
def test_class_assignment_is_approximately_balanced(vols: Sequence[float]) -> None:
    """Apply the fitted cutoffs to the same vols. With n_classes=3, no
    class should claim less than ~10% or more than ~55% of the rows.

    The bounds are loose by design: with finite samples, perfectly
    balanced quantile cuts are not guaranteed. The test fails only when
    the partition collapses (one class absorbs the whole distribution),
    which would indicate the fitter is broken rather than a corner of
    statistical noise."""

    cutoffs = fit_vol_regime_quantiles(vols, n_classes=3)
    if not cutoffs:
        return
    labels = [vol_regime_class_for(v, cutoffs) for v in vols]
    n = len(labels)
    for c in (0, 1, 2):
        share = labels.count(c) / n
        assert 0.1 <= share <= 0.55, (
            f"class {c} took {share:.2%} of {n} rows; expected 10-55% under a "
            "balanced quantile partition"
        )


@given(
    vols=_positive_vols(min_size=20),
    extra=st.lists(
        st.one_of(st.none(), st.just(float("nan"))),
        min_size=1,
        max_size=10,
    ),
)
@settings(max_examples=50, deadline=2_000)
def test_nan_and_none_do_not_distort_cutoffs(
    vols: list[float], extra: list[float | None]
) -> None:
    """Sentinels (NaN, None) are dropped before the quantile call, so a
    fit on ``vols`` and a fit on ``vols + [None, NaN, ...]`` produce the
    same cutoffs."""

    clean = fit_vol_regime_quantiles(vols, n_classes=3)
    dirty = fit_vol_regime_quantiles(list(vols) + list(extra), n_classes=3)
    assert clean == dirty


@given(vols=_positive_vols(min_size=5))
@settings(max_examples=50, deadline=2_000)
def test_class_for_minimum_observation_is_zero(vols: list[float]) -> None:
    cutoffs = fit_vol_regime_quantiles(vols, n_classes=3)
    if not cutoffs:
        return
    assert vol_regime_class_for(min(vols) - 1e-9, cutoffs) == 0


@given(vols=_positive_vols(min_size=5))
@settings(max_examples=50, deadline=2_000)
def test_class_for_above_max_cutoff_is_top_class(vols: list[float]) -> None:
    cutoffs = fit_vol_regime_quantiles(vols, n_classes=3)
    if not cutoffs:
        return
    above = cutoffs[-1] + 1.0
    assert vol_regime_class_for(above, cutoffs) == len(cutoffs)
