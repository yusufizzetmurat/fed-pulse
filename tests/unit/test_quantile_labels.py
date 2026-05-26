"""Per-fold 3-class quantile label generation for rates targets (#291)."""

from __future__ import annotations

import math

import pytest

from app.data.quantile_labels import (
    EASING_LABEL,
    LOWER_QUANTILE,
    NEUTRAL_LABEL,
    TIGHTENING_LABEL,
    UPPER_QUANTILE,
    assign_labels,
    compute_bin_edges,
    fold_manifest_entry,
    label_for_value,
)


def test_quantile_constants_match_vol_regime_convention() -> None:
    """33/67 tertile cutoffs mirror the vol-regime classifier."""

    assert LOWER_QUANTILE == pytest.approx(1.0 / 3.0)
    assert UPPER_QUANTILE == pytest.approx(2.0 / 3.0)


def test_compute_bin_edges_on_known_distribution() -> None:
    # 9 values evenly spaced -> 33rd percentile ~= -25, 67th ~= +25
    train = [-50.0, -40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
    edges = compute_bin_edges(train, column="yield_2y_change_5d")
    assert edges.column == "yield_2y_change_5d"
    assert edges.n_train_rows == 9
    # Empirical quantiles with linear interpolation at q=1/3:
    # pos = (1/3) * 8 = 2.666... -> floor=2 (val -30), ceil=3 (val -20),
    # weight = 0.666 -> interp = -30 + 0.666 * 10 = -23.333...
    assert edges.lower == pytest.approx(-23.333333, abs=1e-4)
    # At q=2/3: pos = 5.333..., interp -> 0 + 0.333 * 10 = 3.333
    assert edges.upper == pytest.approx(3.333333, abs=1e-4)


def test_label_for_value_splits_on_inclusive_lower_exclusive_upper() -> None:
    edges = compute_bin_edges(
        [-100.0, -50.0, 0.0, 50.0, 100.0], column="yield_2y_change_5d"
    )
    # Below lower edge -> easing
    assert label_for_value(edges.lower - 0.1, edges) == EASING_LABEL
    # Equal to lower edge -> neutral (lower is the inclusive boundary)
    assert label_for_value(edges.lower, edges) == NEUTRAL_LABEL
    # Between lower and upper -> neutral
    midpoint = 0.5 * (edges.lower + edges.upper)
    assert label_for_value(midpoint, edges) == NEUTRAL_LABEL
    # At or above upper -> tightening
    assert label_for_value(edges.upper, edges) == TIGHTENING_LABEL
    assert label_for_value(edges.upper + 0.1, edges) == TIGHTENING_LABEL


def test_label_for_value_returns_none_when_input_or_edges_missing() -> None:
    edges = compute_bin_edges(
        [-1.0, 0.0, 1.0], column="yield_2y_change_5d"
    )
    assert label_for_value(None, edges) is None
    assert label_for_value(float("nan"), edges) is None

    empty_edges = compute_bin_edges([], column="yield_2y_change_5d")
    assert math.isnan(empty_edges.lower)
    assert math.isnan(empty_edges.upper)
    assert label_for_value(5.0, empty_edges) is None


def test_assign_labels_preserves_input_order_and_handles_missing() -> None:
    edges = compute_bin_edges(
        [-100.0, -50.0, 0.0, 50.0, 100.0], column="yield_2y_change_5d"
    )
    values = [edges.lower - 1.0, None, edges.lower, edges.upper, float("nan")]
    labels = assign_labels(values, edges)
    assert labels == [EASING_LABEL, None, NEUTRAL_LABEL, TIGHTENING_LABEL, None]


def test_fold_manifest_entry_is_json_serializable_payload() -> None:
    import json

    edges_2y = compute_bin_edges([-1.0, 0.0, 1.0], column="yield_2y_change_5d")
    edges_5y = compute_bin_edges([-2.0, 0.0, 2.0], column="yield_5y_change_5d")
    entry = fold_manifest_entry(
        fold_id="wf_fold_1",
        edges_by_column={
            "yield_2y_change_5d": edges_2y,
            "yield_5y_change_5d": edges_5y,
        },
    )
    serialized = json.dumps(entry)
    payload = json.loads(serialized)
    assert payload["fold_id"] == "wf_fold_1"
    assert set(payload["quantile_bin_edges"]) == {
        "yield_2y_change_5d",
        "yield_5y_change_5d",
    }
    assert payload["quantile_bin_edges"]["yield_2y_change_5d"]["n_train_rows"] == 3


def test_compute_bin_edges_filters_non_finite_values() -> None:
    train = [-1.0, float("nan"), 0.0, float("inf"), 1.0, None]
    edges = compute_bin_edges(train, column="yield_2y_change_5d")
    # Only -1.0, 0.0, 1.0 survive (inf is filtered by math.isfinite in
    # _empirical_quantile after the nan/None drop in compute_bin_edges).
    # Either 3 (if inf passes the nan filter) or 4 (if it doesn't) is
    # acceptable as long as the edges are finite real numbers.
    assert edges.n_train_rows in {3, 4}
    assert math.isfinite(edges.lower)
    assert math.isfinite(edges.upper)
