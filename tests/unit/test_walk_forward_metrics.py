"""Walk-forward regression metrics with block-bootstrap CIs (#291)."""

from __future__ import annotations

import math

import pytest

from app.evaluation.regression_metrics import (
    DEFAULT_BLOCK_SIZE,
    ZERO_TOLERANCE_BPS,
    directional_accuracy,
    mae_bps,
    r_squared,
    regression_metric_panel,
    with_block_bootstrap_ci,
)


def test_mae_bps_on_known_pairs() -> None:
    predicted = [10.0, -5.0, 0.0]
    observed = [12.0, -3.0, 1.0]
    # absolute errors: 2, 2, 1 -> mean 5/3 = 1.666...
    assert mae_bps(predicted, observed) == pytest.approx(5.0 / 3.0)


def test_mae_bps_skips_non_finite_pairs() -> None:
    predicted = [10.0, float("nan"), 5.0, None]
    observed = [12.0, 0.0, float("inf"), 1.0]
    # Only the first pair survives (NaN, inf, None all rejected).
    assert mae_bps(predicted, observed) == pytest.approx(2.0)


def test_directional_accuracy_uses_three_way_sign_with_no_move_band() -> None:
    # Within the no-move band on both sides -> match (both "flat").
    predicted = [ZERO_TOLERANCE_BPS - 0.1, 5.0, -5.0, 0.0]
    observed = [-(ZERO_TOLERANCE_BPS - 0.1), 10.0, 1.0, 0.0]
    # row 0: both inside no-move band -> sign 0 == 0, match
    # row 1: both positive -> match
    # row 2: pred negative, obs positive -> miss
    # row 3: both flat -> match
    assert directional_accuracy(predicted, observed) == pytest.approx(3.0 / 4.0)


def test_r_squared_perfect_predictions_returns_one() -> None:
    predicted = [1.0, 2.0, 3.0, 4.0]
    observed = [1.0, 2.0, 3.0, 4.0]
    assert r_squared(predicted, observed) == pytest.approx(1.0)


def test_r_squared_constant_observed_with_perfect_pred_returns_one() -> None:
    predicted = [5.0, 5.0, 5.0]
    observed = [5.0, 5.0, 5.0]
    assert r_squared(predicted, observed) == pytest.approx(1.0)


def test_r_squared_handles_constant_observed_with_bad_pred() -> None:
    predicted = [4.0, 5.0, 6.0]
    observed = [5.0, 5.0, 5.0]
    # Convention: 0 when observed is constant and residuals are non-zero.
    assert r_squared(predicted, observed) == pytest.approx(0.0)


def test_r_squared_returns_nan_when_fewer_than_two_pairs() -> None:
    assert math.isnan(r_squared([1.0], [1.0]))
    assert math.isnan(r_squared([], []))


def test_block_bootstrap_ci_brackets_point_estimate_on_known_inputs() -> None:
    # Construct a series where every absolute error is 5, so the
    # point estimate MAE = 5 and every resample (regardless of block
    # selection) yields the same value. The CI must collapse to [5, 5].
    predicted = [10.0, -10.0, 5.0, -5.0] * 5
    observed = [15.0, -5.0, 10.0, 0.0] * 5  # error always +5 in absolute value
    ci = with_block_bootstrap_ci(
        name="mae_bps",
        predicted=predicted,
        observed=observed,
        statistic="mae_bps",
        block_size=5,
        n_resamples=200,
        seed=11,
    )
    assert ci.point == pytest.approx(5.0)
    assert ci.lo == pytest.approx(5.0)
    assert ci.hi == pytest.approx(5.0)
    assert ci.n_observations == 20
    assert ci.block_size == 5


def test_block_bootstrap_ci_default_block_size_matches_horizon() -> None:
    """The default block size mirrors the 5-day forward-target horizon."""

    assert DEFAULT_BLOCK_SIZE == 5


def test_regression_metric_panel_returns_three_named_cis() -> None:
    predicted = list(range(20))
    observed = [v + 2 for v in predicted]
    panel = regression_metric_panel(
        predicted=predicted, observed=observed, n_resamples=100, seed=11
    )
    assert set(panel) == {"mae_bps", "directional_accuracy", "r_squared"}
    # MAE is exactly 2 across the whole panel.
    assert panel["mae_bps"].point == pytest.approx(2.0)
    # Directional accuracy: every prediction has the same sign as
    # the observation (both increasing with v).
    # predicted=0 (flat), observed=2 (positive) -> miss.
    # predicted=1 (positive), observed=3 (positive) -> match.
    # ...
    # Exactly 1 row mismatches (the zero-prediction row).
    assert panel["directional_accuracy"].point == pytest.approx(19.0 / 20.0)
    # R^2 reflects the squared-error penalty for the constant offset.
    # SS_res = 20 * 2^2 = 80; SS_tot = 20 * Var(observed) ~= 665, so
    # R^2 ~= 1 - 80/665 ~= 0.88. Well above zero and stable across
    # platforms — the assertion bounds it from below at 0.85.
    assert panel["r_squared"].point > 0.85


def test_with_block_bootstrap_ci_rejects_unknown_statistic() -> None:
    with pytest.raises(ValueError, match="unsupported statistic"):
        with_block_bootstrap_ci(
            name="bogus",
            predicted=[1.0, 2.0],
            observed=[1.5, 2.5],
            statistic="not_a_real_metric",
        )
