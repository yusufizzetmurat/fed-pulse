from __future__ import annotations

import math

from app.evaluation.bootstrap import block_bootstrap_ci, bootstrap_paired_diff


def test_block_bootstrap_ci_centers_on_mean() -> None:
    values = [1.0, 1.0, 1.0, 1.0, 1.0]
    ci = block_bootstrap_ci(values, block_size=2, n_resamples=200, seed=11)
    assert ci.point == 1.0
    assert ci.lo == 1.0
    assert ci.hi == 1.0


def test_block_bootstrap_ci_brackets_mean_on_noisy_input() -> None:
    values = [1.0, 1.5, 0.5, 2.0, 0.0, 1.2, 0.8, 1.6, 0.4, 1.0]
    ci = block_bootstrap_ci(values, block_size=3, n_resamples=500, coverage=0.9, seed=11)
    assert ci.lo <= ci.point <= ci.hi
    assert ci.lo < ci.hi
    assert math.isclose(ci.point, sum(values) / len(values))


def test_bootstrap_paired_diff_sees_signal() -> None:
    a = [2.0, 2.1, 1.9, 2.2, 2.0, 2.1, 1.95]
    b = [1.0, 1.1, 0.9, 1.2, 1.0, 1.1, 0.95]
    ci = bootstrap_paired_diff(a, b, block_size=2, n_resamples=300, seed=11)
    assert ci.lo > 0.0


def test_empty_inputs_return_nan_ci() -> None:
    ci = block_bootstrap_ci([], block_size=2)
    assert math.isnan(ci.point) and math.isnan(ci.lo) and math.isnan(ci.hi)


def test_block_bootstrap_median_statistic() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    ci = block_bootstrap_ci(values, statistic="median", block_size=3, n_resamples=200, seed=11)
    assert 4.0 <= ci.point <= 6.0
