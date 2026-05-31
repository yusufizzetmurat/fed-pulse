"""Pure-function tests for the intraday RV forecast harness."""

from __future__ import annotations
import numpy as np
from app.data import intraday_rv_forecast as f


def test_har_lags_shapes_and_smoothing() -> None:
    log_rv = np.log(np.arange(1, 31, dtype=float))
    har = f._har_lags(log_rv)
    assert har.shape == (30, 3)
    assert har[0, 0] == har[0, 1] == har[0, 2]  # day-0 all equal (no history)
    assert har[-1, 0] > har[-1, 1] > har[-1, 2]  # rising series: daily > weekly > monthly


def test_forward_log_rv_target() -> None:
    rv = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    y = f._forward_log_rv(rv, 1)
    assert np.isclose(y[0], np.log(2.0)) and np.isnan(y[-1])
    y3 = f._forward_log_rv(rv, 3)
    assert np.isclose(y3[0], np.log((2 + 4 + 8) / 3))
    assert np.isnan(y3[-1]) and np.isnan(y3[-3])


def test_qlike_perfect_prediction_is_zero() -> None:
    true = np.log(np.array([1e-4, 4e-4, 9e-4, 1.6e-3]))
    assert np.isclose(f._qlike(true, true), 0.0, atol=1e-9)


def test_qlike_penalizes_underprediction_of_spike_more() -> None:
    # one spike day, off-by the same multiplicative factor in log space
    true = np.array([np.log(1e-4), np.log(1e-2)])  # second day is a 100x variance spike
    delta = np.log(2.0)  # factor-of-2 miss either direction
    under = true.copy()
    under[1] -= delta  # under-predict the spike's variance
    over = true.copy()
    over[1] += delta  # over-predict it by the same factor
    assert f._qlike(under, true) > f._qlike(over, true)
