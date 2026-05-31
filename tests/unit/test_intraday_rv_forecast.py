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
