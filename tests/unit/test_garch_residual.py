"""Unit tests for the GARCH(1,1) baseline + residual helper (#236)."""

from __future__ import annotations

import datetime as _dt
import math
import random

import pytest

pytest.importorskip("arch")

from app.data.garch_residual import (
    GARCH_FORECAST_HORIZON,
    GarchResidualResult,
    MIN_FIT_RETURNS,
    compute_for_event,
    compute_garch_residual,
)


def _synth_lognormal_closes(n: int, *, seed: int, vol: float = 0.012) -> list[float]:
    """Synthesise ``n`` closes from an iid log-normal return process.

    iid log-returns are the simplest convergent regime for GARCH(1,1):
    the fitted alpha + beta should land near zero (no persistence), the
    omega should land near ``vol**2``, and the forecast should reduce
    to the unconditional vol regardless of horizon. We use this to anchor
    the helper's scale-handling without coupling the test to a specific
    QMLE optimum.
    """

    rng = random.Random(seed)
    closes = [100.0]
    for _ in range(n - 1):
        r = rng.gauss(0.0, vol)
        closes.append(closes[-1] * math.exp(r))
    return closes


def _trading_dates(n: int, *, start: _dt.date) -> list[_dt.date]:
    """Generate ``n`` consecutive weekday dates starting from ``start``."""

    out: list[_dt.date] = []
    cursor = start
    while len(out) < n:
        if cursor.weekday() < 5:
            out.append(cursor)
        cursor = cursor + _dt.timedelta(days=1)
    return out


def test_compute_garch_residual_returns_baseline_and_residual_on_iid_returns() -> None:
    """On iid log-normal closes the GARCH fit converges and the residual is finite.

    The exact baseline value depends on the QMLE optimum (which is
    sensitive to numerical paths); we anchor on the contract: a non-None
    baseline, a non-None residual matching raw - baseline, and a finite
    fit_returns_n equal to the input length minus the dropped initial bar.
    """

    closes = _synth_lognormal_closes(MIN_FIT_RETURNS + 50, seed=11)
    raw_target = 0.018  # arbitrary realised-vol target in log-return units

    result = compute_garch_residual(
        prior_closes=closes, forward_realized_vol_10d=raw_target
    )

    assert isinstance(result, GarchResidualResult)
    assert result.baseline is not None
    assert math.isfinite(result.baseline)
    assert result.baseline > 0.0
    assert result.residual is not None
    assert math.isfinite(result.residual)
    # residual = raw - baseline (within float-roundtrip tolerance)
    assert abs(result.residual - (raw_target - result.baseline)) < 1e-12
    # Initial bar drops out of the log-return series, so fit_returns_n
    # is len(closes) - 1.
    assert result.fit_returns_n == len(closes) - 1


def test_compute_garch_residual_below_min_fit_returns_returns_none() -> None:
    """Insufficient strict-prior returns -> baseline + residual both None.

    The gate at ``MIN_FIT_RETURNS`` exists so synthetic / truncated
    fixtures degrade gracefully instead of crashing on a singular
    Hessian. The helper still reports the would-be fit length so the
    caller can log the gate.
    """

    closes = _synth_lognormal_closes(MIN_FIT_RETURNS - 50, seed=7)
    result = compute_garch_residual(
        prior_closes=closes, forward_realized_vol_10d=0.020
    )
    assert result.baseline is None
    assert result.residual is None
    assert result.fit_returns_n == len(closes) - 1


def test_compute_garch_residual_missing_raw_target_returns_none_residual() -> None:
    """A None ``forward_realized_vol_10d`` propagates to the residual.

    The baseline is still computed (it only depends on strict-prior
    data, not on the forward target), but the residual is None because
    we cannot subtract from a missing quantity. This matters for events
    near the end of the price series where the 10-day forward window
    is truncated.
    """

    closes = _synth_lognormal_closes(MIN_FIT_RETURNS + 30, seed=21)
    result = compute_garch_residual(
        prior_closes=closes, forward_realized_vol_10d=None
    )
    assert result.baseline is not None
    assert result.residual is None


def test_compute_garch_residual_non_finite_raw_target_returns_none_residual() -> None:
    """NaN / inf raw target propagates to a None residual."""

    closes = _synth_lognormal_closes(MIN_FIT_RETURNS + 30, seed=22)
    nan_result = compute_garch_residual(
        prior_closes=closes, forward_realized_vol_10d=float("nan")
    )
    assert nan_result.baseline is not None
    assert nan_result.residual is None

    inf_result = compute_garch_residual(
        prior_closes=closes, forward_realized_vol_10d=float("inf")
    )
    assert inf_result.baseline is not None
    assert inf_result.residual is None


def test_compute_for_event_respects_strict_prior_slice() -> None:
    """The strict-prior slice drops every close dated at or after event_date.

    We seed a series of MIN_FIT_RETURNS + 20 closes ending past
    ``event_date``, run the helper, and assert the helper saw only the
    closes whose date is strictly less than the event. The test is
    structural (fit_returns_n must equal the strict-prior length minus
    the initial-bar drop), so we don't pin a specific baseline value.
    """

    total = MIN_FIT_RETURNS + 20
    dates = _trading_dates(total, start=_dt.date(2018, 1, 1))
    closes = _synth_lognormal_closes(total, seed=33)

    # Event sits 15 bars before the end of the series so 15 closes
    # post-date the event and must be excluded from the fit.
    event_date = dates[total - 15]
    expected_strict_prior_len = total - 15

    result = compute_for_event(
        dates=dates,
        closes=closes,
        event_date=event_date,
        forward_realized_vol_10d=0.014,
    )

    assert result.baseline is not None
    assert result.residual is not None
    # Log-return series has length (strict-prior closes - 1); the helper
    # reports that count.
    assert result.fit_returns_n == expected_strict_prior_len - 1


def test_compute_for_event_event_predates_series_returns_none() -> None:
    """An event dated before the first close -> no strict-prior window.

    The fit_returns_n is zero and both baseline / residual are None.
    """

    dates = _trading_dates(50, start=_dt.date(2020, 6, 1))
    closes = _synth_lognormal_closes(50, seed=44)
    early_event = _dt.date(2019, 1, 1)

    result = compute_for_event(
        dates=dates,
        closes=closes,
        event_date=early_event,
        forward_realized_vol_10d=0.012,
    )

    assert result.baseline is None
    assert result.residual is None
    assert result.fit_returns_n == 0


def test_compute_for_event_rejects_mismatched_input_lengths() -> None:
    """``dates`` and ``closes`` must have the same length."""

    with pytest.raises(ValueError, match="length mismatch"):
        compute_for_event(
            dates=[_dt.date(2020, 1, 1), _dt.date(2020, 1, 2)],
            closes=[100.0],
            event_date=_dt.date(2020, 6, 1),
            forward_realized_vol_10d=0.01,
        )


def test_compute_garch_residual_baseline_units_match_log_return_std() -> None:
    """The baseline lands in log-return units (same scale as the raw target).

    Sanity check that the percentage rescaling round-trip is correct:
    on an iid Gaussian process with std ~vol, the GARCH baseline should
    sit within an order of magnitude of vol. Anything off by a factor of
    100 would mean we forgot to de-scale.
    """

    vol = 0.010  # 1% daily std
    closes = _synth_lognormal_closes(MIN_FIT_RETURNS + 100, seed=55, vol=vol)
    result = compute_garch_residual(
        prior_closes=closes, forward_realized_vol_10d=vol
    )
    assert result.baseline is not None
    # Within 10x of the true vol (loose because the QMLE fit on 350
    # samples carries non-trivial estimation noise).
    assert vol / 10.0 < result.baseline < vol * 10.0


def test_garch_forecast_horizon_constant_matches_target_window() -> None:
    """The horizon constant matches the 10-day raw target window.

    Pinning the constant locks the scale-matching contract: the
    baseline must be forecast over the same window as
    ``_forward_realized_vol`` reports for the subtraction to land in
    consistent units.
    """

    assert GARCH_FORECAST_HORIZON == 10
