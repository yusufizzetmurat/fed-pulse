"""Unit tests for the #478 VIX term-structure + VRP feature block.

Covers:

- Hand-computed term-structure scalars on a fixture price series.
- Strict-prior boundary: the helper never reads ``series.close`` at or
  after ``event_date``.
- Graceful missing-data fallback: pre-coverage events, asset-history
  gaps, and per-scalar absence all degrade independently.
- Loader-side composer (`_compute_vix_features_for_event`) returns
  ``None`` when every column is missing and zero-fills partials.
"""

from __future__ import annotations

import datetime as _dt
import math

import pytest

from app.data import event_dataset_builder as edb
from app.training import loaders


def _series(dates: list[_dt.date], closes: list[float]) -> edb._CloseSeries:
    return edb._CloseSeries(
        dates=list(dates),
        close=[float(c) for c in closes],
        volume=[0.0] * len(closes),
    )


def _make_trading_dates(start: _dt.date, n: int) -> list[_dt.date]:
    out: list[_dt.date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += _dt.timedelta(days=1)
    return out


def test_vix_term_structure_scalars_are_t_minus_one_closes() -> None:
    """Hand-checked: each scalar equals the last close strictly before T."""

    dates = _make_trading_dates(_dt.date(2020, 1, 2), 100)
    as_of = dates[60]
    t_minus_one = dates[59]

    # Each series carries a known close at T-1 plus a wildly different
    # close on T itself; the helper must read T-1, never T.
    vix_closes = [15.0 + i * 0.01 for i in range(100)]
    vix_closes[60] = 9999.0
    vix1m_closes = [16.0 + i * 0.01 for i in range(100)]
    vix1m_closes[60] = 9999.0
    vix3m_closes = [18.0 + i * 0.01 for i in range(100)]
    vix3m_closes[60] = 9999.0
    vix6m_closes = [20.0 + i * 0.01 for i in range(100)]
    vix6m_closes[60] = 9999.0
    asset_closes = [100.0 + math.sin(i / 7.0) for i in range(100)]
    asset_closes[60] = 9999.0

    series_by_symbol = {
        "^VIX": _series(dates, vix_closes),
        "^VIX1M": _series(dates, vix1m_closes),
        "^VIX3M": _series(dates, vix3m_closes),
        "^VIX6M": _series(dates, vix6m_closes),
    }
    asset_series = _series(dates, asset_closes)

    out = edb._vix_term_structure_features(
        as_of=as_of,
        series_by_symbol=series_by_symbol,
        asset_series=asset_series,
    )

    assert out["vix_t_minus_1"] == pytest.approx(vix_closes[59])
    assert out["vix1m_t_minus_1"] == pytest.approx(vix1m_closes[59])
    assert out["vix3m_t_minus_1"] == pytest.approx(vix3m_closes[59])
    assert out["vix6m_t_minus_1"] == pytest.approx(vix6m_closes[59])
    assert out["vix_3m_over_1m_slope"] == pytest.approx(
        vix3m_closes[59] / vix1m_closes[59]
    )
    # VRP = implied (vix/100/sqrt(252)) - realised(30d log returns at T-1).
    base = asset_series.index_strictly_before(as_of)
    rets = edb._log_returns(asset_closes[base - 30 : base + 1])
    n = len(rets)
    mean = sum(rets) / n
    expected_realised = (sum((v - mean) ** 2 for v in rets) / (n - 1)) ** 0.5
    expected_implied = vix_closes[59] / 100.0 / (252.0**0.5)
    assert out["vrp_t_minus_1"] == pytest.approx(
        expected_implied - expected_realised
    )

    # The strict-prior boundary: confirm the helper would crash if it
    # read close[60] — we set that to 9999.0, so any contamination would
    # surface as an absurd scalar.
    for key, value in out.items():
        assert value is not None
        assert abs(value) < 100.0, (
            f"{key}={value} suggests the helper read a same-day or "
            "post-event close"
        )
    # Same date axis check: T_minus_one is what we expect.
    assert dates[59] == t_minus_one


def test_vix_term_structure_returns_none_when_series_missing() -> None:
    """Pre-coverage event: empty series-by-symbol -> every scalar None."""

    dates = _make_trading_dates(_dt.date(1985, 1, 2), 100)
    out = edb._vix_term_structure_features(
        as_of=dates[50],
        series_by_symbol={},
        asset_series=_series(dates, [100.0] * 100),
    )
    for value in out.values():
        assert value is None


def test_vix_term_structure_partial_coverage_degrades_per_scalar() -> None:
    """Only ^VIX present (pre-2008 era): vix scalar lands, term/slope None."""

    dates = _make_trading_dates(_dt.date(1995, 1, 2), 100)
    series_by_symbol = {
        "^VIX": _series(dates, [20.0 + i * 0.01 for i in range(100)]),
    }
    asset_series = _series(dates, [100.0 + 0.01 * i for i in range(100)])
    out = edb._vix_term_structure_features(
        as_of=dates[60],
        series_by_symbol=series_by_symbol,
        asset_series=asset_series,
    )
    assert out["vix_t_minus_1"] is not None
    assert out["vix1m_t_minus_1"] is None
    assert out["vix3m_t_minus_1"] is None
    assert out["vix6m_t_minus_1"] is None
    # Slope needs both vix1m + vix3m; both missing -> slope None.
    assert out["vix_3m_over_1m_slope"] is None
    # VRP still computable: vix alone covers the implied leg.
    assert out["vrp_t_minus_1"] is not None


def test_rolling_realized_vol_t_minus_1_is_strict_prior() -> None:
    """The realised baseline reads only closes strictly before as_of."""

    dates = _make_trading_dates(_dt.date(2020, 1, 2), 100)
    closes = [100.0] * 60 + [9999.0] + [100.0] * 39
    series = _series(dates, closes)
    # The 9999.0 jump on day 60 must not enter the trailing 30d window
    # at as_of=dates[60] (strict-prior contract).
    out = edb._rolling_realized_vol_t_minus_1(series, dates[60], window=30)
    assert out is not None
    assert out < 1e-9, (
        f"realised baseline = {out} contaminated by close[60]=9999.0"
    )
    # If the event sits at day 61 the jump now enters the trailing
    # window (close[60] is strictly before dates[61]) and the realised
    # leg must reflect it.
    out_after = edb._rolling_realized_vol_t_minus_1(series, dates[61], window=30)
    assert out_after is not None
    assert out_after > 1e-3, (
        f"realised baseline = {out_after} should pick up the jump at T-1"
    )


def test_rolling_realized_vol_returns_none_when_history_too_short() -> None:
    dates = _make_trading_dates(_dt.date(2020, 1, 2), 10)
    series = _series(dates, [100.0 + i for i in range(10)])
    out = edb._rolling_realized_vol_t_minus_1(series, dates[5], window=30)
    assert out is None


def test_compute_vix_features_for_event_returns_none_when_all_columns_missing() -> None:
    """Loader-side composer: events.parquet without #478 columns -> None."""

    row = {"event_date": "2020-01-15", "vix_t_minus_1": None}
    out = loaders._compute_vix_features_for_event(row)
    assert out is None


def test_compute_vix_features_for_event_zero_fills_partial_rows() -> None:
    """Partial population: missing scalars zero-fill, populated stay."""

    row = {
        "vix_t_minus_1": 18.5,
        "vix1m_t_minus_1": None,
        "vix3m_t_minus_1": 20.0,
        "vix6m_t_minus_1": None,
        "vix_3m_over_1m_slope": None,
        "vrp_t_minus_1": -0.001,
    }
    out = loaders._compute_vix_features_for_event(row)
    assert out is not None
    assert out == [18.5, 0.0, 20.0, 0.0, 0.0, -0.001]


def test_compute_vix_features_for_event_propagates_all_six_scalars() -> None:
    """All scalars populated: passthrough in the documented order."""

    row = {
        "vix_t_minus_1": 18.5,
        "vix1m_t_minus_1": 19.2,
        "vix3m_t_minus_1": 20.0,
        "vix6m_t_minus_1": 21.5,
        "vix_3m_over_1m_slope": 1.041,
        "vrp_t_minus_1": -0.0008,
    }
    out = loaders._compute_vix_features_for_event(row)
    assert out == [18.5, 19.2, 20.0, 21.5, 1.041, -0.0008]
