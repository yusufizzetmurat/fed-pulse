"""Tests for daily realized-measure computation (pure)."""

from __future__ import annotations
import numpy as np
import pytest
from app.data import intraday_realized as ir
from app.data.alphavantage_spx import IntradayBar


def _ohlcv(closes):
    c = np.asarray(closes, float)
    return list(c), list(c * 1.001), list(c * 0.999), [1000.0] * len(c)  # close, high, low, vol


def test_realized_measures_decomposition_and_formulas() -> None:
    rng = np.random.default_rng(0)
    closes = (100 * np.exp(np.cumsum(rng.normal(0, 0.001, 40)))).tolist()
    cl, hi, lo, vol = _ohlcv(closes)
    m = ir.daily_realized_measures(cl, hi, lo, vol)
    assert m is not None
    r = np.diff(np.log(np.asarray(closes)))
    assert m["rv"] == pytest.approx(float(np.sum(r**2)))
    assert m["rs_pos"] + m["rs_neg"] == pytest.approx(m["rv"])
    assert m["bv"] == pytest.approx((np.pi / 2) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))
    assert m["rvol"] == pytest.approx(1000.0 * 40)
    assert m["parkinson"] > 0 and np.isfinite(m["rskew"]) and m["rkurt"] > 0
    assert m["n_ret"] == len(r)


def test_too_few_bars_returns_none() -> None:
    assert (
        ir.daily_realized_measures([100.0, 101.0], [101.0, 102.0], [99.0, 100.0], [1.0, 1.0])
        is None
    )


def test_months_between_inclusive() -> None:
    assert ir._months_between("2020-11", "2021-02") == ["2020-11", "2020-12", "2021-01", "2021-02"]


def test_measures_by_day_groups_dates() -> None:
    rng = np.random.default_rng(1)
    bars = []
    for d in ("2020-01-02", "2020-01-03"):
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, 30)))
        for i, p in enumerate(px):
            bars.append(
                IntradayBar(
                    f"{d} {9 + i // 12:02d}:{(i * 5) % 60:02d}:00", p, p * 1.001, p * 0.999, p, 1.0
                )
            )
    out = ir._measures_by_day(bars)
    assert set(out) == {"2020-01-02", "2020-01-03"}
    assert out["2020-01-02"]["rv"] > 0 and "rvol" in out["2020-01-02"]


def test_measures_by_day_drops_post_cutoff_bars_on_fomc_days(monkeypatch) -> None:
    """Bars strictly after 14:00 ET on an FOMC meeting day must not enter the
    daily reduction. The 14:00 bar itself stays (the cutoff is strict), and a
    non-FOMC day keeps every bar regardless of the time-of-day stamp."""

    from datetime import date as _date

    from app.services import market_data as _md

    monkeypatch.setattr(_md, "_fomc_days", lambda: frozenset({_date(2020, 1, 2)}))

    rng = np.random.default_rng(2)
    bars = []
    # 9:00 through 16:00 in 5-minute increments → 85 bars per day; well above
    # MIN_RETURNS_PER_DAY so both days survive the per-day reduction.
    for hh in range(9, 16):
        for mm in range(0, 60, 5):
            for d in ("2020-01-02", "2020-01-03"):
                p = float(100 * np.exp(rng.normal(0, 0.001)))
                bars.append(
                    IntradayBar(f"{d} {hh:02d}:{mm:02d}:00", p, p * 1.001, p * 0.999, p, 1.0)
                )
    for d in ("2020-01-02", "2020-01-03"):
        p = float(100 * np.exp(rng.normal(0, 0.001)))
        bars.append(IntradayBar(f"{d} 16:00:00", p, p * 1.001, p * 0.999, p, 1.0))

    out = ir._measures_by_day(bars)
    # Both dates survive the reduction.
    assert set(out) == {"2020-01-02", "2020-01-03"}
    # Non-FOMC day keeps all 85 bars → 84 returns.
    assert out["2020-01-03"]["n_ret"] == 84
    # FOMC day keeps 9:00 through 14:00 inclusive (the 14:00 bar stays
    # because the cutoff is strict-greater-than). That's 5 hours × 12 bars
    # plus the 14:00 bar = 61 bars → 60 returns.
    assert out["2020-01-02"]["n_ret"] == 60
