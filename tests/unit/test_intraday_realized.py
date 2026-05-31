"""Tests for daily realized-measure computation (pure)."""

from __future__ import annotations
import numpy as np
import pytest
from app.data import intraday_realized as ir
from app.data.alphavantage_spx import IntradayBar


def test_realized_measures_decomposition_and_formulas() -> None:
    rng = np.random.default_rng(0)
    closes = (100 * np.exp(np.cumsum(rng.normal(0, 0.001, 40)))).tolist()
    m = ir.daily_realized_measures(closes)
    assert m is not None
    r = np.diff(np.log(np.asarray(closes)))
    assert m["rv"] == pytest.approx(float(np.sum(r**2)))
    assert m["rs_pos"] + m["rs_neg"] == pytest.approx(m["rv"])  # RV = RS+ + RS-
    assert m["bv"] == pytest.approx((np.pi / 2) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))
    assert m["n_ret"] == len(r)


def test_too_few_bars_returns_none() -> None:
    assert ir.daily_realized_measures([100.0, 101.0, 102.0]) is None


def test_months_between_inclusive() -> None:
    assert ir._months_between("2020-11", "2021-02") == ["2020-11", "2020-12", "2021-01", "2021-02"]


def test_measures_by_day_groups_dates() -> None:
    rng = np.random.default_rng(1)
    bars = []
    for d in ("2020-01-02", "2020-01-03"):
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, 30)))
        for i, p in enumerate(px):
            bars.append(IntradayBar(f"{d} {9 + i//12:02d}:{(i*5)%60:02d}:00", p, p, p, p, 1.0))
    out = ir._measures_by_day(bars)
    assert set(out) == {"2020-01-02", "2020-01-03"}
    assert out["2020-01-02"]["rv"] > 0
