"""Behavioural tests for the live intraday realized-measure fetcher.

The fetcher must:
- never raise out; failures degrade to ``None`` so the RV forecaster
  falls back to the HAR-grade training-mean path
- cache against (symbol, as_of) with a TTL so dashboard repeats do not
  hammer yfinance
- skip half-days / sessions with fewer than the minimum bars
- return a fully-populated measure dict for a typical full session
"""

from __future__ import annotations

import datetime
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.services import intraday_features


@pytest.fixture(autouse=True)
def _clear_cache():
    intraday_features.reset_cache()
    yield
    intraday_features.reset_cache()


def _make_bars(days: list[str], bars_per_day: int = 78) -> pd.DataFrame:
    """Build a yfinance-shaped intraday DataFrame.

    Each row carries Open/High/Low/Close/Volume; the index is a tz-aware
    DatetimeIndex per yfinance's convention.
    """

    rows: list[dict[str, Any]] = []
    timestamps: list[pd.Timestamp] = []
    rng = np.random.default_rng(seed=11)
    base_close = 5000.0
    for day in days:
        d = datetime.date.fromisoformat(day)
        # Session: 09:30 to 16:00 ET in 5m steps. The exact timezone is
        # irrelevant to the reducer; it groups by date.
        for i in range(bars_per_day):
            ts = pd.Timestamp(
                datetime.datetime.combine(d, datetime.time(13, 30, tzinfo=datetime.timezone.utc))
                + datetime.timedelta(minutes=5 * i),
            )
            r = float(rng.normal(0.0, 0.001))
            close = base_close * (1.0 + r)
            high = close * (1.0 + abs(rng.normal(0.0, 0.0005)))
            low = close * (1.0 - abs(rng.normal(0.0, 0.0005)))
            timestamps.append(ts)
            rows.append(
                {
                    "Open": close,
                    "High": high,
                    "Low": low,
                    "Close": close,
                    "Volume": float(1_000_000 + i),
                }
            )
            base_close = close
    return pd.DataFrame(rows, index=pd.DatetimeIndex(timestamps))


def test_recent_realized_measures_returns_dict_for_full_session(monkeypatch):
    bars = _make_bars(["2026-05-30", "2026-05-31", "2026-06-01"])
    monkeypatch.setattr(
        intraday_features, "_fetch_intraday_bars_yf", lambda _symbol: bars
    )
    out = intraday_features.recent_realized_measures("^GSPC", as_of="2026-06-02")
    assert out is not None
    assert out["date"] == "2026-06-01"  # most-recent day
    # All eight realized-measure keys plus rv and n_ret should be present.
    for key in (
        "rv",
        "rs_pos",
        "rs_neg",
        "bv",
        "rq",
        "rskew",
        "rkurt",
        "parkinson",
        "rvol",
        "n_ret",
    ):
        assert key in out
    assert out["rv"] > 0
    assert out["rvol"] > 0


def test_recent_realized_measures_skips_half_days(monkeypatch):
    # A 30-bar half-day must NOT be returned; the next-most-recent full
    # session should be picked instead.
    bars = pd.concat(
        [
            _make_bars(["2026-05-30"]),  # full
            _make_bars(["2026-06-01"], bars_per_day=30),  # half-day, below MIN
        ]
    )
    monkeypatch.setattr(
        intraday_features, "_fetch_intraday_bars_yf", lambda _symbol: bars
    )
    out = intraday_features.recent_realized_measures("^GSPC", as_of="2026-06-02")
    assert out is not None
    assert out["date"] == "2026-05-30"


def test_recent_realized_measures_none_on_empty_fetch(monkeypatch):
    monkeypatch.setattr(
        intraday_features, "_fetch_intraday_bars_yf", lambda _symbol: None
    )
    assert intraday_features.recent_realized_measures("^GSPC", as_of="2026-06-02") is None


def test_recent_realized_measures_none_on_reducer_exception(monkeypatch):
    bars = _make_bars(["2026-05-30"])
    monkeypatch.setattr(
        intraday_features, "_fetch_intraday_bars_yf", lambda _symbol: bars
    )

    def _explode(*_args: Any, **_kwargs: Any) -> dict[str, float]:
        raise RuntimeError("simulated reducer crash")

    # Patch the reducer the module imports inside _reduce_to_daily.
    monkeypatch.setattr(
        "app.data.intraday_realized.daily_realized_measures", _explode
    )
    assert intraday_features.recent_realized_measures("^GSPC", as_of="2026-06-02") is None


def test_recent_realized_measures_caches_per_as_of(monkeypatch):
    bars = _make_bars(["2026-05-30"])
    calls = MagicMock(return_value=bars)
    monkeypatch.setattr(intraday_features, "_fetch_intraday_bars_yf", calls)

    a = intraday_features.recent_realized_measures("^GSPC", as_of="2026-06-02")
    b = intraday_features.recent_realized_measures("^GSPC", as_of="2026-06-02")
    assert a == b
    # Cache must hit on the second call: yfinance fetched exactly once.
    assert calls.call_count == 1

    # Distinct as_of key triggers a fresh fetch.
    intraday_features.recent_realized_measures("^GSPC", as_of="2026-06-03")
    assert calls.call_count == 2


def test_recent_realized_measures_caches_none_payload(monkeypatch):
    """A negative result is cached so we don't re-hit yfinance every page load."""

    calls = MagicMock(return_value=None)
    monkeypatch.setattr(intraday_features, "_fetch_intraday_bars_yf", calls)

    intraday_features.recent_realized_measures("^GSPC", as_of="2026-06-02")
    intraday_features.recent_realized_measures("^GSPC", as_of="2026-06-02")
    assert calls.call_count == 1
