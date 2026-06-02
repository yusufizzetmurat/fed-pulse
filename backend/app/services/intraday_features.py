"""Live intraday realized-measure fetcher for the RV forecaster.

The QLIKE-DLq forecaster was trained on an 11-feature row:
``[har_daily, har_weekly, har_monthly, rs_pos, rs_neg, bv, rq, rskew,
rkurt, parkinson, log(rvol+1)]``. At serve time the three HAR lags are
trivially derivable from any daily realized-variance history, but the
eight realized-measure columns require intraday bars to compute.

Without those features the head reduces to HAR plus a tiny learned
bias on the lags — i.e. the project's "QLIKE beats HAR by ~10%" edge
is not delivered in production. This module closes that gap: it pulls
5-minute bars from yfinance for the recent window, runs the existing
``intraday_realized.daily_realized_measures`` reducer, and returns the
most-recent complete day's measures so the forecaster fills the row
with live values instead of training-set means.

Failure modes are silent by design: a yfinance hiccup, a half-day, or
a symbol without intraday coverage all return ``None`` and the
forecaster falls back to its existing feat_mean substitution. The
caller logs the source so the dashboard can surface "QLIKE-full" vs
"HAR-fallback" honestly.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger("app.services.intraday_features")

# Cache key: (symbol, as_of). Value: (monotonic_timestamp, payload).
# Four-hour TTL — the realized stat changes once per trading day, so a
# multi-hour window absorbs dashboard repeats without going stale.
_CACHE_TTL_SECONDS = 4 * 60 * 60
_cache: dict[tuple[str, str], tuple[float, dict[str, Any] | None]] = {}

# How many calendar days of 5m bars to request. yfinance allows up to
# 60 days at the 5m interval; 30 is plenty for the trailing 22-day HAR
# window plus a few half-days / holidays.
_LOOKBACK_DAYS = 30

# Minimum 5m bars per day for the realized-measure reducer to be
# trustworthy. NYSE regular hours are 6.5h × 12 bars/h ≈ 78 bars; this
# threshold drops half-days and pre-/post-only sessions.
_MIN_BARS_PER_DAY = 60


def reset_cache() -> None:
    """Clear the in-process TTL cache (test hook)."""
    _cache.clear()


def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _trading_day_et() -> str:
    """Anchor cache key to America/New_York trading day, not UTC date.

    Bars publish on the NYSE schedule, so a UTC date boundary at midnight
    cuts across the 14:00 / 16:00 ET trading-day boundary and can stitch
    pre-close and post-close bars under the same cache key. Bucketing on
    ET date keeps each trading session in its own slot.
    """

    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def _fetch_intraday_bars_yf(symbol: str) -> Any | None:
    """Pull 5m bars off yfinance.

    Wrapped behind a try/except so a yfinance hiccup never breaks the
    request path — the caller falls back to training-mean features.
    """

    try:
        import yfinance as yf
    except Exception:
        logger.warning("intraday_features: yfinance import failed; degrading to None")
        return None

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            period=f"{_LOOKBACK_DAYS}d",
            interval="5m",
            auto_adjust=False,
            prepost=False,
        )
    except Exception as exc:
        logger.warning(
            "intraday_features: yfinance fetch failed for %r (%s); degrading", symbol, exc
        )
        return None

    if df is None or len(df) == 0:
        return None
    return df


def _reduce_to_daily(bars: Any) -> dict[str, Any] | None:
    """Group bars by ET trading date and return the most-recent day's measures."""

    from app.data.intraday_realized import daily_realized_measures

    # yfinance returns a DatetimeIndex tz-aware in market timezone for intraday.
    # Group by the date part of that index.
    grouped: dict[date, dict[str, list[float]]] = defaultdict(
        lambda: {"closes": [], "highs": [], "lows": [], "volumes": []}
    )
    for ts, row in bars.iterrows():
        d = ts.date() if hasattr(ts, "date") else ts
        try:
            close = float(row["Close"])
            high = float(row["High"])
            low = float(row["Low"])
            volume = float(row["Volume"])
        except Exception:
            continue
        if not (close > 0 and high > 0 and low > 0):
            continue
        grouped[d]["closes"].append(close)
        grouped[d]["highs"].append(high)
        grouped[d]["lows"].append(low)
        grouped[d]["volumes"].append(volume)

    # Pick the most-recent day with enough bars to trust the reducer.
    for d in sorted(grouped.keys(), reverse=True):
        day = grouped[d]
        if len(day["closes"]) < _MIN_BARS_PER_DAY:
            continue
        measures = daily_realized_measures(
            closes=day["closes"],
            highs=day["highs"],
            lows=day["lows"],
            volumes=day["volumes"],
        )
        if measures is None:
            continue
        # daily_realized_measures returns dict[str, float]; widen here so
        # the date string sits alongside the numeric measures without
        # tripping the strict-typed dict.
        out: dict[str, Any] = dict(measures)
        out["date"] = d.isoformat()
        return out
    return None


def recent_realized_measures(symbol: str, *, as_of: str | None = None) -> dict[str, Any] | None:
    """Live realized-measure row for ``symbol`` keyed on ``as_of``.

    Returns ``{"date": iso, "rv", "rs_pos", "rs_neg", "bv", "rq",
    "rskew", "rkurt", "parkinson", "rvol", "n_ret"}`` for the latest
    full day inside the lookback window, or ``None`` when the fetch
    fails, the reducer cannot trust the bars, or the symbol has no
    intraday coverage. Caller treats ``None`` as a signal to fall back
    to training-mean features and surface the "HAR-fallback" flag.

    Cached for :data:`_CACHE_TTL_SECONDS` against ``(symbol, as_of)`` so
    the dashboard does not hammer yfinance on every page refresh.
    """

    key = (symbol, as_of or _trading_day_et())
    now = time.monotonic()
    cached = _cache.get(key)
    if cached is not None and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]

    bars = _fetch_intraday_bars_yf(symbol)
    if bars is None:
        _cache[key] = (now, None)
        return None
    try:
        measures = _reduce_to_daily(bars)
    except Exception as exc:
        logger.warning("intraday_features: reducer failed for %r (%s); degrading", symbol, exc)
        measures = None
    if measures is None:
        # Reducer returned cleanly but found no full-session day in the
        # lookback window (holiday gap, half-day-only symbol, etc.).
        # The endpoint will surface "HAR-fallback" so ops know to
        # check yfinance coverage for this symbol.
        logger.warning(
            "intraday_features: no full session within %dd window for %r; "
            "downstream RV head will fall back to training-mean features",
            _LOOKBACK_DAYS,
            symbol,
        )
    _cache[key] = (now, measures)
    return measures
