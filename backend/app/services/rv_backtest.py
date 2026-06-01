"""Backtest QLIKE-RV h=1 predictions against realized RV.

Walks the published FOMC meeting calendar (most recent first), runs
``services.rv_forecaster.predict_rv`` on the rolling daily-RV history
strictly before each event, and compares the h=1 point + 80% / 90%
conformal bands against the realized RV one trading day forward.

This is the on-demand variant of the panel. The earlier service read
the bands off ``analysis_runs.payload``, but the ``/analyze`` pipeline
never persisted a ``realized_vol_forecast.historical_bands`` block, so
the bands and the "in band" flags pointed at unrelated cutoffs.
Resolving the prediction on demand against the same RV history the
live ``/forecast/realized-vol`` endpoint sees keeps the predicted +
realized columns in the same calibration space.

Per-row resolution:
  * Pull the trailing daily realized variance strictly before the
    meeting date. If fewer than ``_MIN_RV_HISTORY`` bars come back,
    emit a pending row -- the HAR monthly lag needs ~22 days of warmup
    and the calibrated 60-day window the live card uses sits well
    above that floor.
  * Call ``predict_rv`` on the trailing window to derive the h=1
    point + 80% / 90% bands.
  * Fetch realized RV for the first trading day strictly after the
    meeting. The row is "in band" when the realized number falls
    between band_lo and band_hi.
  * Rows whose realized RV has not yet resolved (forward window not
    closed) surface with ``realized_rv = None`` and pending hit flags.
"""

from __future__ import annotations

import math
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np

from sqlalchemy.orm import Session  # kept for signature compatibility


# In-process TTL cache for the yfinance round trips that resolve
# realized RV and trailing RV history. The backtest panel renders the
# same ``(event_date, symbol)`` pairs across consecutive dashboard
# hits, so a short staleness window amortises the network hop without
# leaking yesterday's realisation into today's chart. Six hours is a
# safe upper bound: the h=1 realized bar resolves within a day, well
# past the TTL.
_BACKTEST_CACHE_TTL_SECONDS = 6 * 60 * 60
_realized_rv_cache: dict[tuple[str, str], tuple[float, float | None]] = {}
_rv_history_cache: dict[tuple[str, str], tuple[float, list[float]]] = {}


def reset_caches() -> None:
    """Clear the TTL caches. Exposed for tests."""

    _realized_rv_cache.clear()
    _rv_history_cache.clear()


_FORECAST_HORIZON = 1

# Trailing window the RV-history fetcher pulls per event. Matches the
# live ``/forecast/realized-vol`` 60-day default so the backtest's
# prediction surface is byte-for-byte the same calibration window the
# card consumes. ``_MIN_RV_HISTORY`` enforces enough leading data for
# HAR's monthly lag.
_RV_HISTORY_WINDOW = 60
_MIN_RV_HISTORY = 60


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _fetch_realized_rv_yf_uncached(event_date: str, symbol: str) -> float | None:
    """Pull the realized RV for the first trading bar after ``event_date``.

    Wrapped behind a try / except so a yfinance flake on one row never
    nukes the whole backtest -- the offending row just lands
    unresolved. The returned scalar is the squared log-return for the
    h=1 bar (single-period daily variance), matching the daily RV
    space ``predict_rv`` emits its bands in.
    """

    try:
        from app.services.market_data import fetch_event_study_window
    except Exception:  # pragma: no cover -- import-time defensive
        return None
    try:
        bars = fetch_event_study_window(
            event_date=event_date,
            symbol=symbol,
            steps=1,
            window_days=7,
        )
    except Exception:
        return None
    if not bars:
        return None
    log_return = _coerce_float(bars[0].get("log_return"))
    if log_return is None:
        return None
    rv = log_return * log_return
    if not math.isfinite(rv) or rv <= 0.0:
        return None
    return rv


def _fetch_realized_rv_yf(event_date: str, symbol: str) -> float | None:
    """TTL-cached wrapper around :func:`_fetch_realized_rv_yf_uncached`.

    The backtest panel renders the same ``(event_date, symbol)`` pairs
    repeatedly as the dashboard mounts; caching the variance scalar
    for six hours absorbs those repeats without losing freshness once
    the forward bar resolves on a later day.
    """

    key = (event_date, symbol)
    now = time.monotonic()
    cached = _realized_rv_cache.get(key)
    if cached is not None and now - cached[0] < _BACKTEST_CACHE_TTL_SECONDS:
        return cached[1]
    value = _fetch_realized_rv_yf_uncached(event_date, symbol)
    _realized_rv_cache[key] = (now, value)
    return value


def _fetch_rv_history_uncached(event_date: str, symbol: str) -> list[float]:
    """Fetch the trailing daily realized variance strictly before the event.

    Returns a list of per-bar variance values (squared log-returns)
    ending strictly before ``event_date``, in chronological order.
    The trailing window is sized to match the live ``/forecast/realized-vol``
    card's 60-day calibration window. Returns an empty list on any
    failure so the caller knows to surface the row as pending rather
    than raise.
    """

    try:
        from app.services.market_data import _download_close_series_in_window

        anchor = date.fromisoformat(event_date)
        start = anchor - timedelta(days=_RV_HISTORY_WINDOW * 3)
        end = anchor  # exclusive in the underlying yfinance call
        close_series = _download_close_series_in_window(symbol=symbol, start=start, end=end)
    except Exception:
        return []
    if close_series is None or len(close_series) < 3:
        return []
    try:
        closes = np.asarray(close_series, dtype=np.float64)
        closes = closes[np.isfinite(closes) & (closes > 0.0)]
        if closes.size < 2:
            return []
        log_returns = np.diff(np.log(closes))
        rv = log_returns * log_returns
        rv = rv[np.isfinite(rv) & (rv > 0.0)]
        if rv.size == 0:
            return []
        tail = rv[-_RV_HISTORY_WINDOW:]
        return [float(v) for v in tail]
    except Exception:
        return []


def _fetch_rv_history(event_date: str, symbol: str) -> list[float]:
    """TTL-cached wrapper around :func:`_fetch_rv_history_uncached`.

    Mirrors the realized-RV cache: the trailing variance series is
    keyed off ``(event_date, symbol)`` so a single dashboard render
    pays the yfinance hop once per event regardless of how many panel
    reloads land within the six-hour window.
    """

    key = (event_date, symbol)
    now = time.monotonic()
    cached = _rv_history_cache.get(key)
    if cached is not None and now - cached[0] < _BACKTEST_CACHE_TTL_SECONDS:
        return list(cached[1])
    value = _fetch_rv_history_uncached(event_date, symbol)
    _rv_history_cache[key] = (now, list(value))
    return value


def _predict_for_meeting(
    rv_history: list[float],
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Run ``predict_rv`` on ``rv_history`` and read off the h=1 row.

    Returns ``(point, lo80, hi80, lo90, hi90)`` in RV (variance) space.
    Any failure inside the predict call collapses to a no-op tuple so
    the caller can surface the row as pending.
    """

    if not rv_history or len(rv_history) < _MIN_RV_HISTORY:
        return None, None, None, None, None
    try:
        from app.services.rv_forecaster import predict_rv

        out = predict_rv(rv_history)
    except Exception:
        return None, None, None, None, None
    horizons = out.get("horizons") if isinstance(out, dict) else None
    if not isinstance(horizons, list):
        return None, None, None, None, None
    target = None
    for row in horizons:
        if isinstance(row, dict) and int(row.get("h", 0) or 0) == _FORECAST_HORIZON:
            target = row
            break
    if target is None:
        return None, None, None, None, None
    point = _coerce_float(target.get("point"))
    lo80 = _coerce_float(target.get("band_lo_80"))
    hi80 = _coerce_float(target.get("band_hi_80"))
    lo90 = _coerce_float(target.get("band_lo_90"))
    hi90 = _coerce_float(target.get("band_hi_90"))
    if None in (point, lo80, hi80, lo90, hi90):
        return None, None, None, None, None
    return point, lo80, hi80, lo90, hi90


def _aggregate_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute empirical band coverage across resolved rows.

    Denominator for each coverage figure is the count of rows whose
    realized RV resolved (``realized_rv`` not None). The nominal
    coverage levels are pinned at the calibration targets so the
    frontend can render the gap chip without re-deriving them.

    ``pending_runs`` carries the rows that could not be scored -- the
    trailing RV history was too short, the prediction call failed, or
    the realized bar has not yet resolved. Keeping it separate from
    ``resolved_runs`` lets the panel show "X resolved / Y pending"
    without polluting the coverage ratio with rows we never even
    attempted to score.
    """

    total = len(rows)
    resolved = [r for r in rows if r.get("realized_rv") is not None]
    n_res = len(resolved)
    n_pending = total - n_res
    empirical_80: float | None = None
    empirical_90: float | None = None
    if n_res > 0:
        empirical_80 = sum(1 for r in resolved if r.get("in_band_80")) / n_res
        empirical_90 = sum(1 for r in resolved if r.get("in_band_90")) / n_res
    return {
        "total_runs": total,
        "resolved_runs": n_res,
        "pending_runs": n_pending,
        "empirical_coverage_80": empirical_80,
        "empirical_coverage_90": empirical_90,
        "nominal_coverage_80": 0.80,
        "nominal_coverage_90": 0.90,
    }


def _pending_row(event_date: str) -> dict[str, Any]:
    """Build a pending-state row for ``event_date``."""

    return {
        "event_date": event_date,
        "point_forecast_rv": None,
        "band_lo_80": None,
        "band_hi_80": None,
        "band_lo_90": None,
        "band_hi_90": None,
        "realized_rv": None,
        "in_band_80": None,
        "in_band_90": None,
    }


def get_rv_backtest(
    session: Session,
    *,
    symbol: str = "^GSPC",
    limit: int = 10,
) -> dict[str, Any]:
    """Assemble the QLIKE-RV backtest payload for the panel.

    Walks the published FOMC meeting calendar (most recent first) up
    to ``limit`` events, predicts the h=1 RV + 80% / 90% bands against
    the rolling RV history strictly before each event, and resolves
    the realized RV from the first trading bar one day forward.
    Deduplicates by meeting date so the panel never carries two rows
    for the same event. Returns the dict-shape the endpoint wraps in
    ``RvBacktestResponse``. ``session`` is accepted for API
    compatibility but no longer consulted -- the source of truth is
    the FOMC calendar + on-demand RV history, not the persisted runs.
    """

    from app.services.fomc_calendar import list_past_meetings

    meetings = list_past_meetings(limit=limit)

    out_rows: list[dict[str, Any]] = []
    seen_dates: set[str] = set()
    for meeting in meetings:
        event_date = meeting.meeting_date.isoformat()
        if event_date in seen_dates:
            continue
        seen_dates.add(event_date)

        rv_history = _fetch_rv_history(event_date, symbol)
        if len(rv_history) < _MIN_RV_HISTORY:
            # Not enough leading data to run predict_rv honestly.
            # Surface the row as pending rather than dropping it; the
            # panel still keeps a placeholder so the counter stays
            # meaningful.
            out_rows.append(_pending_row(event_date))
            continue

        point, lo80, hi80, lo90, hi90 = _predict_for_meeting(rv_history)
        # _predict_for_meeting collapses any failure mode (predict raised,
        # horizons missing, partial band coverage) to an all-None tuple, so
        # narrowing on any single component is sufficient — but explicitly
        # narrowing all five keeps mypy --strict happy without runtime cost.
        if point is None or lo80 is None or hi80 is None or lo90 is None or hi90 is None:
            out_rows.append(_pending_row(event_date))
            continue

        realized_rv = _fetch_realized_rv_yf(event_date, symbol)
        if realized_rv is None:
            in_band_80: bool | None = None
            in_band_90: bool | None = None
        else:
            in_band_80 = bool(lo80 <= realized_rv <= hi80)
            in_band_90 = bool(lo90 <= realized_rv <= hi90)

        out_rows.append(
            {
                "event_date": event_date,
                "point_forecast_rv": point,
                "band_lo_80": lo80,
                "band_hi_80": hi80,
                "band_lo_90": lo90,
                "band_hi_90": hi90,
                "realized_rv": realized_rv,
                "in_band_80": in_band_80,
                "in_band_90": in_band_90,
            }
        )

    coverage = _aggregate_coverage(out_rows)
    return {
        "symbol": symbol,
        "horizon": _FORECAST_HORIZON,
        "rows": out_rows,
        "coverage": coverage,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
