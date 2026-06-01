"""Backtest HAR-tercile regime predictions against realized terciles.

Walks the published FOMC meeting calendar (most recent first), replays
``services.har_tercile.predict_har_regime`` against the rolling-window
RV history that the model would have seen at each historical event
date, and compares the predicted tercile to the realized tercile from
the forward 10-trading-day window.

This is the on-demand variant of the panel. The earlier service read a
``har_baselines`` block off ``analysis_runs.payload``, but the
``/analyze`` pipeline never wrote that block — so the predicted-tercile
column was actually the late-fusion regime argmax mapped onto the
tercile vocabulary, and the realized column was bucketed against
unrelated cutoffs. Resolving the prediction on demand against the same
RV history the live HAR-tercile endpoint sees keeps the predicted +
realized columns in the same cutoff space.

All bucketing is done in daily **realized-variance** space so the
comparison reproduces what ``services.har_tercile.predict_har_regime``
sees at prediction time. The realized forward window scalar is the
mean of squared log-returns over the post-event bars — a one-period
daily variance averaged across the 10-bar forward window, in the same
variance space as the per-bar RV series the cutoffs were quantiled on.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

from sqlalchemy.orm import Session  # kept for signature compatibility


# In-process TTL cache for the yfinance round trips that resolve
# realized RV and trailing RV history. The backtest panel renders
# the same ``(event_date, symbol)`` pairs across consecutive
# dashboard hits, so a short staleness window amortises the network
# hop without leaking yesterday's realisation into today's chart.
# Six hours is a safe upper bound: FOMC forward 10-bar windows
# resolve within a couple of trading days, well past the TTL.
_BACKTEST_CACHE_TTL_SECONDS = 6 * 60 * 60
_realized_rv_cache: dict[tuple[str, str], tuple[float, float | None]] = {}
_rv_history_cache: dict[tuple[str, str], tuple[float, list[float]]] = {}


def reset_caches() -> None:
    """Clear the TTL caches. Exposed for tests."""

    _realized_rv_cache.clear()
    _rv_history_cache.clear()


_TERCILE_LABELS: tuple[str, str, str] = ("low", "medium", "high")

# Trailing window the RV-history fetcher pulls per event. Sized to give
# ``predict_har_regime`` (which requires at least 22 daily RV values
# for the monthly HAR lag) a generous margin while keeping the cutoffs
# anchored to a recent volatility regime. Matches the upstream HAR
# trainer's 60-trading-day cutoff window.
_RV_HISTORY_WINDOW = 60
_MIN_RV_HISTORY = 22
_FORWARD_STEPS = 10
_FORWARD_WINDOW_DAYS = 30

# Picks the closest horizon to the 10-bar forward window so the panel
# compares apples-to-apples. ``predict_har_regime`` emits per-horizon
# rows at h=1/5/22; h=22 is the closest match to the forward window.
_PREDICTION_HORIZON = 22


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


def _bucket_against_cutoffs(value: float, q33: float, q67: float) -> str:
    if value < q33:
        return "low"
    if value < q67:
        return "medium"
    return "high"


def _realized_variance_from_log_returns(log_returns: list[float]) -> float | None:
    """Forward-window realized **variance** in the upstream RV convention.

    The HAR-tercile cutoffs are quantiles of a per-bar variance series
    (``r * r``). To stay in the same space, the forward-window realized
    stat is the mean of squared log-returns over the post-event bars —
    a one-period daily variance averaged across the 10-bar forward
    window. Returns None when the series is too short to be meaningful.
    """

    if len(log_returns) < 2:
        return None
    sq = [float(r) * float(r) for r in log_returns]
    if not sq:
        return None
    return sum(sq) / len(sq)


# Backwards-compat shim. Older tests import the legacy name; both
# refer to the same variance-space helper.
_realized_vol_from_log_returns = _realized_variance_from_log_returns


def _fetch_realized_rv_yf_uncached(event_date: str, symbol: str) -> float | None:
    """Pull forward-10d realized **variance** off yfinance.

    Wrapped behind a try / except so a yfinance flake on one row never
    nukes the whole backtest — the offending row just lands unresolved.
    The returned scalar is variance, matching the daily RV space the
    upstream HAR-tercile cutoffs live in.
    """

    try:
        from app.services.market_data import fetch_event_study_window
    except Exception:  # pragma: no cover — import-time defensive
        return None
    try:
        bars = fetch_event_study_window(
            event_date=event_date,
            symbol=symbol,
            steps=_FORWARD_STEPS,
            window_days=_FORWARD_WINDOW_DAYS,
        )
    except Exception:
        return None
    if not bars:
        return None
    log_returns = [float(bar.get("log_return", 0.0)) for bar in bars]
    return _realized_variance_from_log_returns(log_returns)


def _fetch_realized_rv_yf(event_date: str, symbol: str) -> float | None:
    """TTL-cached wrapper around :func:`_fetch_realized_rv_yf_uncached`.

    The backtest panel renders the same ``(event_date, symbol)`` pairs
    repeatedly as the dashboard mounts; caching the variance scalar for
    six hours absorbs those repeats without losing freshness once the
    forward window resolves on a later day.
    """

    key = (event_date, symbol)
    now = time.monotonic()
    cached = _realized_rv_cache.get(key)
    if cached is not None and now - cached[0] < _BACKTEST_CACHE_TTL_SECONDS:
        return cached[1]
    value = _fetch_realized_rv_yf_uncached(event_date, symbol)
    _realized_rv_cache[key] = (now, value)
    return value


def _fetch_rv_history_for_cutoffs_uncached(event_date: str, symbol: str) -> list[float]:
    """Fetch the trailing daily realized **variance** strictly before the event.

    Returns a list of per-bar variance values (squared log-returns)
    ending strictly before ``event_date`` — the same series upstream's
    ``main._load_rv_history`` writes, in the same units the cutoffs
    derived from ``predict_har_regime`` will compare against. Returns
    an empty list on any failure so the caller knows to leave the row
    unresolved rather than raise.
    """

    try:
        from app.services.market_data import _download_close_series_in_window
        from datetime import timedelta
        from datetime import datetime as _dt

        anchor = _dt.fromisoformat(event_date).date()
        start = anchor - timedelta(days=_RV_HISTORY_WINDOW * 2)
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


def _fetch_rv_history_for_cutoffs(event_date: str, symbol: str) -> list[float]:
    """TTL-cached wrapper around :func:`_fetch_rv_history_for_cutoffs_uncached`.

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
    value = _fetch_rv_history_for_cutoffs_uncached(event_date, symbol)
    _rv_history_cache[key] = (now, list(value))
    return value


def _cutoffs_from_history(history: Any) -> tuple[float | None, float | None]:
    """Tercile cutoffs on the supplied RV series.

    Uses ``np.quantile`` with the default linear-interpolation method on
    the [1/3, 2/3] quantiles — byte-for-byte the same call
    ``services.har_tercile._tercile_cutoffs`` makes at prediction time.
    This is what keeps the backtest's per-tercile hit-rate a faithful
    proxy of the live HAR-tercile endpoint's bucketing on the same
    realized series.
    """

    values = [float(v) for v in history if math.isfinite(float(v))]
    if len(values) < 3:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    q33, q67 = np.quantile(arr, [1.0 / 3.0, 2.0 / 3.0])
    return float(q33), float(q67)


def _predict_for_meeting(
    rv_history: list[float],
) -> tuple[str | None, float | None, float | None, float | None]:
    """Run ``predict_har_regime`` on ``rv_history`` and read off the panel row.

    Returns ``(predicted_tercile, predicted_prob, q33, q67)``. Picks the
    h=22 row from the per-horizon output (closest to the 10-day forward
    window the realized stat covers). Any failure inside the predict
    call collapses to a no-op tuple so the caller can skip the row.
    """

    if not rv_history or len(rv_history) < _MIN_RV_HISTORY:
        return None, None, None, None
    try:
        from app.services.har_tercile import predict_har_regime

        out = predict_har_regime(rv_history)
    except Exception:
        return None, None, None, None
    horizons = out.get("horizons") if isinstance(out, dict) else None
    if not isinstance(horizons, list):
        return None, None, None, None
    target = None
    for row in horizons:
        if isinstance(row, dict) and int(row.get("h", 0) or 0) == _PREDICTION_HORIZON:
            target = row
            break
    if target is None:
        return None, None, None, None
    label = target.get("tercile") if isinstance(target.get("tercile"), str) else None
    if label not in _TERCILE_LABELS:
        return None, None, None, None
    raw_probs = target.get("tercile_probs")
    probs: dict[str, Any] = raw_probs if isinstance(raw_probs, dict) else {}
    prob = _coerce_float(probs.get(label))
    q33 = _coerce_float(out.get("cutoffs_q33"))
    q67 = _coerce_float(out.get("cutoffs_q67"))
    return label, prob, q33, q67


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate accuracy + per-tercile hit-rate over the rows.

    Denominator for ``accuracy_overall`` is rows whose realized tercile
    resolved. Denominator for each ``per_tercile_hit_rate`` entry is
    rows whose PREDICTED tercile was that label AND whose realized
    tercile resolved.
    """

    total = len(rows)
    resolved_rows = [r for r in rows if r.get("realized_tercile") is not None]
    resolved = len(resolved_rows)
    overall: float | None = None
    if resolved > 0:
        hits = sum(1 for r in resolved_rows if r.get("correct"))
        overall = hits / resolved

    per_tercile: dict[str, float] = {}
    for label in _TERCILE_LABELS:
        subset = [r for r in resolved_rows if r.get("predicted_tercile") == label]
        if not subset:
            continue
        hits = sum(1 for r in subset if r.get("correct"))
        per_tercile[label] = hits / len(subset)

    return {
        "total_runs": total,
        "resolved_runs": resolved,
        "accuracy_overall": overall,
        "per_tercile_hit_rate": per_tercile,
    }


def build_har_tercile_backtest(
    session: Session,
    *,
    symbol: str = "^GSPC",
    limit: int = 10,
) -> dict[str, Any]:
    """Assemble the HAR-tercile backtest payload for the panel.

    Walks the published FOMC meeting calendar (most recent first) up to
    ``limit`` events, predicts the HAR-tercile regime against the
    rolling RV history strictly before each event, and resolves the
    realized tercile from the forward 10-bar window. Deduplicates by
    meeting date so the panel never carries two rows for the same
    event. Returns the dict-shape the endpoint wraps in
    ``HarTercileBacktestResponse``. ``session`` is accepted for API
    compatibility but no longer consulted — the source of truth is the
    FOMC calendar + on-demand RV history, not the persisted runs.
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

        rv_history = _fetch_rv_history_for_cutoffs(event_date, symbol)
        if len(rv_history) < _MIN_RV_HISTORY:
            # Not enough leading data to run HAR; skip the row entirely
            # so the panel denominator stays honest.
            continue

        predicted_label, predicted_prob, q33, q67 = _predict_for_meeting(rv_history)
        if predicted_label is None or q33 is None or q67 is None:
            continue

        realized_rv = _fetch_realized_rv_yf(event_date, symbol)
        if realized_rv is None:
            realized_label: str | None = None
            correct: bool | None = None
        else:
            realized_label = _bucket_against_cutoffs(realized_rv, q33, q67)
            correct = realized_label == predicted_label

        out_rows.append(
            {
                "event_date": event_date,
                "predicted_tercile": predicted_label,
                "predicted_prob": float(predicted_prob) if predicted_prob is not None else 0.0,
                "realized_tercile": realized_label,
                "realized_rv": realized_rv,
                "correct": correct,
            }
        )

    metrics = _aggregate_metrics(out_rows)
    return {
        "symbol": symbol,
        "horizon": _FORWARD_STEPS,
        "rows": out_rows,
        "metrics": metrics,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
