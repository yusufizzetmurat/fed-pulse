"""Backtest the last N HAR-tercile predictions against realized terciles.

Walks the persisted ``analysis_runs`` table for ``^GSPC``, extracts the
predicted tercile from each row's analyze payload, and resolves the
realized tercile off the forward 10-trading-day market history. Powers
the HarAccuracyPanel card: an aggregate hit-rate plus per-tercile
break-down + a compact row table.

All bucketing is done in daily **realized-variance** space so the
comparison reproduces what ``services.har_tercile.predict_har_regime``
sees at prediction time. That upstream operates on daily RV =
``log_return ** 2`` (the convention ``main._load_rv_history`` writes for
both the parquet path and the yfinance fallback), and the persisted
``har_baselines.cutoffs_q33`` / ``cutoffs_q67`` are quantiles of that
series via ``np.quantile`` with linear interpolation. The backtest
mirrors both choices:

* Realized RV over the forward 10-bar window is the mean of squared
  log-returns (so the scalar lives in the same variance space as the
  per-bar series that fed the prediction).
* Fallback cutoffs are computed off a 60-day series of daily
  ``log_return ** 2`` values, then quantiled with ``np.quantile``
  ``[1/3, 2/3]`` — byte-for-byte the same op the upstream uses on its
  own training fold.

Realized RV is read from the persisted payload's
``forward_realized_vol_10d`` slot when present (forward-compat with the
analyze response carrying that variance summary on future builds) and
falls back to a fresh yfinance pull via ``fetch_event_study_window``
otherwise. Cutoffs default to the ones the prediction recorded
(``cutoffs_q33`` / ``cutoffs_q67`` on the persisted ``har_baselines``
block when set) and otherwise recompute on a 60-day RV window so a row
that pre-dates the cutoff persistence still bucks honest.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import AnalysisRun


# Mirror the HAR-tercile label space (low / medium / high). The
# /analyze response stores the late-fusion classifier under
# ``regime_classification`` using the (calm / normal / high) label
# space; this table maps it onto the HAR-tercile vocabulary so the
# backtest is comparable.
_TERCILE_LABELS: tuple[str, str, str] = ("low", "medium", "high")
_REGIME_TO_TERCILE = {
    "calm": "low",
    "low": "low",
    "normal": "medium",
    "medium": "medium",
    "high": "high",
}

# Trailing window the realized-tercile fallback uses to recompute
# cutoffs when the persisted prediction did not pin them. Matches the
# 60-trading-day window the HAR-tercile baseline trains its cutoffs on
# in the research-side trainer.
_FALLBACK_CUTOFF_WINDOW = 60
_FORWARD_STEPS = 10
_FORWARD_WINDOW_DAYS = 30


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


def _normalize_tercile_label(label: Any) -> str | None:
    if not isinstance(label, str):
        return None
    key = label.strip().lower()
    if not key:
        return None
    return _REGIME_TO_TERCILE.get(key) or (key if key in _TERCILE_LABELS else None)


def _tercile_from_har_block(har_block: Any) -> tuple[str | None, float | None]:
    """Read predicted tercile + prob from a persisted ``har_baselines`` block.

    Prefers the 22-day horizon (closest to the 10-day forward resolution)
    and falls back to whichever row carries a valid label.
    """

    if not isinstance(har_block, dict):
        return None, None
    horizons = har_block.get("horizons")
    if not isinstance(horizons, list):
        return None, None
    ordered = sorted(
        (h for h in horizons if isinstance(h, dict)),
        key=lambda h: abs(int(h.get("h", 0) or 0) - 22),
    )
    for row in ordered:
        label = _normalize_tercile_label(row.get("tercile"))
        if label is None:
            continue
        probs = row.get("tercile_probs") or {}
        prob = _coerce_float(probs.get(label)) if isinstance(probs, dict) else None
        return label, prob
    return None, None


def _tercile_from_regime_block(regime: Any) -> tuple[str | None, float | None]:
    """Read predicted tercile + prob from a ``regime_classification`` block.

    Maps the regime vocabulary (calm / normal / high) to terciles via
    :func:`_normalize_tercile_label` and looks the probability up under
    the original regime key first, falling back to the normalized label.
    """

    if not isinstance(regime, dict):
        return None, None
    argmax = _normalize_tercile_label(regime.get("argmax_class"))
    if argmax is None:
        return None, None
    distribution = regime.get("distribution") or {}
    if not isinstance(distribution, dict):
        return argmax, None
    raw_argmax = regime.get("argmax_class")
    prob: float | None = None
    if isinstance(raw_argmax, str):
        prob = _coerce_float(distribution.get(raw_argmax))
    if prob is None:
        prob = _coerce_float(distribution.get(argmax))
    return argmax, prob


def _extract_predicted_tercile(payload: Any) -> tuple[str | None, float | None]:
    """Pull the predicted tercile + its probability off a persisted payload.

    Order of precedence:
      1. ``har_baselines`` block (forward-compat: future builds may
         persist the HAR-tercile card directly into the analyze payload).
      2. ``regime_classification`` (active late-fusion regime card).
    Returns ``(label, prob)`` with prob in [0, 1] when available; either
    half may be None on a degraded payload.
    """

    if not isinstance(payload, dict):
        return None, None
    label, prob = _tercile_from_har_block(payload.get("har_baselines"))
    if label is not None:
        return label, prob
    return _tercile_from_regime_block(payload.get("regime_classification"))


def _extract_persisted_cutoffs(payload: Any) -> tuple[float | None, float | None]:
    """Read q33 / q67 off the persisted ``har_baselines`` block when present."""

    if not isinstance(payload, dict):
        return None, None
    har_block = payload.get("har_baselines")
    if not isinstance(har_block, dict):
        return None, None
    q33 = _coerce_float(har_block.get("cutoffs_q33"))
    q67 = _coerce_float(har_block.get("cutoffs_q67"))
    return q33, q67


def _extract_persisted_realized_rv(payload: Any) -> float | None:
    """Pull the realized 10-day forward RV off the persisted payload.

    Forward-compat: future analyze responses may stash the realized
    forward-vol summary under ``forward_realized_vol_10d`` so the
    backtest no longer needs the yfinance round trip. The current build
    does not write that field, so this returns None and the caller
    falls back to a live market fetch.
    """

    if not isinstance(payload, dict):
        return None
    direct = _coerce_float(payload.get("forward_realized_vol_10d"))
    if direct is not None:
        return direct
    market = payload.get("market")
    if isinstance(market, dict):
        nested = _coerce_float(market.get("forward_realized_vol_10d"))
        if nested is not None:
            return nested
    return None


def _bucket_against_cutoffs(value: float, q33: float, q67: float) -> str:
    if value < q33:
        return "low"
    if value < q67:
        return "medium"
    return "high"


def _realized_variance_from_log_returns(log_returns: list[float]) -> float | None:
    """Forward-window realized **variance** in the upstream RV convention.

    Upstream (``app.main._load_rv_history``) builds the rv_history that
    feeds ``predict_har_regime`` as the per-bar squared log-return
    series (``r * r``). The HAR-tercile cutoffs are quantiles of that
    variance series. To stay in the same space, the forward-window
    realized stat is the mean of squared log-returns over the post-event
    bars — a one-period daily variance averaged across the 10-bar
    forward window. Returns None when the series is too short to be
    meaningful.
    """

    if len(log_returns) < 2:
        return None
    sq = [float(r) * float(r) for r in log_returns]
    if not sq:
        return None
    return sum(sq) / len(sq)


# Backwards-compat shim. The pre-fix helper returned a std (daily vol),
# which disagreed with the variance-space upstream cutoffs and inflated
# the panel's annualised vol column by ~sqrt(252). Kept as a thin alias
# around the corrected variance form so older callers (and tests
# asserting the helper is finite/positive) keep working.
_realized_vol_from_log_returns = _realized_variance_from_log_returns


def _fetch_realized_rv_yf(event_date: str, symbol: str) -> float | None:
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


def _fetch_rv_history_for_cutoffs(event_date: str, symbol: str) -> list[float]:
    """Fetch the trailing 60-day daily realized **variance** for cutoff fallback.

    Returns a list of per-bar variance values (squared log-returns) in
    the same space upstream's ``main._load_rv_history`` writes, so the
    quantiles derived here are directly comparable to the cutoffs the
    HAR-tercile model itself would have used. Returns an empty list on
    any failure so the caller knows to leave the row unresolved rather
    than raise.
    """

    try:
        from app.services.market_data import _download_close_series_in_window
        from datetime import timedelta
        from datetime import datetime as _dt

        anchor = _dt.fromisoformat(event_date).date()
        start = anchor - timedelta(days=_FALLBACK_CUTOFF_WINDOW * 2)
        end = anchor
        close_series = _download_close_series_in_window(
            symbol=symbol, start=start, end=end
        )
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
        rv = rv[np.isfinite(rv)]
        if rv.size == 0:
            return []
        tail = rv[-_FALLBACK_CUTOFF_WINDOW:]
        return [float(v) for v in tail]
    except Exception:
        return []


def _cutoffs_from_history(history: Iterable[float]) -> tuple[float | None, float | None]:
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


def _resolve_realized_tercile(
    *,
    payload: Any,
    event_date: str,
    symbol: str,
    pred_q33: float | None,
    pred_q67: float | None,
) -> tuple[str | None, float | None]:
    """Best-effort resolution of the realized tercile for one row.

    Returns ``(label, realized_rv)``. Either half may be None on a
    failure (no forward window yet, yfinance flake, missing cutoffs).
    """

    rv = _extract_persisted_realized_rv(payload)
    if rv is None:
        rv = _fetch_realized_rv_yf(event_date, symbol)
    if rv is None:
        return None, None

    if pred_q33 is not None and pred_q67 is not None and pred_q33 <= pred_q67:
        return _bucket_against_cutoffs(rv, pred_q33, pred_q67), rv

    history = _fetch_rv_history_for_cutoffs(event_date, symbol)
    if not history:
        return None, rv
    q33, q67 = _cutoffs_from_history(history)
    if q33 is None or q67 is None:
        return None, rv
    return _bucket_against_cutoffs(rv, q33, q67), rv


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

    Walks the last ``limit`` ``analysis_runs`` rows for ``symbol`` in
    chronological-descending order, extracts the predicted tercile and
    resolves the realized one, then aggregates a top-line accuracy KPI
    + per-tercile hit-rate. Returns the dict-shape the endpoint wraps
    in ``HarTercileBacktestResponse``.
    """

    stmt = (
        select(AnalysisRun)
        .where(AnalysisRun.symbol == symbol)
        .order_by(AnalysisRun.created_at.desc())
        .limit(limit)
    )
    rows = list(session.execute(stmt).scalars().all())

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        payload = row.payload
        predicted_label, predicted_prob = _extract_predicted_tercile(payload)
        if predicted_label is None:
            # No prediction we can backtest — skip the row entirely so
            # the panel's denominator stays honest.
            continue
        pred_q33, pred_q67 = _extract_persisted_cutoffs(payload)
        realized_label, realized_rv = _resolve_realized_tercile(
            payload=payload,
            event_date=row.document_date,
            symbol=row.symbol,
            pred_q33=pred_q33,
            pred_q67=pred_q67,
        )
        correct: bool | None = None
        if realized_label is not None:
            correct = realized_label == predicted_label
        out_rows.append(
            {
                "event_date": row.document_date,
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
