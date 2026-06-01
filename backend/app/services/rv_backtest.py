"""Backtest the last N QLIKE-RV predictions against realized RV.

Walks the persisted ``analysis_runs`` table for ``^GSPC``, aligns each
row's event date to the daily realized variance series, replays the
QLIKE-DLq h=1 ensemble + conformal bands on the leading prefix, and
counts how many resolved rows fell inside the published 80% / 90%
bands. Powers the RvAccuracyPanel card: empirical band coverage as a
top-line KPI plus a per-row table the user can scan for outliers.

The accuracy surface is **empirical band coverage** (in-band hit rate)
rather than tercile / direction accuracy — that mirrors how the
production model is calibrated. The conformal quantiles are tuned so
80% of out-of-sample residuals land inside the 80% band on the eval
pool; the backtest simply verifies that the published bands still hold
their nominal coverage on the recent FOMC rows.

Per-row resolution:
  * Look up the row's ``document_date`` in the daily RV history.
  * If the date sits at or beyond the warmup horizon (the HAR monthly
    lag needs ~22 days of leading data), run the cached ensemble on
    the leading prefix to derive the h=1 point + bands.
  * The realized RV is the actual variance for the same bar; the row
    is "in band" when the realized number falls between band_lo and
    band_hi.
  * Rows whose event date sits before the warmup horizon — or beyond
    the right edge of the available RV history — surface as pending.

The yfinance / parquet fetch for the RV history is shared with the
``/forecast/realized-vol`` endpoint via ``app.main._load_rv_history``,
so a backtest pull on a warm cache reuses the same data the
VolatilityOutlookCard already consumes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data.intraday_rv_forecast import _EPS as _RVEPS
from app.db import AnalysisRun
from app.services.rv_forecaster import (
    _HISTORICAL_BANDS_WARMUP,
    _RvPredictor,
    _ensemble_log_rv,
)


_FORECAST_HORIZON = 1

# Backtest history window — wide enough to align persisted FOMC event
# dates from the trailing year (~8 meetings/yr × 2y = ~16 events, well
# above the default ``limit=10``) against the daily RV series. The live
# ``/forecast/realized-vol`` card stays on the narrower 60-day default;
# only the backtest pulls this wider tail so older persisted runs land
# inside the dates index rather than the pre-history miss branch.
_BACKTEST_HISTORY_DAYS = 504


def _load_rv_series(symbol: str) -> tuple[list[float], list[str]]:
    """Pull the daily RV series + ISO date stamps for ``symbol``.

    Defers to ``app.main._load_rv_history`` so the backtest reads from
    the same intraday-parquet-preferred / yfinance-fallback path the
    ``/forecast/realized-vol`` endpoint uses, but requests a much wider
    trailing window — the backtest needs to align persisted FOMC event
    dates from prior months, not just the live forecast prefix. The
    import is local so the service module does not pull main's full
    transitive surface at import time.
    """

    from app.main import _load_rv_history

    rv, dates = _load_rv_history(symbol, days=_BACKTEST_HISTORY_DAYS)
    return rv, dates


def _predict_h1_bands(
    rv_prefix: np.ndarray, predictor: _RvPredictor
) -> tuple[float, float, float, float, float]:
    """Run the QLIKE-DLq h=1 ensemble on ``rv_prefix``.

    Returns ``(point, lo80, hi80, lo90, hi90)`` in RV (variance) space.
    Reuses the singleton predictor + ``_ensemble_log_rv`` from
    :mod:`app.services.rv_forecaster` so the bands match the live
    ``/forecast/realized-vol`` surface byte-for-byte.
    """

    log_rv = np.log(rv_prefix + _RVEPS)
    row = predictor.spec["by_horizon"][f"h{_FORECAST_HORIZON}"]
    seeds = predictor.seed_models[f"h{_FORECAST_HORIZON}"]
    log_point = _ensemble_log_rv(log_rv, row, seeds)
    quants = row["conformal_quantiles"]
    q80 = float(quants.get("0.20", 0.0))
    q90 = float(quants.get("0.10", 0.0))
    point = float(np.exp(log_point))
    lo80 = float(np.exp(log_point - q80))
    hi80 = float(np.exp(log_point + q80))
    lo90 = float(np.exp(log_point - q90))
    hi90 = float(np.exp(log_point + q90))
    return point, lo80, hi80, lo90, hi90


def _resolve_row(
    *,
    event_date: str,
    rv: np.ndarray,
    dates: list[str],
    predictor: _RvPredictor,
) -> dict[str, Any] | None:
    """Build a single backtest row for ``event_date``.

    Returns None when the date is not in the available RV history at
    all — the caller surfaces the row as pending in that case via a
    separate code path. Returns a populated row dict when the event
    date sits at or beyond the warmup horizon; ``realized_rv`` is the
    variance observed on the same bar, and ``in_band_*`` are populated
    accordingly. Rows whose date falls inside the warmup window are
    returned with ``realized_rv = None`` and pending hit flags so the
    panel still renders them.
    """

    try:
        idx = dates.index(event_date)
    except ValueError:
        return None
    if idx < _HISTORICAL_BANDS_WARMUP:
        # Not enough leading history to run HAR's monthly lag honestly.
        # Surface the row as pending rather than dropping it; the panel
        # still keeps a placeholder so the counter stays meaningful.
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
            "_pending": True,
        }
    prefix = rv[:idx]
    point, lo80, hi80, lo90, hi90 = _predict_h1_bands(prefix, predictor)
    realized = float(rv[idx])
    return {
        "event_date": event_date,
        "point_forecast_rv": point,
        "band_lo_80": lo80,
        "band_hi_80": hi80,
        "band_lo_90": lo90,
        "band_hi_90": hi90,
        "realized_rv": realized,
        "in_band_80": bool(lo80 <= realized <= hi80),
        "in_band_90": bool(lo90 <= realized <= hi90),
    }


def _aggregate_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute empirical band coverage across resolved rows.

    Denominator for each coverage figure is the count of rows whose
    realized RV resolved (``realized_rv`` not None). The nominal
    coverage levels are pinned at the calibration targets so the
    frontend can render the gap chip without re-deriving them.

    ``pending_runs`` carries the rows that could not be resolved — the
    event date either sat inside HAR's warmup window or fell outside the
    available RV history. Keeping it separate from ``resolved_runs``
    lets the panel show "X resolved / Y pending" without polluting the
    coverage ratio with rows we never even attempted to score.
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


def get_rv_backtest(
    session: Session,
    *,
    symbol: str = "^GSPC",
    limit: int = 10,
) -> dict[str, Any]:
    """Assemble the QLIKE-RV backtest payload for the panel.

    Walks the last ``limit`` ``analysis_runs`` rows for ``symbol`` in
    chronological-descending order, replays the QLIKE-DLq h=1 ensemble
    on each event date's leading RV prefix, and tallies empirical band
    coverage against the realized RV at the same bar. Returns the
    dict-shape the endpoint wraps in ``RvBacktestResponse``.
    """

    stmt = (
        select(AnalysisRun)
        .where(AnalysisRun.symbol == symbol)
        .order_by(AnalysisRun.created_at.desc())
        .limit(limit)
    )
    run_rows = list(session.execute(stmt).scalars().all())

    if not run_rows:
        return {
            "symbol": symbol,
            "horizon": _FORECAST_HORIZON,
            "rows": [],
            "coverage": _aggregate_coverage([]),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    rv_list, dates = _load_rv_series(symbol)
    rv = np.asarray(rv_list, dtype=np.float64)
    predictor = _RvPredictor.get()

    out_rows: list[dict[str, Any]] = []
    for run in run_rows:
        event_date = str(run.document_date)
        resolved = _resolve_row(
            event_date=event_date,
            rv=rv,
            dates=dates,
            predictor=predictor,
        )
        if resolved is None:
            # Event sits outside the available RV history (future
            # meeting / pre-history). Surface as a pending row so the
            # panel's denominator stays meaningful.
            out_rows.append(
                {
                    "event_date": event_date,
                    "point_forecast_rv": float("nan"),
                    "band_lo_80": float("nan"),
                    "band_hi_80": float("nan"),
                    "band_lo_90": float("nan"),
                    "band_hi_90": float("nan"),
                    "realized_rv": None,
                    "in_band_80": None,
                    "in_band_90": None,
                }
            )
            continue
        resolved.pop("_pending", None)
        out_rows.append(resolved)

    coverage = _aggregate_coverage(out_rows)
    return {
        "symbol": symbol,
        "horizon": _FORECAST_HORIZON,
        "rows": out_rows,
        "coverage": coverage,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
