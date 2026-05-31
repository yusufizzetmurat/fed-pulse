"""Daily realized-volatility measures from intraday 5-minute SPY bars.

The daily-close realized-vol proxy used elsewhere is noisy and misses the
intraday structure HAR needs. This module backfills continuous 5-minute
bars (Alpha Vantage, full regular-session days) and reduces each day to the
standard realized measures used in the HAR/HARQ literature:

  - RV   realized variance   = Σ r_i²            (Andersen-Bollerslev)
  - RS±  realized semivariance = Σ r_i² · 1(r_i ≷ 0)   (RV = RS⁺ + RS⁻)
  - BV   bipower variation   = (π/2) Σ |r_i||r_{i-1}|  (jump-robust)
  - RQ   realized quarticity = (n/3) Σ r_i⁴      (the HARQ correction term)

r_i are intraday 5-minute log returns within a single trading day. Output is
a daily series (`data/external/alphavantage_bars/spx_5min_daily_rv.parquet`)
that the intraday HAR/HARQ + DL forecaster consumes.
"""

from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.config import DATA_DIR
from app.data.alphavantage_spx import _api_key_from_env, fetch_intraday_minute_bars

DEFAULT_RV_PARQUET = DATA_DIR / "external" / "alphavantage_bars" / "spx_5min_daily_rv.parquet"
DEFAULT_INTERVAL = "5min"
DEFAULT_SYMBOL = "SPY"
MIN_RETURNS_PER_DAY = 20  # drop half-days / sparse sessions


def daily_realized_measures(closes: list[float]) -> dict[str, float] | None:
    """Realized measures from one day's intraday closes (chronological)."""

    c = np.asarray(closes, dtype=np.float64)
    if len(c) < MIN_RETURNS_PER_DAY + 1 or np.any(c <= 0):
        return None
    r = np.diff(np.log(c))  # 5-minute log returns
    rv = float(np.sum(r**2))
    rs_pos = float(np.sum(r[r > 0] ** 2))
    rs_neg = float(np.sum(r[r < 0] ** 2))
    bv = float((np.pi / 2) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))
    n = len(r)
    rq = float((n / 3) * np.sum(r**4))
    return {"rv": rv, "rs_pos": rs_pos, "rs_neg": rs_neg, "bv": bv, "rq": rq, "n_ret": float(n)}


def _measures_by_day(bars: list[Any]) -> dict[str, dict[str, float]]:
    """Group AV intraday bars by ET date → daily realized measures."""

    by_date: dict[str, list[float]] = {}
    for bar in bars:
        by_date.setdefault(bar.timestamp_et[:10], []).append(bar.close)
    out: dict[str, dict[str, float]] = {}
    for date_iso, closes in by_date.items():
        m = daily_realized_measures(closes)
        if m is not None:
            out[date_iso] = m
    return out


def backfill_intraday_rv(
    *,
    start_month: str,
    end_month: str,
    cache_path: Path | str = DEFAULT_RV_PARQUET,
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    api_key: str | None = None,
    request_interval_seconds: float = 1.0,
    sleep_fn: Any = time.sleep,
    client: Any = None,
) -> Path:
    """Fetch full-session 5-min bars month-by-month; persist a daily RV series."""

    import pandas as pd

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    months = _months_between(start_month, end_month)
    resolved_key = api_key or _api_key_from_env()

    rows: list[dict[str, Any]] = []
    for i, month in enumerate(months):
        if i > 0 and request_interval_seconds > 0:
            sleep_fn(float(request_interval_seconds))
        bars = fetch_intraday_minute_bars(
            api_key=resolved_key,
            symbol=symbol,
            interval=interval,
            month=month,
            client=client,
        )
        for date_iso, m in _measures_by_day(bars).items():
            rows.append({"date": date_iso, **m})

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    frame.to_parquet(cache_path, index=False)
    span = f"{frame['date'].iloc[0]}..{frame['date'].iloc[-1]}" if not frame.empty else "empty"
    print(f"[intraday_realized] wrote {len(frame)} daily-RV rows ({span}) to {cache_path}")
    return cache_path


def _months_between(start_month: str, end_month: str) -> list[str]:
    """Inclusive list of YYYY-MM strings from start to end."""

    sy, sm = (int(x) for x in start_month.split("-"))
    ey, em = (int(x) for x in end_month.split("-"))
    out: list[str] = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill daily realized-vol measures from 5-min SPY."
    )
    parser.add_argument("--start-month", required=True, help="YYYY-MM")
    parser.add_argument(
        "--end-month",
        default=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m"),
        help="YYYY-MM (default: current month)",
    )
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_RV_PARQUET)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--request-interval-seconds", type=float, default=1.0)
    args = parser.parse_args()
    backfill_intraday_rv(
        start_month=args.start_month,
        end_month=args.end_month,
        cache_path=args.cache_path,
        symbol=str(args.symbol),
        request_interval_seconds=float(args.request_interval_seconds),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
