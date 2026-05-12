from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import yfinance as yf

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "data" / "raw" / "market"


def _market_source() -> str:
    return (os.environ.get("FED_PULSE_MARKET_SOURCE") or "live").strip().lower()


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("^", "").replace("=", "_").replace("/", "_").replace(":", "_")


def _snapshot_lock_path() -> Path:
    override = os.environ.get("FED_PULSE_MARKET_SNAPSHOT_DIR")
    base = Path(override) if override else DEFAULT_SNAPSHOT_DIR
    return base / "SOURCES.lock"


def _snapshot_dir() -> Path:
    override = os.environ.get("FED_PULSE_MARKET_SNAPSHOT_DIR")
    return Path(override) if override else DEFAULT_SNAPSHOT_DIR


@lru_cache(maxsize=64)
def _load_snapshot_series(symbol: str):
    import pandas as pd

    snapshot_dir = _snapshot_dir()
    lock_path = _snapshot_lock_path()
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        entry = (lock.get("entries") or {}).get(symbol)
        if entry and entry.get("parquet_path"):
            candidate = REPO_ROOT / entry["parquet_path"]
            if candidate.exists():
                frame = pd.read_parquet(candidate)
                return _snapshot_frame_to_series(frame)
    fallback = snapshot_dir / f"{_safe_symbol(symbol)}.parquet"
    if not fallback.exists():
        raise FileNotFoundError(
            f"Market snapshot for symbol={symbol!r} is missing at {fallback}. "
            f"Run `python scripts/snapshot_market_data.py --symbols {symbol} ...` "
            f"or set FED_PULSE_MARKET_SOURCE=live."
        )
    frame = pd.read_parquet(fallback)
    return _snapshot_frame_to_series(frame)


def _snapshot_frame_to_series(frame):
    import pandas as pd

    if "date" not in frame.columns or "close" not in frame.columns:
        raise RuntimeError(
            "Snapshot parquet must contain 'date' and 'close' columns; got "
            f"{list(frame.columns)}"
        )
    index = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    series = pd.Series(frame["close"].astype(float).to_numpy(), index=index, name="Close")
    return series.sort_index().dropna()


def _close_series_from_frame(frame):
    close_data = frame["Close"]
    if hasattr(close_data, "columns"):
        # yfinance may return a DataFrame (e.g., MultiIndex columns).
        if close_data.shape[1] == 0:
            raise RuntimeError("No close prices available")
        return close_data.iloc[:, 0].dropna()
    return close_data.dropna()


def _parse_iso_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("date must be in YYYY-MM-DD format") from exc


def _download_close_series_in_window(symbol: str, start: date, end: date):
    if _market_source() == "snapshot":
        series = _load_snapshot_series(symbol)
        window = series.loc[
            (series.index.date >= start) & (series.index.date < end)
        ]
        if window.empty:
            raise RuntimeError(
                f"Snapshot has no rows for {symbol} in [{start}, {end}). "
                "Re-run scripts/snapshot_market_data.py to widen the window."
            )
        return window

    ticker = yf.Ticker(symbol)
    frame = ticker.history(
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
    )
    if frame.empty:
        raise RuntimeError(f"No market data found for {symbol}")

    close_series = _close_series_from_frame(frame)
    if close_series.empty:
        raise RuntimeError(f"No close prices available for {symbol}")
    return close_series


def _download_close_series(symbol: str, requested_date: date, lookback_days: int, extra_days: int):
    start = requested_date - timedelta(days=lookback_days + extra_days)
    end = requested_date + timedelta(days=1)
    return _download_close_series_in_window(symbol=symbol, start=start, end=end)


def fetch_market_snapshot(
    target_date: str,
    symbol: str = "^GSPC",
    lookback_days: int = 7,
    volatility_window: int = 5,
) -> dict[str, Any]:
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")
    if volatility_window < 2:
        raise ValueError("volatility_window must be >= 2")

    requested_date = _parse_iso_date(target_date)
    close_series = _download_close_series(
        symbol=symbol,
        requested_date=requested_date,
        lookback_days=max(lookback_days, volatility_window + 2),
        extra_days=12,
    )

    valid = close_series.loc[close_series.index.date <= requested_date]
    if valid.empty:
        raise RuntimeError(f"No market data on or before {requested_date.isoformat()} for {symbol}")

    latest_idx = valid.index[-1]
    date_used = latest_idx.date()
    lag_days = (requested_date - date_used).days
    if lag_days > lookback_days:
        raise RuntimeError(
            f"Nearest trading day is {lag_days} days before requested date; increase lookback window."
        )

    returns = close_series.pct_change().dropna()
    rolling = returns.rolling(volatility_window).std()
    vol = float(rolling.loc[:latest_idx].iloc[-1]) if not rolling.loc[:latest_idx].dropna().empty else 0.0

    return {
        "symbol": symbol,
        "requested_date": requested_date.isoformat(),
        "date_used": date_used.isoformat(),
        "lookback_days": lookback_days,
        "close": float(valid.iloc[-1]),
        "volatility_5d": vol,
    }


def fetch_market_sequence(
    target_date: str,
    symbol: str = "^GSPC",
    sequence_length: int = 5,
    lookback_days: int = 14,
) -> list[dict[str, float | str]]:
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")

    requested_date = _parse_iso_date(target_date)
    close_series = _download_close_series(
        symbol=symbol,
        requested_date=requested_date,
        lookback_days=max(lookback_days, sequence_length + 5),
        extra_days=16,
    )

    valid = close_series.loc[close_series.index.date <= requested_date]
    if valid.empty:
        raise RuntimeError(f"No market data on or before {requested_date.isoformat()} for {symbol}")

    returns = close_series.pct_change().dropna()
    rolling = returns.rolling(5).std()

    points: list[dict[str, float | str]] = []
    for idx, close_value in valid.tail(sequence_length).items():
        vol = rolling.loc[:idx].iloc[-1] if not rolling.loc[:idx].dropna().empty else 0.0
        points.append(
            {
                "date": idx.date().isoformat(),
                "close": float(close_value),
                "volatility_5d": float(vol),
            }
        )
    return points


def fetch_market_history(
    target_date: str,
    symbol: str = "^GSPC",
    history_length: int = 30,
    lookback_days: int = 45,
) -> list[dict[str, float | str]]:
    adaptive_lookback = max(lookback_days, int(history_length * 2.2) + 20)
    return fetch_market_sequence(
        target_date=target_date,
        symbol=symbol,
        sequence_length=history_length,
        lookback_days=adaptive_lookback,
    )


def fetch_realized_forward(
    target_date: str,
    symbol: str = "^GSPC",
    steps: int = 3,
    lookback_days: int = 45,
) -> list[dict[str, float | str]]:
    if steps < 1:
        raise ValueError("steps must be >= 1")

    requested_date = _parse_iso_date(target_date)
    start = requested_date - timedelta(days=lookback_days)
    end = requested_date + timedelta(days=max(steps * 4, 16))

    close_series = _download_close_series_in_window(symbol=symbol, start=start, end=end)
    returns = close_series.pct_change().dropna()
    rolling = returns.rolling(5).std()

    future = close_series.loc[close_series.index.date > requested_date]
    if future.empty:
        return []

    realized: list[dict[str, float | str]] = []
    for idx, close_value in future.head(steps).items():
        vol = rolling.loc[:idx].iloc[-1] if not rolling.loc[:idx].dropna().empty else 0.0
        realized.append(
            {
                "date": idx.date().isoformat(),
                "close": float(close_value),
                "volatility_5d": float(vol),
            }
        )
    return realized


def fetch_forward_trading_dates(
    target_date: str,
    symbol: str = "^GSPC",
    steps: int = 3,
    lookback_days: int = 45,
) -> list[str]:
    if steps < 1:
        raise ValueError("steps must be >= 1")

    requested_date = _parse_iso_date(target_date)
    start = requested_date - timedelta(days=lookback_days)
    end = requested_date + timedelta(days=max(steps * 4, 16))

    close_series = _download_close_series_in_window(symbol=symbol, start=start, end=end)
    future = close_series.loc[close_series.index.date > requested_date]
    return [idx.date().isoformat() for idx in future.head(steps).index]
