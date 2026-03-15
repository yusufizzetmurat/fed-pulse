from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import yfinance as yf


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


def _download_close_series(symbol: str, requested_date: date, lookback_days: int, extra_days: int):
    start = requested_date - timedelta(days=lookback_days + extra_days)
    end = requested_date + timedelta(days=1)
    frame = yf.download(
        symbol,
        start=start.isoformat(),
        end=end.isoformat(),
        progress=False,
        auto_adjust=True,
    )
    if frame.empty:
        raise RuntimeError(f"No market data found for {symbol}")

    close_series = _close_series_from_frame(frame)
    if close_series.empty:
        raise RuntimeError(f"No close prices available for {symbol}")
    return close_series


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
