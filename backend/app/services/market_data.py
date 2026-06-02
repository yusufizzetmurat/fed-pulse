from __future__ import annotations

import csv
import json
import os
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yfinance as yf

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "data" / "raw" / "market"
FOMC_MEETINGS_CSV = REPO_ROOT / "data" / "external" / "fomc_meetings_2010_2026.csv"

# The FOMC press release lands at 2pm Eastern. Any market-side feature dated
# on an FOMC day with a timestamp strictly after this boundary is looking at
# information the model would not have had at decision time, so we reject it
# at the feature-assembly seam. The comparison is done in America/New_York
# local time so DST is handled automatically: EST months land 14:00 ET at
# 19:00 UTC, EDT months at 18:00 UTC, and either way the assertion fires
# on bars later than 14:00 local time.
FOMC_LOCAL_CUTOFF_TIME = time(14, 0, 0)
FOMC_ZONE = ZoneInfo("America/New_York")


def _market_source() -> str:
    return (os.environ.get("FED_PULSE_MARKET_SOURCE") or "live").strip().lower()


@lru_cache(maxsize=1)
def _fomc_days() -> frozenset[date]:
    """Load the scheduled / unscheduled FOMC meeting dates as a set.

    The CSV is checked into ``data/external/`` (PR #154) and never changes
    at runtime, so caching the parsed set is safe. The set is consulted by
    ``assert_fomc_day_market_cutoff`` to decide whether the cutoff applies.
    """

    if not FOMC_MEETINGS_CSV.exists():
        return frozenset()
    days: set[date] = set()
    with FOMC_MEETINGS_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = (row.get("meeting_date") or "").strip()
            if not value:
                continue
            try:
                days.add(datetime.strptime(value, "%Y-%m-%d").date())
            except ValueError:
                continue
    return frozenset(days)


def is_fomc_day(value: date) -> bool:
    """Return True when `value` is a scheduled or unscheduled FOMC meeting."""

    return value in _fomc_days()


def assert_fomc_day_market_cutoff(
    timestamp: datetime,
    *,
    feature_name: str = "market feature",
) -> None:
    """Hard-fail when a same-day market feature lands after the 14:00 ET cutoff.

    The FOMC statement is released at 2pm Eastern. Any market-side bar (close,
    volatility, rolling stat) whose timestamp falls on an FOMC day **and**
    sits strictly after 14:00 ET embeds post-announcement information that
    the model would not have had at decision time. Letting such a bar into
    the feature frame is a textbook lookahead bug.

    ``timestamp`` is expected to be tz-aware (UTC or otherwise). A naive
    datetime is interpreted as UTC. The comparison happens in
    ``America/New_York`` local time so DST is handled automatically and
    the cutoff is the same 14:00 wall-clock both halves of the year.

    Raises ``ValueError`` on violation so the caller can surface a clear
    error to the operator. No silent coercion: a bad row is a bug, not a
    rounding error.
    """

    if not is_fomc_day(timestamp.date()):
        return  # Not an FOMC day -> cutoff does not apply.

    if timestamp.tzinfo is None:
        anchored = timestamp.replace(tzinfo=timezone.utc)
    else:
        anchored = timestamp
    local = anchored.astimezone(FOMC_ZONE)
    cutoff = datetime.combine(local.date(), FOMC_LOCAL_CUTOFF_TIME, tzinfo=FOMC_ZONE)
    if local > cutoff:
        raise ValueError(
            f"{feature_name} dated {local.isoformat()} is after the FOMC "
            f"14:00 ET cutoff ({cutoff.isoformat()}) on a meeting day. "
            "Same-day post-announcement bars leak the policy decision into the "
            "feature frame. Drop the bar or rebuild the feature with an "
            "as-of timestamp ≤ 14:00 ET."
        )


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
def _load_snapshot_series(symbol: str) -> Any:
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


def _snapshot_frame_to_series(frame: Any) -> Any:
    import pandas as pd

    if "date" not in frame.columns or "close" not in frame.columns:
        raise RuntimeError(
            "Snapshot parquet must contain 'date' and 'close' columns; got "
            f"{list(frame.columns)}"
        )
    index = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    series = pd.Series(frame["close"].astype(float).to_numpy(), index=index, name="Close")
    return series.sort_index().dropna()


def _close_series_from_frame(frame: Any) -> Any:
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


def _snapshot_window(symbol: str, start: date, end: date) -> Any:
    series = _load_snapshot_series(symbol)
    return series.loc[(series.index.date >= start) & (series.index.date < end)]


@lru_cache(maxsize=128)
def _download_close_series_in_window(symbol: str, start: date, end: date) -> Any:
    # Returns a defensive copy of the underlying Series. The lru_cache
    # holds the canonical object; copying on the way out keeps any caller
    # in-place mutation (``.iloc[i] = ...``, ``.where(..., inplace=True)``)
    # from corrupting the cached value for the next request.
    if _market_source() == "snapshot":
        window = _snapshot_window(symbol, start, end)
        if window.empty:
            raise RuntimeError(
                f"Snapshot has no rows for {symbol} in [{start}, {end}). "
                "Re-run scripts/snapshot_market_data.py to widen the window."
            )
        return window.copy()

    try:
        ticker = yf.Ticker(symbol)
        frame = ticker.history(
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=True,
        )
    except Exception as exc:
        # yfinance throws YFRateLimitError plus various transient network
        # errors. When a committed snapshot covers the window, prefer the
        # snapshot over surfacing the upstream failure to the user.
        try:
            window = _snapshot_window(symbol, start, end)
        except Exception:
            window = None
        if window is not None and not window.empty:
            return window.copy()
        raise RuntimeError(
            f"Live market fetch failed for {symbol} ({type(exc).__name__}); "
            "no snapshot fallback available. "
            f"Run scripts/snapshot_market_data.py --symbols {symbol} to seed one."
        ) from exc

    if frame.empty:
        raise RuntimeError(f"No market data found for {symbol}")

    close_series = _close_series_from_frame(frame)
    if close_series.empty:
        raise RuntimeError(f"No close prices available for {symbol}")
    return close_series.copy()


def _download_close_series(
    symbol: str, requested_date: date, lookback_days: int, extra_days: int
) -> Any:
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
    vol = (
        float(rolling.loc[:latest_idx].iloc[-1])
        if not rolling.loc[:latest_idx].dropna().empty
        else 0.0
    )

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


def fetch_event_study_window(
    event_date: str,
    symbol: str = "^GSPC",
    steps: int = 10,
    window_days: int = 30,
) -> list[dict[str, float | str]]:
    """Return the first ``steps`` trading bars strictly after ``event_date``.

    Each bar carries ``date``, ``close`` and ``log_return`` (vs the prior
    close on the same series). Window is anchored at ``event_date`` and
    extended forward by ``window_days`` calendar days so weekends and
    holidays still surface ten trading bars.
    """

    if steps < 1:
        raise ValueError("steps must be >= 1")
    if window_days < steps:
        raise ValueError("window_days must be >= steps")

    requested_date = _parse_iso_date(event_date)
    start = requested_date
    end = requested_date + timedelta(days=window_days + 1)

    close_series = _download_close_series_in_window(symbol=symbol, start=start, end=end)
    forward = close_series.loc[close_series.index.date > requested_date]
    if forward.empty:
        return []

    bars: list[dict[str, float | str]] = []
    prev_close: float | None = None
    # Anchor the first log-return against the last close on or before the
    # event date when present, otherwise leave it 0.0 (no prior reference).
    pre = close_series.loc[close_series.index.date <= requested_date]
    if not pre.empty:
        prev_close = float(pre.iloc[-1])

    import math

    for idx, value in forward.head(steps).items():
        close_value = float(value)
        if prev_close is not None and prev_close > 0:
            log_return = math.log(close_value / prev_close)
        else:
            log_return = 0.0
        bars.append(
            {
                "date": idx.date().isoformat(),
                "close": close_value,
                "log_return": log_return,
            }
        )
        prev_close = close_value
    return bars


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
