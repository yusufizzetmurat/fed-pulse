"""Alpha Vantage SPX intraday backfill for FOMC announcement windows.

The Cieslak-Vissing-Jorgensen (2021) decomposition uses the SPX return
in a ±30 minute window around the policy announcement. The free tier of
FRED carries only daily SPX (FRED-licensed CRSP), so `mp_surprise.py`
defaults to a same-day close-to-close proxy and stamps the row with
``fed_info_factor_source = "daily_window_proxy"``.

This module backfills the ±30 minute SPX return from Alpha Vantage's
free intraday endpoint (SPY as the SPX proxy, since direct
``^GSPC`` intraday is not available in the free tier). The cache lands
at ``data/external/alphavantage/spx_intraday_fomc_days.parquet``;
``mp_surprise.py`` reads it and re-stamps upgraded rows with
``fed_info_factor_source = "alphavantage_intraday_30min"``.

Rate limits:

- 5 requests/minute, 500/day on the free Alpha Vantage tier
- one ``TIME_SERIES_INTRADAY`` call covers one calendar month at 1-min
  resolution (~30 trading days × 390 minutes = ~11,700 rows per call)
- ~150 FOMC meetings across the 2010-2025 window means ~120 distinct
  (year, month) cells, well within a one-day budget at 13s spacing

``ALPHA_VANTAGE_API_KEY`` must be set on the environment. Get a free key
at https://www.alphavantage.co/support/#api-key.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import httpx

from app.config import DATA_DIR


ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
DEFAULT_CACHE_DIR = DATA_DIR / "external" / "alphavantage"
SOURCES_LOCK_NAME = "SOURCES.lock"
INTRADAY_PARQUET = "spx_intraday_fomc_days.parquet"
DEFAULT_SYMBOL = "SPY"
DEFAULT_INTERVAL = "1min"
DEFAULT_TIMEOUT_SECONDS = 30.0
# Alpha Vantage free tier is 5 req/min. 13 seconds keeps us safely under
# even with clock drift; the env var lets paid tiers raise the rate.
DEFAULT_REQUEST_INTERVAL_SECONDS = 13.0
# FOMC announcements at 2:00pm ET; the CVJ ±30 min window is 13:30-14:30.
DEFAULT_ANNOUNCEMENT_TIME = datetime.time(hour=14, minute=0)
DEFAULT_WINDOW_MINUTES = 30


@dataclass(frozen=True)
class IntradayBar:
    """One minute bar from Alpha Vantage's TIME_SERIES_INTRADAY."""

    timestamp_et: str  # ISO 8601 in US/Eastern, no tz suffix (matches AV format)
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class FomcWindowReturn:
    """±30 min SPX-proxy return + raw ``(pre, post)`` close pair."""

    event_date: str
    pre_close: float
    post_close: float
    return_pct: float
    window_minutes: int
    symbol: str
    fetched_at_utc: str


def _api_key_from_env() -> str:
    key = os.environ.get("ALPHA_VANTAGE_API_KEY") or os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        raise RuntimeError(
            "ALPHA_VANTAGE_API_KEY is not set. Get a free key at "
            "https://www.alphavantage.co/support/#api-key and export it."
        )
    return key


def _parse_intraday_payload(payload: dict[str, Any]) -> list[IntradayBar]:
    series_key = next(
        (k for k in payload if k.startswith("Time Series")),
        None,
    )
    if series_key is None:
        # Note message: "Thank you for using Alpha Vantage..." indicates a rate-limit hit.
        if "Note" in payload:
            raise RuntimeError(f"Alpha Vantage rate limit hit: {payload['Note']}")
        if "Error Message" in payload:
            raise RuntimeError(f"Alpha Vantage error: {payload['Error Message']}")
        raise RuntimeError(f"unexpected Alpha Vantage response shape: keys={list(payload.keys())}")
    rows = payload[series_key]
    bars: list[IntradayBar] = []
    for timestamp_et, fields in rows.items():
        bars.append(
            IntradayBar(
                timestamp_et=str(timestamp_et),
                open=float(fields["1. open"]),
                high=float(fields["2. high"]),
                low=float(fields["3. low"]),
                close=float(fields["4. close"]),
                volume=float(fields["5. volume"]),
            )
        )
    # Alpha Vantage returns the series newest-first; flip so chronological
    # callers can binary-search by timestamp.
    bars.sort(key=lambda bar: bar.timestamp_et)
    return bars


def fetch_intraday_minute_bars(
    *,
    api_key: str,
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    month: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    client: httpx.Client | None = None,
) -> list[IntradayBar]:
    """Hit ``TIME_SERIES_INTRADAY`` for one (symbol, month) pair.

    ``month`` is a ``YYYY-MM`` string (e.g. ``"2024-01"``). When None,
    Alpha Vantage returns the most recent month available — usually
    fine for live operations but explicit months are preferred for
    deterministic backfills.
    """

    params: dict[str, str] = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "outputsize": "full",
        "adjusted": "true",
        "extended_hours": "false",
        "apikey": api_key,
        "datatype": "json",
    }
    if month is not None:
        params["month"] = month
    owns_client = client is None
    if owns_client:
        client = httpx.Client(timeout=timeout_seconds)
    try:
        response = client.get(ALPHA_VANTAGE_BASE_URL, params=params)
        response.raise_for_status()
        payload = response.json()
    finally:
        if owns_client:
            client.close()
    return _parse_intraday_payload(payload)


def _et_naive(date_value: datetime.date, time_value: datetime.time) -> datetime.datetime:
    """Combine date + time into a naive datetime (no tz suffix).

    Alpha Vantage timestamps come back in US/Eastern WITHOUT a tz
    suffix; matching that format keeps the timestamp comparison
    string-based and timezone-agnostic.
    """

    return datetime.datetime.combine(date_value, time_value)


def _isoformat_minute(timestamp: datetime.datetime) -> str:
    return timestamp.strftime("%Y-%m-%d %H:%M:00")


def _bracket_window_returns(
    bars: Sequence[IntradayBar],
    event_date: datetime.date,
    *,
    announcement_time: datetime.time,
    window_minutes: int,
) -> tuple[float, float] | None:
    """Return ``(pre_close, post_close)`` for the ±window around the
    announcement, or ``None`` if either side is missing.

    The "pre" close is the last bar at or before
    ``announcement_time - window_minutes``; the "post" close is the
    first bar at or after ``announcement_time + window_minutes``.
    Using closest-bar lookups (not exact equality) tolerates the
    occasional gap in the Alpha Vantage stream.
    """

    pre_target = _et_naive(
        event_date,
        (
            datetime.datetime.combine(event_date, announcement_time)
            - datetime.timedelta(minutes=window_minutes)
        ).time(),
    )
    post_target = _et_naive(
        event_date,
        (
            datetime.datetime.combine(event_date, announcement_time)
            + datetime.timedelta(minutes=window_minutes)
        ).time(),
    )
    pre_target_iso = _isoformat_minute(pre_target)
    post_target_iso = _isoformat_minute(post_target)

    pre_close: float | None = None
    post_close: float | None = None
    for bar in bars:
        if bar.timestamp_et[:10] != event_date.isoformat():
            continue
        if bar.timestamp_et <= pre_target_iso:
            pre_close = bar.close
        if bar.timestamp_et >= post_target_iso and post_close is None:
            post_close = bar.close
            break
    if pre_close is None or post_close is None or pre_close <= 0:
        return None
    return (float(pre_close), float(post_close))


def compute_window_returns(
    bars: Sequence[IntradayBar],
    event_dates: Iterable[datetime.date],
    *,
    symbol: str = DEFAULT_SYMBOL,
    announcement_time: datetime.time = DEFAULT_ANNOUNCEMENT_TIME,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
    fetched_at_utc: str | None = None,
) -> list[FomcWindowReturn]:
    fetched_at = fetched_at_utc or datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    out: list[FomcWindowReturn] = []
    for event_date in event_dates:
        window = _bracket_window_returns(
            bars,
            event_date,
            announcement_time=announcement_time,
            window_minutes=window_minutes,
        )
        if window is None:
            continue
        pre_close, post_close = window
        out.append(
            FomcWindowReturn(
                event_date=event_date.isoformat(),
                pre_close=pre_close,
                post_close=post_close,
                return_pct=(post_close - pre_close) / pre_close,
                window_minutes=int(window_minutes),
                symbol=symbol,
                fetched_at_utc=fetched_at,
            )
        )
    return out


def _months_covering(event_dates: Iterable[datetime.date]) -> list[str]:
    """Return distinct ``YYYY-MM`` strings spanning ``event_dates``.

    One Alpha Vantage call retrieves a whole month, so grouping by
    month keeps the request count down — the rate-limit budget is the
    binding constraint, not bandwidth.
    """

    months = {f"{d.year:04d}-{d.month:02d}" for d in event_dates}
    return sorted(months)


def _write_sources_lock_entry(
    cache_dir: Path,
    parquet_path: Path,
    *,
    sha256: str,
    rows: int,
    months_fetched: Sequence[str],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / SOURCES_LOCK_NAME
    payload: dict[str, Any] = {}
    if lock_path.exists():
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    payload[parquet_path.name] = {
        "sha256": sha256,
        "rows": int(rows),
        "months_fetched": list(months_fetched),
        "fetched_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "source": "alphavantage",
        "endpoint": "TIME_SERIES_INTRADAY",
        "symbol": DEFAULT_SYMBOL,
        "interval": DEFAULT_INTERVAL,
    }
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def backfill_fomc_days(
    *,
    fomc_dates: Iterable[datetime.date],
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    api_key: str | None = None,
    request_interval_seconds: float = DEFAULT_REQUEST_INTERVAL_SECONDS,
    announcement_time: datetime.time = DEFAULT_ANNOUNCEMENT_TIME,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
    symbol: str = DEFAULT_SYMBOL,
    sleep_fn: Any = time.sleep,
    client: httpx.Client | None = None,
) -> Path:
    """Fetch ±30 min SPY closes for every ``fomc_dates`` entry and persist
    the resulting per-event returns to a parquet cache.

    The fetch is grouped by ``(year, month)`` so each Alpha Vantage call
    serves many events. ``sleep_fn`` is parameterised so tests can pass
    a no-op; the production default is ``time.sleep`` with the safe
    13-second spacing that stays under the 5/min free-tier ceiling.
    """

    import pandas as pd

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / INTRADAY_PARQUET

    date_list = sorted(set(fomc_dates))
    if not date_list:
        raise ValueError("backfill_fomc_days called with an empty fomc_dates iterable")
    months = _months_covering(date_list)
    resolved_key = api_key or _api_key_from_env()

    rows_by_date: dict[datetime.date, list[FomcWindowReturn]] = {d: [] for d in date_list}
    owns_client = client is None
    if owns_client:
        client = httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS)
    try:
        for i, month in enumerate(months):
            if i > 0:
                sleep_fn(float(request_interval_seconds))
            bars = fetch_intraday_minute_bars(
                api_key=resolved_key,
                symbol=symbol,
                interval=DEFAULT_INTERVAL,
                month=month,
                client=client,
            )
            month_dates = [d for d in date_list if f"{d.year:04d}-{d.month:02d}" == month]
            for window_row in compute_window_returns(
                bars,
                month_dates,
                symbol=symbol,
                announcement_time=announcement_time,
                window_minutes=window_minutes,
            ):
                rows_by_date[
                    datetime.date.fromisoformat(window_row.event_date)
                ].append(window_row)
    finally:
        if owns_client:
            client.close()

    flat: list[dict[str, Any]] = []
    for event_date in date_list:
        for window_row in rows_by_date.get(event_date, []):
            flat.append(
                {
                    "event_date": window_row.event_date,
                    "pre_close": window_row.pre_close,
                    "post_close": window_row.post_close,
                    "return_pct": window_row.return_pct,
                    "window_minutes": window_row.window_minutes,
                    "symbol": window_row.symbol,
                    "fetched_at_utc": window_row.fetched_at_utc,
                }
            )

    frame = pd.DataFrame(flat)
    if not frame.empty:
        frame = frame.sort_values("event_date").reset_index(drop=True)
    frame.to_parquet(parquet_path, index=False)
    _write_sources_lock_entry(
        cache_dir,
        parquet_path,
        sha256=_file_sha256(parquet_path),
        rows=len(frame),
        months_fetched=months,
    )
    print(
        f"[alphavantage_spx] wrote {len(frame)} window returns to {parquet_path} "
        f"({len(months)} month(s) fetched, "
        f"{len(date_list) - len(frame)} event(s) without coverage)"
    )
    return parquet_path


def load_window_returns(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> dict[str, float]:
    """Return ``{event_date_iso: return_pct}`` from the intraday cache.

    Missing parquet returns an empty dict so callers can degrade
    cleanly to the daily-close proxy when the backfill has not run.
    """

    import pandas as pd

    parquet_path = Path(cache_dir) / INTRADAY_PARQUET
    if not parquet_path.exists():
        return {}
    frame = pd.read_parquet(parquet_path)
    if frame.empty:
        return {}
    return {
        str(row["event_date"]): float(row["return_pct"])
        for _, row in frame.iterrows()
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill SPY 1-min closes around each FOMC announcement."
    )
    parser.add_argument(
        "--fomc-dates",
        nargs="+",
        required=True,
        help="One or more YYYY-MM-DD FOMC event dates.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
    )
    parser.add_argument(
        "--request-interval-seconds",
        type=float,
        default=DEFAULT_REQUEST_INTERVAL_SECONDS,
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    fomc_dates = [datetime.date.fromisoformat(s) for s in args.fomc_dates]
    backfill_fomc_days(
        fomc_dates=fomc_dates,
        cache_dir=args.cache_dir,
        request_interval_seconds=float(args.request_interval_seconds),
        symbol=str(args.symbol),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
