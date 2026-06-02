"""Polygon.io (Massive) SPX intraday backfill for FOMC announcement windows.

The intraday pivot (Round 6 / Path 2) replaces the forward-10-day-vol
target with the SPX reaction in the FOMC announcement window. This
module backfills the raw 1-minute SPY bars (SPX proxy, since direct
``^GSPC`` intraday is not on the Starter tier) for the 13:30-15:00 ET
slice of every FOMC day. Unlike ``alphavantage_spx.py`` it persists the
**raw bars** rather than a scalar window return, because the intraday
event builder (Phase 2) needs the pre-announcement bar *sequence* to
form market features and the post-announcement bars to form targets.

The cache lands at
``data/external/polygon/spx_intraday_fomc_days.parquet``.

``POLYGON_API_KEY`` must be set on the environment. Starter tier
($29/mo) carries 15+ years of 1-min history and unlimited calls.
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
from zoneinfo import ZoneInfo

import httpx

from app.config import DATA_DIR

POLYGON_BASE_URL = "https://api.polygon.io"
DEFAULT_CACHE_DIR = DATA_DIR / "external" / "polygon"
SOURCES_LOCK_NAME = "SOURCES.lock"
INTRADAY_PARQUET = "spx_intraday_fomc_days.parquet"
DEFAULT_SYMBOL = "SPY"
DEFAULT_TIMEOUT_SECONDS = 30.0
# Starter tier is unlimited; keep a tiny default spacing as politeness
# and so a free-tier key (5/min) still completes (12s spacing).
DEFAULT_REQUEST_INTERVAL_SECONDS = 0.0
_ET = ZoneInfo("America/New_York")
# Bars kept for each FOMC day: 13:30 ET (pre-announcement start) through
# 15:00 ET (covers both the immediate 14:00-14:30 and delayed
# 14:30-15:00 target windows).
DEFAULT_WINDOW_START = datetime.time(13, 30)
DEFAULT_WINDOW_END = datetime.time(15, 0)
# The announcement-reaction target keys off the FOMC policy *statement*
# (released 2:00pm ET). The event registry has no "fomc" kind; the
# statement kind is the 2pm decision the intraday window reacts to.
DEFAULT_FOMC_EVENT_KIND = "statement"


@dataclass(frozen=True)
class PolygonBar:
    """One 1-minute aggregate bar, timestamp normalised to naive ET."""

    timestamp_et: str  # "YYYY-MM-DD HH:MM:00" in America/New_York, no tz suffix
    open: float
    high: float
    low: float
    close: float
    volume: float


def _api_key_from_env() -> str:
    key = os.environ.get("POLYGON_API_KEY")
    if not key:
        raise RuntimeError(
            "POLYGON_API_KEY is not set. Subscribe to Polygon/Massive and export the key."
        )
    return key


def _ms_to_et_string(ms: int) -> str:
    dt = datetime.datetime.fromtimestamp(ms / 1000.0, tz=datetime.timezone.utc).astimezone(_ET)
    return dt.strftime("%Y-%m-%d %H:%M:00")


def _parse_aggs_payload(payload: dict[str, Any]) -> list[PolygonBar]:
    status = str(payload.get("status", ""))
    if status not in {"OK", "DELAYED"}:
        msg = payload.get("error") or payload.get("message") or f"status={status!r}"
        raise RuntimeError(f"Polygon error: {msg}")
    results = payload.get("results") or []
    bars = [
        PolygonBar(
            timestamp_et=_ms_to_et_string(int(row["t"])),
            open=float(row["o"]),
            high=float(row["h"]),
            low=float(row["l"]),
            close=float(row["c"]),
            volume=float(row["v"]),
        )
        for row in results
    ]
    bars.sort(key=lambda bar: bar.timestamp_et)
    return bars


def filter_window_bars(
    bars: Sequence[PolygonBar],
    event_date: datetime.date,
    *,
    window_start: datetime.time = DEFAULT_WINDOW_START,
    window_end: datetime.time = DEFAULT_WINDOW_END,
) -> list[PolygonBar]:
    """Keep bars on ``event_date`` whose ET time is in [start, end] inclusive."""

    date_iso = event_date.isoformat()
    start_iso = f"{date_iso} {window_start.strftime('%H:%M:00')}"
    end_iso = f"{date_iso} {window_end.strftime('%H:%M:00')}"
    return [b for b in bars if start_iso <= b.timestamp_et <= end_iso]


def fetch_day_minute_bars(
    *,
    api_key: str,
    event_date: datetime.date,
    symbol: str = DEFAULT_SYMBOL,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    client: httpx.Client | None = None,
) -> list[PolygonBar]:
    """Fetch every 1-min aggregate for ``symbol`` on ``event_date`` (full day)."""

    date_iso = event_date.isoformat()
    url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{symbol}/range/1/minute/{date_iso}/{date_iso}"
    params = {"adjusted": "true", "sort": "asc", "limit": "50000"}
    # Auth via Authorization header, not an ``apiKey`` query param, so the
    # key never lands in request URLs, server logs, or HTTP error messages.
    headers = {"Authorization": f"Bearer {api_key}"}
    http_client = client if client is not None else httpx.Client(timeout=timeout_seconds)
    try:
        response = http_client.get(url, params=params, headers=headers)
        response.raise_for_status()
        payload = response.json()
    finally:
        if client is None:
            http_client.close()
    return _parse_aggs_payload(payload)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_sources_lock_entry(
    cache_dir: Path, parquet_path: Path, *, sha256: str, rows: int, dates_fetched: Sequence[str]
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
        "dates_fetched": list(dates_fetched),
        "fetched_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "source": "polygon",
        "endpoint": "v2/aggs/ticker/{sym}/range/1/minute",
        "symbol": DEFAULT_SYMBOL,
    }
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def backfill_fomc_days(
    *,
    fomc_dates: Iterable[datetime.date],
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    api_key: str | None = None,
    request_interval_seconds: float = DEFAULT_REQUEST_INTERVAL_SECONDS,
    symbol: str = DEFAULT_SYMBOL,
    window_start: datetime.time = DEFAULT_WINDOW_START,
    window_end: datetime.time = DEFAULT_WINDOW_END,
    sleep_fn: Any = time.sleep,
    client: httpx.Client | None = None,
) -> Path:
    """Fetch + window-filter 1-min bars for every FOMC day; persist raw bars.

    One Polygon call per date (each call returns a full trading day).
    ``sleep_fn`` is injectable so tests pass a no-op.
    """

    import pandas as pd

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / INTRADAY_PARQUET

    date_list = sorted(set(fomc_dates))
    if not date_list:
        raise ValueError("backfill_fomc_days called with an empty fomc_dates iterable")
    resolved_key = api_key or _api_key_from_env()
    fetched_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    flat: list[dict[str, Any]] = []
    skipped_unauthorized: list[str] = []
    http_client = client if client is not None else httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS)
    try:
        for i, event_date in enumerate(date_list):
            if i > 0 and request_interval_seconds > 0:
                sleep_fn(float(request_interval_seconds))
            try:
                day_bars = fetch_day_minute_bars(
                    api_key=resolved_key, event_date=event_date, symbol=symbol, client=http_client
                )
            except httpx.HTTPStatusError as exc:
                # 403 NOT_AUTHORIZED means the date predates the plan's
                # history window; skip it rather than abort the whole
                # backfill. Any other status is a real error — re-raise.
                if exc.response.status_code == 403:
                    skipped_unauthorized.append(event_date.isoformat())
                    continue
                raise
            for bar in filter_window_bars(
                day_bars, event_date, window_start=window_start, window_end=window_end
            ):
                flat.append(
                    {
                        "event_date": event_date.isoformat(),
                        "timestamp_et": bar.timestamp_et,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                        "symbol": symbol,
                        "fetched_at_utc": fetched_at,
                    }
                )
    finally:
        if client is None:
            http_client.close()

    frame = pd.DataFrame(
        flat,
        columns=[
            "event_date",
            "timestamp_et",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "symbol",
            "fetched_at_utc",
        ],
    )
    if not frame.empty:
        frame = frame.sort_values(["event_date", "timestamp_et"]).reset_index(drop=True)
    frame.to_parquet(parquet_path, index=False)
    _write_sources_lock_entry(
        cache_dir,
        parquet_path,
        sha256=_file_sha256(parquet_path),
        rows=len(frame),
        dates_fetched=[d.isoformat() for d in date_list],
    )
    covered = frame["event_date"].nunique() if not frame.empty else 0
    print(
        f"[polygon_spx] wrote {len(frame)} bars across {covered}/{len(date_list)} "
        f"FOMC day(s) to {parquet_path}"
    )
    if skipped_unauthorized:
        print(
            f"[polygon_spx] skipped {len(skipped_unauthorized)} date(s) outside the plan's "
            f"history window (403): {skipped_unauthorized[0]}..{skipped_unauthorized[-1]}"
        )
    return parquet_path


def load_intraday_bars(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> dict[str, list[PolygonBar]]:
    """Return ``{event_date_iso: [PolygonBar, ...]}`` from the cache.

    Missing parquet returns an empty dict so callers can degrade
    cleanly when the backfill has not run.
    """

    import pandas as pd

    parquet_path = Path(cache_dir) / INTRADAY_PARQUET
    if not parquet_path.exists():
        return {}
    frame = pd.read_parquet(parquet_path)
    if frame.empty:
        return {}
    out: dict[str, list[PolygonBar]] = {}
    for date_iso, group in frame.groupby("event_date"):
        bars = [
            PolygonBar(
                timestamp_et=str(r["timestamp_et"]),
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=float(r["volume"]),
            )
            for _, r in group.sort_values("timestamp_et").iterrows()
        ]
        out[str(date_iso)] = bars
    return out


def fomc_dates_from_events_parquet(
    events_parquet: Path | str,
    *,
    event_kind: str | None = DEFAULT_FOMC_EVENT_KIND,
    min_date: datetime.date | None = None,
) -> list[datetime.date]:
    """Distinct sorted FOMC announcement dates from a training package.

    The pivot target is the reaction to the policy *statement* released
    at 2:00pm ET, so the default kind is ``"statement"`` (the event
    registry has no ``"fomc"`` kind; kinds are statement / minutes /
    speech / press_conference / testimony). ``min_date`` drops dates
    before the provider's intraday-history floor — Polygon Starter
    carries ~15 years of 1-min bars, so pre-2010 statements would come
    back empty and also predate the consistent 2:00pm release.
    """

    import pandas as pd

    frame = pd.read_parquet(events_parquet)
    if event_kind is not None and "event_kind" in frame.columns:
        frame = frame[frame["event_kind"].astype(str).str.lower() == event_kind]
    dates = sorted({datetime.date.fromisoformat(str(d)[:10]) for d in frame["event_date"]})
    if min_date is not None:
        dates = [d for d in dates if d >= min_date]
    return dates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill SPY 1-min bars for the 13:30-15:00 ET window of each FOMC day."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--fomc-dates", nargs="+", help="One or more YYYY-MM-DD FOMC dates.")
    src.add_argument(
        "--events-parquet",
        type=Path,
        help="Path to a training package events.parquet; FOMC event_dates are read from it.",
    )
    parser.add_argument(
        "--event-kind",
        default=DEFAULT_FOMC_EVENT_KIND,
        help="event_kind to select from --events-parquet (default: statement).",
    )
    parser.add_argument(
        "--since",
        type=datetime.date.fromisoformat,
        default=None,
        help="Drop event dates before this YYYY-MM-DD (provider history floor).",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--request-interval-seconds", type=float, default=DEFAULT_REQUEST_INTERVAL_SECONDS
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.fomc_dates:
        dates = [datetime.date.fromisoformat(s) for s in args.fomc_dates]
        if args.since is not None:
            dates = [d for d in dates if d >= args.since]
    else:
        dates = fomc_dates_from_events_parquet(
            args.events_parquet, event_kind=args.event_kind, min_date=args.since
        )
    backfill_fomc_days(
        fomc_dates=dates,
        cache_dir=args.cache_dir,
        request_interval_seconds=float(args.request_interval_seconds),
        symbol=str(args.symbol),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
