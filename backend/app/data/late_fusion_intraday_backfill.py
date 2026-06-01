"""Backfill pre-2013 FOMC-day intraday SPY bars for the event-frame extension.

The existing intraday cache covers FOMC statement days 2013-2026 (announcement at
2:00pm ET). To grow the event frame we add the 2006-2012 statement days — but the
announcement time that era was 2:15pm (and 12:30pm for the early press-conference
meetings), so we fetch a WIDE 12:00-16:00 ET window per day. The event builder
then locates the announcement from the intraday volume spike rather than assuming
a fixed clock time, so all eras align correctly.

One Alpha Vantage TIME_SERIES_INTRADAY call returns a whole month; we fetch each
covering month once and slice out the FOMC days.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

from app.config import DATA_DIR

logger = logging.getLogger(__name__)
_URL = "https://www.alphavantage.co/query"
_WINDOW_START, _WINDOW_END = "12:00:00", "16:00:00"


def _api_key() -> str:
    for name in ("ALPHAVANTAGE_API_KEY", "ALPHA_VANTAGE_API_KEY", "AV_API_KEY"):
        val = os.environ.get(name)
        if val:
            return val
    raise RuntimeError("no Alpha Vantage API key in env (ALPHAVANTAGE_API_KEY)")


def fetch_month(symbol: str, month: str, key: str) -> pd.DataFrame:
    """Return all 1-minute bars for one YYYY-MM month as a tidy frame."""
    resp = requests.get(
        _URL,
        params={
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": "1min",
            "month": month,
            "outputsize": "full",
            "apikey": key,
        },
        timeout=60,
    )
    payload = resp.json()
    ts_key = next((k for k in payload if "Time Series" in k), None)
    if ts_key is None:
        note = payload.get("Information") or payload.get("Note") or payload.get("Error Message")
        raise RuntimeError(f"AV returned no series for {month}: {note}")
    frame = pd.DataFrame(payload[ts_key]).T.reset_index(names="timestamp_et")
    frame = frame.rename(
        columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume",
        }
    )
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = frame[col].astype(float)
    frame["timestamp_et"] = pd.to_datetime(frame["timestamp_et"])
    return frame


def backfill(
    dates: list[str], out_path: Path, symbol: str = "SPY", pause: float = 13.0
) -> pd.DataFrame:
    """Fetch wide intraday windows for each FOMC date; append to the raw-bar parquet."""
    key = _api_key()
    months = sorted({d[:7] for d in dates})
    by_month: dict[str, pd.DataFrame] = {}
    for i, month in enumerate(months):
        try:
            by_month[month] = fetch_month(symbol, month, key)
            logger.info(
                "fetched %s (%d/%d): %d bars", month, i + 1, len(months), len(by_month[month])
            )
        except Exception as exc:  # noqa: BLE001 - log and continue past a bad month
            logger.warning("skip %s: %s", month, exc)
        time.sleep(pause)

    rows: list[pd.DataFrame] = []
    for date in dates:
        month = date[:7]
        if month not in by_month:
            continue
        day = by_month[month]
        clock = day["timestamp_et"].dt.strftime("%Y-%m-%d %H:%M:%S")
        in_day = day["timestamp_et"].dt.strftime("%Y-%m-%d") == date
        in_win = (day["timestamp_et"].dt.time >= pd.to_datetime(_WINDOW_START).time()) & (
            day["timestamp_et"].dt.time <= pd.to_datetime(_WINDOW_END).time()
        )
        sub = day[in_day & in_win].copy()
        if sub.empty:
            logger.warning("no bars in window for %s", date)
            continue
        sub["event_date"] = date
        sub["symbol"] = symbol
        sub["timestamp_et"] = clock[in_day & in_win].to_numpy()
        rows.append(
            sub[["event_date", "timestamp_et", "open", "high", "low", "close", "volume", "symbol"]]
        )

    new = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if out_path.exists() and not new.empty:
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, new], ignore_index=True)
        combined = combined.drop_duplicates(["event_date", "timestamp_et"], keep="last")
    else:
        combined = new
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    logger.info(
        "wrote %d new bars across %d dates; total events now %d -> %s",
        len(new),
        new["event_date"].nunique() if not new.empty else 0,
        combined["event_date"].nunique() if not combined.empty else 0,
        out_path,
    )
    return combined


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Backfill pre-2013 FOMC intraday bars.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DATA_DIR / "external" / "fed_comms" / "fed_communications.parquet",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "external" / "alphavantage_bars" / "spx_intraday_fomc_days.parquet",
    )
    parser.add_argument("--before", default="2013-01-01")
    parser.add_argument("--pause", type=float, default=13.0)
    args = parser.parse_args()

    corpus = pd.read_parquet(args.corpus)
    statements = corpus[corpus["doc_type"] == "statement"].copy()
    statements["d"] = statements["date"].astype(str).str[:10]
    dates = sorted({d for d in statements["d"] if d < args.before})
    logger.info("backfilling %d pre-%s FOMC statement days", len(dates), args.before)
    backfill(dates, args.out, pause=args.pause)


if __name__ == "__main__":
    main()
