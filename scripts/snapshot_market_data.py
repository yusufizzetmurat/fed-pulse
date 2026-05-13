"""Pull daily close series from FRED (preferred) or yfinance (fallback) and
persist as committed parquet plus a SHA-256 entry in SOURCES.lock.

FRED is the authoritative free source for VIX, treasury yields, USD broad
index, gold AM fix, WTI, fed funds, and the NASDAQ composite. yfinance is
kept only for ^GSPC and ^DJI, where FRED's series are licence-capped at a
10-year rolling window and would not cover a 2010-start benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "data" / "raw" / "market"
DEFAULT_LOCK_PATH = DEFAULT_SNAPSHOT_DIR / "SOURCES.lock"

FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

SOURCE_ROUTES: dict[str, tuple[str, str]] = {
    "^VIX": ("fred", "VIXCLS"),
    "^TNX": ("fred", "DGS10"),
    "^IRX": ("fred", "DGS3MO"),
    "DGS2": ("fred", "DGS2"),
    "DGS10": ("fred", "DGS10"),
    "DTWEXBGS": ("fred", "DTWEXBGS"),
    "DX-Y.NYB": ("fred", "DTWEXBGS"),
    "GC=F": ("fred", "GOLDAMGBD228NLBM"),
    "CL=F": ("fred", "DCOILWTICO"),
    "DFF": ("fred", "DFF"),
    "^IXIC": ("fred", "NASDAQCOM"),
    "NASDAQCOM": ("fred", "NASDAQCOM"),
}

DEFAULT_SYMBOLS = ("^GSPC", "^DJI", "^IXIC", "^VIX", "^TNX", "DX-Y.NYB", "GC=F", "CL=F", "DFF")


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("^", "").replace("=", "_").replace("/", "_").replace(":", "_")


def _hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _route_source(symbol: str) -> tuple[str, str]:
    return SOURCE_ROUTES.get(symbol, ("yfinance", symbol))


def _fetch_from_fred(series_id: str, start: str, end: str):
    import pandas as pd
    import requests

    params = {"id": series_id, "cosd": start, "coed": end}
    response = requests.get(FRED_CSV_BASE, params=params, timeout=30)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text), na_values=["."])
    if frame.shape[1] < 2:
        raise RuntimeError(f"FRED returned unexpected CSV for {series_id}: {response.text[:200]}")
    date_col, value_col = frame.columns[0], frame.columns[1]
    series = pd.Series(
        pd.to_numeric(frame[value_col], errors="coerce").to_numpy(),
        index=pd.to_datetime(frame[date_col]),
        name="Close",
    ).dropna()
    if series.empty:
        raise RuntimeError(f"FRED returned no observations for {series_id} in [{start}, {end}]")
    return series


def _fetch_from_yfinance(symbol: str, start: str, end: str):
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    frame = ticker.history(start=start, end=end, auto_adjust=True)
    if frame.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol} in [{start}, {end})")
    close = frame["Close"]
    if hasattr(close, "columns"):
        if close.shape[1] == 0:
            raise RuntimeError(f"Empty close column for symbol={symbol}")
        close = close.iloc[:, 0]
    return close.dropna()


def _fetch_close_series(symbol: str, start: str, end: str):
    source, source_symbol = _route_source(symbol)
    if source == "fred":
        return _fetch_from_fred(source_symbol, start, end)
    return _fetch_from_yfinance(source_symbol, start, end)


def _write_parquet(series, out_path: Path, symbol: str) -> int:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "symbol": symbol,
            "date": [idx.date().isoformat() for idx in series.index],
            "close": [float(value) for value in series.values],
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    return len(frame)


def _load_lock(lock_path: Path) -> dict:
    if not lock_path.exists():
        return {"format_version": 1, "entries": {}}
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"format_version": 1, "entries": {}}
    payload.setdefault("format_version", 1)
    payload.setdefault("entries", {})
    return payload


def _write_lock(lock_path: Path, payload: dict) -> None:
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def snapshot(
    *,
    symbols: Sequence[str],
    start: str,
    end: str,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
    lock_path: Path = DEFAULT_LOCK_PATH,
) -> list[dict]:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    lock = _load_lock(lock_path)
    results: list[dict] = []
    fetched_at = datetime.now(timezone.utc).isoformat()

    for symbol in symbols:
        source, source_symbol = _route_source(symbol)
        series = _fetch_close_series(symbol, start, end)
        out_path = snapshot_dir / f"{_safe_symbol(symbol)}.parquet"
        rows = _write_parquet(series, out_path, symbol)
        sha = _hash_file(out_path)
        try:
            stored_path = str(out_path.relative_to(REPO_ROOT))
        except ValueError:
            stored_path = str(out_path)
        lock["entries"][symbol] = {
            "parquet_path": stored_path,
            "sha256": sha,
            "rows": rows,
            "start": start,
            "end": end,
            "fetched_at": fetched_at,
            "source": source,
            "source_symbol": source_symbol,
        }
        results.append(
            {
                "symbol": symbol,
                "rows": rows,
                "sha256": sha,
                "path": str(out_path),
                "source": source,
                "source_symbol": source_symbol,
            }
        )
        print(f"[snapshot] {symbol} via {source}:{source_symbol}: {rows} rows -> {out_path} (sha={sha[:12]}...)")

    _write_lock(lock_path, lock)
    print(f"[snapshot] wrote {lock_path}")
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull daily close series (FRED preferred, yfinance fallback) into committed parquet snapshots."
    )
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT_DIR))
    parser.add_argument("--lock-path", default=str(DEFAULT_LOCK_PATH))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    snapshot(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        snapshot_dir=Path(args.snapshot_dir),
        lock_path=Path(args.lock_path),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
