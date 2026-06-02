"""Build the intraday-pivot event dataset from cached bars + statements.

Reads the provider-agnostic bar cache (``polygon_spx.load_intraday_bars``)
and a training package ``events.parquet``; emits one row per FOMC
announcement with the pre-announcement (13:30-14:00 ET) bar sequence and
the immediate (14:00-14:30) + delayed (14:30-15:00) reaction targets.

Feature engineering is deferred to the Phase 3 loader; this module fixes
the data contract and enforces the no-leakage rule (no target-window bar
ever enters the feature sequence).
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path
from typing import Any, Sequence

from app.data.polygon_spx import DEFAULT_CACHE_DIR, PolygonBar, load_intraday_bars

ANNOUNCEMENT = "14:00:00"
PRE_START = "13:30:00"
IMM_END = "14:30:00"
DELAYED_END = "15:00:00"
DEFAULT_EVENT_KIND = "statement"


def _anchor_close(bars: Sequence[PolygonBar], timestamp_et: str) -> float | None:
    """Exact-minute close lookup; ``None`` if that minute is absent."""

    for bar in bars:
        if bar.timestamp_et == timestamp_et:
            return float(bar.close)
    return None


def _compute_target(pre_close: float, post_close: float) -> dict[str, Any]:
    ret = post_close / pre_close - 1.0
    return {"ret": ret, "dir": 1 if ret > 0 else 0, "mag": abs(ret)}


def _assert_no_leakage(pre_bars: Sequence[PolygonBar]) -> None:
    """No pre-window bar may carry a timestamp strictly after the 14:00 release."""

    for bar in pre_bars:
        minute = bar.timestamp_et[-8:]
        assert minute <= ANNOUNCEMENT, f"target-window bar leaked into features: {bar.timestamp_et}"


def _pre_window_bars(bars: Sequence[PolygonBar], event_date: datetime.date) -> list[PolygonBar]:
    d = event_date.isoformat()
    lo, hi = f"{d} {PRE_START}", f"{d} {ANNOUNCEMENT}"
    pre = [b for b in bars if lo <= b.timestamp_et <= hi]
    _assert_no_leakage(pre)
    return pre


def statement_text_by_date(
    events_parquet: Path | str, *, event_kind: str = DEFAULT_EVENT_KIND
) -> dict[str, str]:
    """Map ``event_date_iso -> canonical statement text`` (one per date).

    The ~4 statement rows per date are identical text differing only by
    forecast ``horizon``; collapse to one row per date.
    """

    import pandas as pd

    frame = pd.read_parquet(events_parquet)
    frame = frame[frame["event_kind"].astype(str).str.lower() == event_kind]
    frame = frame.assign(_d=frame["event_date"].astype(str).str[:10])
    deduped = frame.drop_duplicates(subset="_d", keep="first")
    return {str(r["_d"]): str(r["text"]) for _, r in deduped.iterrows()}


def _build_event_row(
    date_iso: str, bars: Sequence[PolygonBar], text: str, *, symbol: str
) -> dict[str, Any] | None:
    event_date = datetime.date.fromisoformat(date_iso)
    pre = _pre_window_bars(bars, event_date)
    c1400 = _anchor_close(bars, f"{date_iso} {ANNOUNCEMENT}")
    c1430 = _anchor_close(bars, f"{date_iso} {IMM_END}")
    c1500 = _anchor_close(bars, f"{date_iso} {DELAYED_END}")
    if not pre or c1400 is None or c1430 is None or c1500 is None or c1400 <= 0 or c1430 <= 0:
        return None
    imm = _compute_target(c1400, c1430)
    dly = _compute_target(c1430, c1500)
    return {
        "event_date": date_iso,
        "text": text,
        "pre_close": [float(b.close) for b in pre],
        "pre_volume": [float(b.volume) for b in pre],
        "n_pre_bars": len(pre),
        "close_1400": float(c1400),
        "close_1430": float(c1430),
        "close_1500": float(c1500),
        "ret_immediate": imm["ret"],
        "dir_immediate": imm["dir"],
        "mag_immediate": imm["mag"],
        "ret_delayed": dly["ret"],
        "dir_delayed": dly["dir"],
        "mag_delayed": dly["mag"],
        "symbol": symbol,
    }


def build_dataset(
    bars_by_date: dict[str, list[PolygonBar]],
    text_by_date: dict[str, str],
    *,
    out_path: Path | str,
    symbol: str = "SPY",
) -> int:
    """Join bars + statements, build rows, write parquet. Returns row count."""

    import pandas as pd

    built_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[dict[str, Any]] = []
    only_text = sorted(set(text_by_date) - set(bars_by_date))
    only_bars = sorted(set(bars_by_date) - set(text_by_date))
    for date_iso in sorted(set(bars_by_date) & set(text_by_date)):
        row = _build_event_row(
            date_iso, bars_by_date[date_iso], text_by_date[date_iso], symbol=symbol
        )
        if row is not None:
            row["built_at_utc"] = built_at
            rows.append(row)
    frame = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    print(
        f"[intraday_event_builder] wrote {len(frame)} events to {out_path} "
        f"({len(only_text)} statement-date(s) without bars, {len(only_bars)} bar-date(s) "
        f"without statements)"
    )
    return len(frame)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build intraday_events.parquet from cached bars + a package events.parquet."
    )
    parser.add_argument("--events-parquet", type=Path, required=True)
    parser.add_argument("--bars-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--symbol", default="SPY")
    args = parser.parse_args()
    bars_by_date = load_intraday_bars(args.bars_cache_dir)
    text_by_date = statement_text_by_date(args.events_parquet)
    build_dataset(bars_by_date, text_by_date, out_path=args.out, symbol=str(args.symbol))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
