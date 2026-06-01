"""Clean-room daily companion frame for the late-fusion rebuild.

The event frame (``late_fusion_events``) is the precise but small FOMC
statement-reaction set (n~109). This module builds the high-power companion:
one row per typed Fed communication in the full corpus (~2000 docs, 2006-2026),
with a leak-safe daily market reaction target. Its purpose is statistical power
to settle whether text carries *any* signal, not intraday precision.

Target definition (leak-safe). A communication dated D is public by the close of
the first trading day t0 >= D. We predict the **next-day** return
``log(close(t1) / close(t0))`` (t1 the next trading day) — text and all market
features are known by close(t0), the target is strictly forward. This is the
Fed-information / post-communication drift target. Same-day intraday reaction is
NOT captured here for non-FOMC comms (no intraday bars); that precision lives in
the event frame. ``ret_anchor_day`` (the t-1 -> t0 move) is recorded for
reference only and must not be used as a target with t0-dated features.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import DATA_DIR

logger = logging.getLogger(__name__)


def _load_daily_close(path: Path) -> pd.DataFrame:
    """Load the daily SPX close series, sorted, one row per trading day."""
    daily = pd.read_parquet(path)
    if "symbol" in daily.columns and daily["symbol"].nunique() > 1:
        top = daily["symbol"].value_counts().idxmax()
        daily = daily[daily["symbol"] == top]
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"].astype(str).str[:10])
    daily = daily.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return daily[["date", "close"]]


def build_daily_frame(
    corpus: pd.DataFrame,
    daily_close: pd.DataFrame,
    daily_rv: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One leak-safe row per communication with a next-day reaction target.

    Columns: comm_date, anchor_date (t0), doc_type, speaker, title, text,
    ret_5d, ret_22d, rv_22 (features as of t0), ret_anchor_day (reference),
    ret_nextday/dir_nextday/mag_nextday (target t0->t1), fwd_rv (RV at t1).
    """
    closes = daily_close["close"].to_numpy()
    log_close = np.log(closes)
    trade_dates = daily_close["date"].to_numpy()
    n = len(trade_dates)

    rv_by_date: dict[pd.Timestamp, float] = {}
    if daily_rv is not None and not daily_rv.empty:
        rv = daily_rv.copy()
        rv["date"] = pd.to_datetime(rv["date"].astype(str).str[:10])
        rv_by_date = dict(zip(rv["date"], rv["rv"], strict=False))

    rows: list[dict[str, object]] = []
    skipped = 0
    for _, doc in corpus.iterrows():
        comm_date = pd.to_datetime(str(doc["date"])[:10])
        # t0 = first trading day on/after the communication date (comm public by
        # its close); searchsorted on the sorted trade-date array.
        t0 = int(np.searchsorted(trade_dates, np.datetime64(comm_date), side="left"))
        if t0 >= n - 1 or t0 < 22:
            # need a next trading day for the target and >=22 days of history
            skipped += 1
            continue
        t1 = t0 + 1

        ret_nextday = float(log_close[t1] - log_close[t0])
        t1_ts = pd.Timestamp(trade_dates[t1])  # RV lookup target is on t1, not t0
        # Stable content-based id so embeddings join by key, not by row position.
        # Full text + title in the digest so near-duplicate openings stay distinct.
        text_val = str(doc["text"])
        row_hash = hashlib.sha256(
            f"{comm_date.date()}|{doc['doc_type']}|{doc.get('title')}|{text_val}".encode()
        ).hexdigest()[:16]
        rows.append(
            {
                "row_hash": row_hash,
                "comm_date": comm_date.strftime("%Y-%m-%d"),
                "anchor_date": pd.Timestamp(trade_dates[t0]).strftime("%Y-%m-%d"),
                "doc_type": doc["doc_type"],
                "speaker": doc.get("speaker"),
                "title": doc.get("title"),
                "text": doc["text"],
                # features as of close(t0) — strictly known before the target window
                "ret_5d": float(log_close[t0] - log_close[t0 - 5]),
                "ret_22d": float(log_close[t0] - log_close[t0 - 22]),
                "rv_22": float(np.std(np.diff(log_close[t0 - 22 : t0 + 1]))),
                "ret_anchor_day": float(log_close[t0] - log_close[t0 - 1]),
                # target: strictly forward t0 -> t1
                "ret_nextday": ret_nextday,
                "dir_nextday": int(ret_nextday > 0),
                "mag_nextday": abs(ret_nextday),
                "fwd_rv": rv_by_date.get(t1_ts),  # target: RV on t1
            }
        )

    if skipped:
        logger.info("skipped %d comms (insufficient history or no next-day close)", skipped)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    dup = int(out["row_hash"].duplicated().sum())
    if dup:
        logger.warning("%d duplicate row_hash values (non-unique join key)", dup)
    return out.sort_values(["comm_date", "doc_type"]).reset_index(drop=True)


def build(
    corpus_path: Path,
    daily_close_path: Path,
    daily_rv_path: Path,
    out_path: Path,
) -> pd.DataFrame:
    """Assemble the daily companion frame and write it to parquet."""
    corpus = pd.read_parquet(corpus_path)
    daily_close = _load_daily_close(daily_close_path)
    daily_rv = pd.read_parquet(daily_rv_path) if daily_rv_path.exists() else None
    if daily_rv is None:
        logger.warning("daily RV %s absent; fwd_rv will be null", daily_rv_path)

    frame = build_daily_frame(corpus, daily_close, daily_rv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    logger.info("wrote %d daily-frame rows -> %s", len(frame), out_path)
    return frame


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build clean-room daily companion frame.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DATA_DIR / "external" / "fed_comms" / "fed_communications.parquet",
    )
    parser.add_argument(
        "--daily-close",
        type=Path,
        default=DATA_DIR
        / "processed"
        / "tp_v3_full_rebuild_2026_05_30"
        / "_market_cache"
        / "GSPC.parquet",
    )
    parser.add_argument(
        "--daily-rv",
        type=Path,
        default=DATA_DIR / "external" / "alphavantage_bars" / "spx_5min_daily_rv.parquet",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "processed" / "late_fusion" / "daily_frame.parquet",
    )
    args = parser.parse_args()

    frame = build(args.corpus, args.daily_close, args.daily_rv, args.out)
    print(f"\n=== DAILY FRAME: {len(frame)} rows ===")
    print(f"date range: {frame['comm_date'].min()} -> {frame['comm_date'].max()}")
    print("by doc_type:")
    print(frame.groupby("doc_type").size().to_string())
    print(f"dir_nextday balance: {dict(frame['dir_nextday'].value_counts())}")
    print(f"with fwd_rv: {frame['fwd_rv'].notna().sum()}")


if __name__ == "__main__":
    main()
