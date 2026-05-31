"""Align the typed Fed communication corpus with the intraday realized-vol series.

Produces the two leak-safe tables the gated text↔market fusion model consumes:

  1. text↔outcome pairs (communication-level) — each communication paired with
     the forward realized variance that follows it, starting the first trading
     day strictly AFTER the text date. Feeds the InfoNCE contrastive objective
     and the supervised text head.
  2. daily fusion frame (trading-day-level) — HAR/market features at the forecast
     origin, the target forward RV, and a reference to the most-recent
     communication known by that origin (with its age in trading days and type).
     Feeds the gated forecaster, whose gate can discount stale/irrelevant text.

Embargo rule (uniform across text types, so date-only speeches are safe): a
communication dated D contributes only to windows that begin on the first
trading day > D, and a forecast origin D may use only communications dated ≤ D.
The forward RV target itself spans t+1..t+h, so no target overlaps its features.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from app.config import DATA_DIR
from app.data.intraday_realized import DEFAULT_RV_PARQUET
from app.data.intraday_rv_forecast import _EPS, _har_lags
from app.data.fed_comms_scrape import DEFAULT_CORPUS_PARQUET

DEFAULT_OUT_DIR = DATA_DIR / "processed" / "fed_comms_fusion"
DEFAULT_HORIZONS = (1, 5, 22)
_DOC_TYPES = ("statement", "minutes", "press_conference", "speech", "testimony")


def _forward_log_rv_windows(rv: np.ndarray, horizons: tuple[int, ...]) -> dict[int, np.ndarray]:
    """For each horizon h: log mean RV over t+1..t+h (NaN where it runs off)."""

    n = len(rv)
    out: dict[int, np.ndarray] = {}
    for h in horizons:
        col = np.full(n, np.nan)
        for t in range(n - h):
            col[t] = np.log(rv[t + 1 : t + 1 + h].mean() + _EPS)
        out[h] = col
    return out


def _origin_after(date_iso: str, trading_days: list[str]) -> int | None:
    """Index of the first trading day strictly greater than date_iso (embargo)."""

    import bisect

    i = bisect.bisect_right(trading_days, date_iso)
    return i if i < len(trading_days) else None


def _as_of_index(date_iso: str, trading_days: list[str]) -> int | None:
    """Index of the latest trading day ≤ date_iso (most-recent-known origin)."""

    import bisect

    i = bisect.bisect_right(trading_days, date_iso) - 1
    return i if i >= 0 else None


def build_text_outcome_pairs(
    corpus: Any, rv_df: Any, *, horizons: tuple[int, ...] = DEFAULT_HORIZONS
) -> Any:
    """Communication-level table: text + forward RV outcome (origin = day after text)."""

    import pandas as pd

    rv_df = rv_df.sort_values("date").reset_index(drop=True)
    trading_days = rv_df["date"].astype(str).tolist()
    rv = rv_df["rv"].to_numpy(dtype=np.float64)
    fwd = _forward_log_rv_windows(rv, horizons)

    rows: list[dict[str, Any]] = []
    for _, doc in corpus.sort_values("timestamp_et").iterrows():
        origin = _origin_after(str(doc["date"]), trading_days)
        if origin is None:
            continue
        out: dict[str, Any] = {
            "date": doc["date"],
            "origin_date": trading_days[origin],
            "doc_type": doc["doc_type"],
            "time_known": bool(doc["time_known"]),
            "speaker": doc.get("speaker"),
            "text": doc["text"],
        }
        valid = False
        for h in horizons:
            val = fwd[h][origin]
            out[f"rv_fwd_{h}"] = float(val) if np.isfinite(val) else np.nan
            valid = valid or np.isfinite(val)
        if valid:
            rows.append(out)
    return pd.DataFrame(rows)


def build_daily_fusion_frame(
    rv_df: Any, corpus: Any, *, horizons: tuple[int, ...] = DEFAULT_HORIZONS
) -> Any:
    """Trading-day table: HAR target + reference to most-recent known communication."""

    import pandas as pd

    rv_df = rv_df.sort_values("date").reset_index(drop=True)
    trading_days = rv_df["date"].astype(str).tolist()
    rv = rv_df["rv"].to_numpy(dtype=np.float64)
    log_rv = np.log(rv + _EPS)
    har = _har_lags(log_rv)  # [daily, weekly, monthly] log-RV lags at each day
    fwd = _forward_log_rv_windows(rv, horizons)

    # most-recent communication known as of each trading day (any type)
    corpus_sorted = corpus.sort_values("date").reset_index(drop=True)
    comm_dates = corpus_sorted["date"].astype(str).tolist()
    last_doc_row = np.full(len(trading_days), -1, dtype=int)
    j = 0
    for i, day in enumerate(trading_days):
        while j < len(comm_dates) and comm_dates[j] <= day:
            j += 1
        last_doc_row[i] = j - 1  # index into corpus_sorted, or -1 if none yet

    rows: list[dict[str, Any]] = []
    for i, day in enumerate(trading_days):
        row: dict[str, Any] = {
            "date": day,
            "har_daily": float(har[i, 0]),
            "har_weekly": float(har[i, 1]),
            "har_monthly": float(har[i, 2]),
        }
        di = int(last_doc_row[i])
        if di >= 0:
            doc = corpus_sorted.iloc[di]
            origin = _as_of_index(str(doc["date"]), trading_days)
            row["doc_row"] = di
            row["doc_type"] = str(doc["doc_type"])
            row["doc_age_days"] = i - origin if origin is not None else -1
            row["has_text"] = True
        else:
            row["doc_row"] = -1
            row["doc_type"] = None
            row["doc_age_days"] = -1
            row["has_text"] = False
        valid = False
        for h in horizons:
            val = fwd[h][i]
            row[f"rv_fwd_{h}"] = float(val) if np.isfinite(val) else np.nan
            valid = valid or np.isfinite(val)
        if valid:
            rows.append(row)
    return pd.DataFrame(rows)


def build(
    *,
    corpus_path: Path | str = DEFAULT_CORPUS_PARQUET,
    rv_path: Path | str = DEFAULT_RV_PARQUET,
    out_dir: Path | str = DEFAULT_OUT_DIR,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> tuple[Path, Path]:
    """Build + persist both tables; return (pairs_path, daily_path)."""

    import pandas as pd

    corpus = pd.read_parquet(corpus_path)
    rv_df = pd.read_parquet(rv_path)
    pairs = build_text_outcome_pairs(corpus, rv_df, horizons=horizons)
    daily = build_daily_fusion_frame(rv_df, corpus, horizons=horizons)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "text_outcome_pairs.parquet"
    daily_path = out_dir / "daily_fusion.parquet"
    pairs.to_parquet(pairs_path, index=False)
    daily.to_parquet(daily_path, index=False)
    by_type = pairs["doc_type"].value_counts().to_dict() if not pairs.empty else {}
    cov = float(daily["has_text"].mean()) if not daily.empty else 0.0
    print(f"[fed_comms_dataset] pairs={len(pairs)} by_type={by_type}")
    print(f"[fed_comms_dataset] daily={len(daily)} has_text_frac={cov:.3f} → {out_dir}")
    return pairs_path, daily_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Align Fed comms corpus with intraday RV.")
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PARQUET)
    parser.add_argument("--rv-path", type=Path, default=DEFAULT_RV_PARQUET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    build(corpus_path=args.corpus_path, rv_path=args.rv_path, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
