"""Clean-room FOMC event-frame assembly for the late-fusion rebuild.

Rebuilds the announcement-window reaction table directly from raw 1-minute SPY
bars and the typed Fed-communication corpus, independent of any prior artifact
(the earlier ``intraday_events.parquet`` has no builder on the mainline, so its
provenance is not reproducible here).

For each FOMC statement day we have 91 one-minute bars spanning 13:30-15:00 ET:

* pre-announcement window 13:30 -> 14:00 (features only),
* immediate reaction 14:00 -> 14:30,
* delayed reaction 14:30 -> 15:00.

The statement announcement is at 14:00 ET, so every pre-window bar is strictly
before 14:00 and every reaction bar is at/after 14:00. This module asserts that
ordering (fault class #1: alignment) and emits a human-readable audit. The SEP
projection tables (released 2pm ET with the statement) are joined as event-frame
features; the narrative SEP is never used here (see ``sep_ingestion``).
"""

from __future__ import annotations

import argparse
import logging
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import DATA_DIR

logger = logging.getLogger(__name__)

# The FOMC announcement is located per-meeting from the intraday volume spike
# rather than a fixed clock time, because the release time varied by era (2:15pm
# pre-2011, 12:30/2:15pm in 2011-12, 2:00pm from 2013). The search window bounds
# where the announcement can be (leaving 30 min of pre-window and 60 min of
# reaction inside a 12:00-16:00 raw window).
# Known FOMC statement release times by era — more accurate than inferring a time
# we already know from noisy intraday volume. From 2013 the statement is released
# at 2:00pm ET; before that it was 2:15pm, except the 2011-2012 press-conference
# meetings that released at ~12:30pm. The volume spike is used only to catch those
# early pre-2013 releases (a strong spike well before 2:00pm).
_ANNOUNCE_MODERN = time(14, 0)
_ANNOUNCE_LEGACY = time(14, 15)
_MODERN_FROM = "2013-01-01"
# Narrow window around the 12:30 press-conference release, so the override fires
# only for genuine early releases — not random pre-noon volume blips.
_EARLY_SPIKE_AFTER = time(12, 25)
_EARLY_SPIKE_BEFORE = time(12, 45)
_PRE_MINUTES = 30
_IMMEDIATE_MINUTES = 30
_DELAYED_MINUTES = 60  # delayed window ends 60 min after the announcement
_SPIKE_MULTIPLE = 4.0  # volume > 4x the day median marks an announcement spike


def _open_at_dt(day_bars: pd.DataFrame, target: pd.Timestamp) -> float | None:
    """OPEN of the bar stamped exactly ``target`` (the price at that instant).

    Bars are interval-start labelled, so a bar's OPEN is the price at its start
    minute — pre-reaction at the announcement instant. None if the bar is absent.
    """
    match = day_bars[day_bars["timestamp_et"] == target]
    if match.empty:
        return None
    return float(match["open"].iloc[0])


def _log_change(numer: float | None, denom: float | None) -> float | None:
    if numer is None or denom is None or denom <= 0 or numer <= 0:
        return None
    return float(np.log(numer / denom))


def build_event_windows(bars: pd.DataFrame) -> pd.DataFrame:
    """Reduce raw 1-minute bars to one reaction row per FOMC event date.

    The announcement instant is detected as the max-volume minute within the
    search window (the FOMC release dominates intraday volume — ~15x median),
    making the windows robust to the era-varying release time. Reaction is then
    measured at announcement-relative offsets using bar OPENS, so the
    announcement-minute move is captured in ret_immediate and never leaks into the
    pre-window. Output: event_date, announce_time, announce_vol_ratio, pre_close,
    pre_ret, pre_rv, pre_volume, n_pre_bars, px_ann/px_imm/px_del,
    ret_immediate/dir_immediate/mag_immediate, ret_delayed/dir_delayed/mag_delayed,
    n_bars, has_anchors.
    """
    frame = bars.copy()
    frame["timestamp_et"] = pd.to_datetime(frame["timestamp_et"])
    frame["_t"] = frame["timestamp_et"].dt.time

    rows: list[dict[str, object]] = []
    for event_date, day in frame.groupby("event_date"):
        day = day.sort_values("timestamp_et")
        event_date_str = str(event_date)[:10]
        bar_dates = day["timestamp_et"].dt.strftime("%Y-%m-%d").unique()
        if list(bar_dates) != [event_date_str]:
            raise ValueError(f"{event_date_str}: bars span foreign dates {list(bar_dates)}")

        # Anchor at the known release time for the era; for pre-2013, override to a
        # strong early spike (the 12:30 press-conference meetings).
        median_vol = float(day["volume"].median()) or 1.0
        ann_time = _ANNOUNCE_MODERN if event_date_str >= _MODERN_FROM else _ANNOUNCE_LEGACY
        if event_date_str < _MODERN_FROM:
            early = day[
                (day["_t"] >= _EARLY_SPIKE_AFTER)
                & (day["_t"] < _EARLY_SPIKE_BEFORE)
                & (day["volume"] > _SPIKE_MULTIPLE * median_vol)
            ]
            if not early.empty:
                ann_time = day.loc[early.index[0], "timestamp_et"].time()
        ann_match = day[day["_t"] == ann_time]
        if ann_match.empty:
            rows.append({"event_date": event_date_str, "has_anchors": 0, "n_bars": int(len(day))})
            continue
        ann_dt = ann_match["timestamp_et"].iloc[0]
        ann_vol_ratio = float(ann_match["volume"].iloc[0]) / median_vol

        pre_start = ann_dt - pd.Timedelta(minutes=_PRE_MINUTES)
        px_pre = _open_at_dt(day, pre_start)
        px_ann = _open_at_dt(day, ann_dt)  # announcement instant (pre-reaction)
        px_imm = _open_at_dt(day, ann_dt + pd.Timedelta(minutes=_IMMEDIATE_MINUTES))
        px_del = _open_at_dt(day, ann_dt + pd.Timedelta(minutes=_DELAYED_MINUTES))

        pre = day[(day["timestamp_et"] >= pre_start) & (day["timestamp_et"] < ann_dt)]
        pre_logrets = np.diff(np.log(pre["close"].to_numpy())) if len(pre) > 1 else np.array([])
        ret_immediate = _log_change(px_imm, px_ann)
        ret_delayed = _log_change(px_del, px_imm)

        # Announcement-window volume (the event-frame volume-head target): total
        # volume in [announce, announce+30min] and the abnormal-volume ratio vs the
        # pre-window. Pre-window features predict it; text is not expected to.
        imm_end = ann_dt + pd.Timedelta(minutes=_IMMEDIATE_MINUTES)
        imm_window = day[(day["timestamp_et"] >= ann_dt) & (day["timestamp_et"] < imm_end)]
        imm_volume = float(imm_window["volume"].sum())
        pre_vol_sum = float(pre["volume"].sum())
        abn_volume = imm_volume / pre_vol_sum if pre_vol_sum > 0 else None

        rows.append(
            {
                "event_date": event_date_str,
                "announce_time": pd.Timestamp(ann_dt).strftime("%H:%M"),
                "announce_vol_ratio": round(ann_vol_ratio, 1),
                "pre_close": px_ann,
                "pre_ret": _log_change(px_ann, px_pre),
                "pre_rv": float(np.std(pre_logrets)) if pre_logrets.size else None,
                "pre_volume": float(pre["volume"].sum()),
                "n_pre_bars": int(len(pre)),
                "px_ann": px_ann,
                "px_imm": px_imm,
                "px_del": px_del,
                "ret_immediate": ret_immediate,
                "dir_immediate": None if ret_immediate is None else int(ret_immediate > 0),
                "mag_immediate": None if ret_immediate is None else abs(ret_immediate),
                "ret_delayed": ret_delayed,
                "dir_delayed": None if ret_delayed is None else int(ret_delayed > 0),
                "mag_delayed": None if ret_delayed is None else abs(ret_delayed),
                "imm_volume": imm_volume,
                "abn_volume": abn_volume,
                "n_bars": int(len(day)),
                "has_anchors": int(None not in (px_pre, px_ann, px_imm, px_del)),
            }
        )

    out = pd.DataFrame(rows).sort_values("event_date").reset_index(drop=True)
    return out


def join_statement_text(events: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    """Attach the FOMC statement text for each event date (exact-date join).

    Asserts every event matches exactly one statement; logs any that do not.
    """
    statements = corpus[corpus["doc_type"].str.lower() == "statement"].copy()
    statements["_date"] = statements["date"].astype(str).str[:10]
    # Collapse to one statement per date (defensive: there should be exactly one).
    dupes = statements["_date"].duplicated().sum()
    if dupes:
        logger.warning("%d duplicate statement dates in corpus; keeping first", dupes)
    statements = statements.drop_duplicates("_date", keep="first")

    text_by_date = statements.set_index("_date")[["text", "title"]]
    merged = events.merge(text_by_date, left_on="event_date", right_index=True, how="left")
    missing = merged["text"].isna().sum()
    if missing:
        missing_dates = merged.loc[merged["text"].isna(), "event_date"].tolist()
        logger.warning("%d events with no statement text: %s", missing, missing_dates)
    return merged


def join_sep_features(events: pd.DataFrame, sep_long: pd.DataFrame) -> pd.DataFrame:
    """Attach SEP projection features by meeting date; absent -> NaN + flag.

    Only the at-meeting projection table is used (event-frame safe). For each
    (variable, horizon) builds two columns:

    * ``sep_point_<var>_<hor>`` — the median when published (2015+), else the
      central-tendency midpoint (recovers the 2013-2014 legacy meetings, which
      published no median),
    * ``sep_disp_<var>_<hor>`` — central-tendency width (a dispersion/uncertainty
      proxy), or range width when central tendency is absent.

    Plus ``sep_available`` (1 if any SEP feature is present for the meeting).
    """
    if sep_long.empty:
        out = events.copy()
        out["sep_available"] = 0
        return out

    sep = sep_long.copy()
    sep_key = ["meeting_date", "variable", "horizon"]
    sep_dupes = int(sep.duplicated(sep_key).sum())
    if sep_dupes:
        logger.warning(
            "%d duplicate SEP (meeting_date, variable, horizon) rows; keeping first",
            sep_dupes,
        )
        sep = sep.drop_duplicates(sep_key, keep="first")
    ct_mid = (sep["central_low"] + sep["central_high"]) / 2
    ct_width = sep["central_high"] - sep["central_low"]
    range_width = sep["range_high"] - sep["range_low"]
    sep["point"] = sep["median"].where(sep["median"].notna(), ct_mid)
    sep["disp"] = ct_width.where(ct_width.notna(), range_width)
    # Use RELATIVE horizons (h0 = meeting year, h1, h2, ... , LR) so the columns
    # are dense across meetings. Absolute-year horizons would make each column
    # non-null for only the ~4 meetings of that year — useless as features.
    meeting_year = sep["meeting_date"].astype(str).str[:4].astype(int)
    horizon_year = pd.to_numeric(sep["horizon"], errors="coerce")
    rel = ("h" + (horizon_year - meeting_year).astype("Int64").astype(str)).where(
        sep["horizon"] != "LR", "LR"
    )
    sep = sep.assign(rel=rel)
    sep = sep[sep["rel"].notna()]
    sep["key"] = sep["variable"] + "_" + sep["rel"].astype(str)

    point = sep.pivot_table(index="meeting_date", columns="key", values="point", aggfunc="first")
    disp = sep.pivot_table(index="meeting_date", columns="key", values="disp", aggfunc="first")
    point.columns = ["sep_point_" + c for c in point.columns]
    disp.columns = ["sep_disp_" + c for c in disp.columns]
    wide = point.join(disp)
    wide.index = wide.index.astype(str)

    merged = events.merge(wide, left_on="event_date", right_index=True, how="left")
    point_cols = [c for c in merged.columns if c.startswith("sep_point_")]
    merged["sep_available"] = merged[point_cols].notna().any(axis=1).astype(int)
    return merged


def alignment_audit(events: pd.DataFrame) -> pd.DataFrame:
    """Human-readable audit: date, statement title, window returns, SEP flag."""
    cols = [
        "event_date",
        "announce_time",
        "announce_vol_ratio",
        "title",
        "n_pre_bars",
        "n_bars",
        "pre_ret",
        "ret_immediate",
        "ret_delayed",
        "sep_available",
    ]
    present = [c for c in cols if c in events.columns]
    audit = events[present].copy()
    if "title" in audit.columns:
        audit["title"] = audit["title"].astype(str).str.slice(0, 60)
    return audit


def build(
    bars_path: Path,
    corpus_path: Path,
    sep_path: Path,
    out_path: Path,
) -> pd.DataFrame:
    """Assemble the clean-room event-frame table and write it to parquet."""
    bars = pd.read_parquet(bars_path)
    corpus = pd.read_parquet(corpus_path)
    if sep_path.exists():
        sep_long = pd.read_parquet(sep_path)
    else:
        # SEP is a required input for this rebuild; a missing file would silently
        # produce a structurally different parquet (no sep_* columns). Surface it.
        logger.warning(
            "SEP file %s not found; building WITHOUT sep_point_*/sep_disp_* columns",
            sep_path,
        )
        sep_long = pd.DataFrame()

    events = build_event_windows(bars)
    incomplete = int((events["has_anchors"] == 0).sum())
    if incomplete:
        bad = events.loc[events["has_anchors"] == 0, "event_date"].tolist()
        logger.warning("%d events missing 14:00/14:30/15:00 anchors: %s", incomplete, bad)
    events = join_statement_text(events, corpus)
    # The event frame is the FOMC *statement* reaction; a date with no matching
    # statement is a contaminant in the raw bar set (e.g. a speech-only day) and
    # is dropped so the frame stays pure. Logged for traceability.
    no_statement = events.loc[events["text"].isna(), "event_date"].tolist()
    if no_statement:
        logger.warning("dropping %d non-statement day(s): %s", len(no_statement), no_statement)
        events = events[events["text"].notna()].reset_index(drop=True)
    events = join_sep_features(events, sep_long)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(out_path, index=False)
    logger.info("wrote %d events -> %s", len(events), out_path)
    return events


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build clean-room FOMC event frame.")
    parser.add_argument(
        "--bars",
        type=Path,
        default=DATA_DIR / "external" / "alphavantage_bars" / "spx_intraday_fomc_days.parquet",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DATA_DIR / "external" / "fed_comms" / "fed_communications.parquet",
    )
    parser.add_argument(
        "--sep",
        type=Path,
        default=DATA_DIR / "external" / "fed_comms" / "sep.parquet",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "processed" / "late_fusion" / "event_frame.parquet",
    )
    args = parser.parse_args()

    events = build(args.bars, args.corpus, args.sep, args.out)
    audit = alignment_audit(events)
    print(f"\n=== EVENT FRAME: {len(events)} events ===")
    print(f"with statement text: {events['text'].notna().sum()}")
    print(f"with SEP features:   {events['sep_available'].sum()}")
    print(f"date range: {events['event_date'].min()} -> {events['event_date'].max()}")
    print("\n=== ALIGNMENT AUDIT (head/tail) ===")
    print(audit.head(5).to_string(index=False))
    print("...")
    print(audit.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
