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

_ANNOUNCE = time(14, 0)
_IMMEDIATE_END = time(14, 30)
_DELAYED_END = time(15, 0)
_PRE_START = time(13, 30)


def _open_at(day_bars: pd.DataFrame, clock: time) -> float | None:
    """Instantaneous mark price at ``clock`` — the OPEN of the bar stamped ``clock``.

    AlphaVantage labels 1-minute bars by interval START (verified empirically: on
    2024-09-18 the announcement jump lands entirely inside the bar stamped 14:00,
    on 15x volume, while the 13:59 bar is quiescent). So the OPEN of the 14:00 bar
    is the price at 14:00:00 — the announcement instant, before any reaction — and
    the CLOSE of the 14:00 bar already contains the first reaction minute. Using
    opens as marks keeps pre-announcement features strictly pre-reaction and lets
    the reaction window capture the full announcement-minute move. Returns None if
    the bar is absent.
    """
    match = day_bars[day_bars["_t"] == clock]
    if match.empty:
        return None
    return float(match["open"].iloc[0])


def _log_change(numer: float | None, denom: float | None) -> float | None:
    if numer is None or denom is None or denom <= 0 or numer <= 0:
        return None
    return float(np.log(numer / denom))


def build_event_windows(bars: pd.DataFrame) -> pd.DataFrame:
    """Reduce raw 1-minute bars to one reaction row per FOMC event date.

    Output columns: event_date, pre_close, pre_ret, pre_rv, pre_volume, n_pre_bars,
    px_1400, px_1430, px_1500, ret_immediate/dir_immediate/mag_immediate,
    ret_delayed/dir_delayed/mag_delayed, n_bars, has_anchors.

    Mark prices are bar OPENS (instantaneous prices at the mark instant); see
    ``_open_at``. The reaction is measured from the 14:00 announcement instant, so
    the announcement-minute move is included in ret_immediate, not leaked into the
    pre-window.
    """
    frame = bars.copy()
    frame["timestamp_et"] = pd.to_datetime(frame["timestamp_et"])
    frame["_t"] = frame["timestamp_et"].dt.time

    rows: list[dict[str, object]] = []
    for event_date, day in frame.groupby("event_date"):
        day = day.sort_values("timestamp_et")

        # Alignment invariant #1: every bar for this event must fall on the event's
        # own calendar date. Cross-day contamination is a real misalignment risk
        # (e.g. a join pulling adjacent-day bars) and would corrupt the windows.
        # str(...)[:10] normalises event_date whether it is a str, date, or Timestamp.
        event_date_str = str(event_date)[:10]
        bar_dates = day["timestamp_et"].dt.strftime("%Y-%m-%d").unique()
        if list(bar_dates) != [event_date_str]:
            raise ValueError(
                f"{event_date_str}: bars span foreign dates {list(bar_dates)}"
            )

        # Pre-window: bars strictly before the announcement minute. Realized vol
        # from pre-window closes is pure pre-announcement.
        pre = day[(day["_t"] >= _PRE_START) & (day["_t"] < _ANNOUNCE)]
        px_1330 = _open_at(day, _PRE_START)
        px_1400 = _open_at(day, _ANNOUNCE)  # announcement instant (pre-reaction)
        px_1430 = _open_at(day, _IMMEDIATE_END)
        px_1500 = _open_at(day, _DELAYED_END)

        pre_logrets = np.diff(np.log(pre["close"].to_numpy())) if len(pre) > 1 else np.array([])
        ret_immediate = _log_change(px_1430, px_1400)  # full reaction incl. 14:00 min
        ret_delayed = _log_change(px_1500, px_1430)

        rows.append(
            {
                "event_date": event_date_str,
                "pre_close": px_1400,
                "pre_ret": _log_change(px_1400, px_1330),
                "pre_rv": float(np.std(pre_logrets)) if pre_logrets.size else None,
                "pre_volume": float(pre["volume"].sum()),
                "n_pre_bars": int(len(pre)),
                "px_1400": px_1400,
                "px_1430": px_1430,
                "px_1500": px_1500,
                "ret_immediate": ret_immediate,
                "dir_immediate": None if ret_immediate is None else int(ret_immediate > 0),
                "mag_immediate": None if ret_immediate is None else abs(ret_immediate),
                "ret_delayed": ret_delayed,
                "dir_delayed": None if ret_delayed is None else int(ret_delayed > 0),
                "mag_delayed": None if ret_delayed is None else abs(ret_delayed),
                "n_bars": int(len(day)),
                "has_anchors": int(
                    None not in (px_1330, px_1400, px_1430, px_1500)
                ),
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
    merged = events.merge(
        text_by_date, left_on="event_date", right_index=True, how="left"
    )
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
