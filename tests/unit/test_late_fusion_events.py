"""Unit tests for clean-room FOMC event-frame assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.late_fusion_events import (
    build_event_windows,
    join_sep_features,
    join_statement_text,
)


def _bars_for_day(date: str, prices_by_minute: dict[str, float]) -> pd.DataFrame:
    rows = [
        {
            "event_date": date,
            "timestamp_et": f"{date} {hhmm}:00",
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 1000.0,
            "symbol": "SPY",
        }
        for hhmm, price in prices_by_minute.items()
    ]
    return pd.DataFrame(rows)


def _full_day(date: str, p1330: float, p1400: float, p1430: float, p1500: float) -> pd.DataFrame:
    """A minimal but alignment-valid day: anchor bars at 13:30/14:00/14:30/15:00
    plus one extra pre-window bar so pre_rv is defined."""
    minutes = {
        "13:30": p1330,
        "13:45": (p1330 + p1400) / 2,
        "14:00": p1400,
        "14:30": p1430,
        "15:00": p1500,
    }
    return _bars_for_day(date, minutes)


def test_window_returns_and_direction() -> None:
    bars = _full_day("2024-03-20", 100.0, 100.0, 101.0, 100.5)
    out = build_event_windows(bars)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["px_1400"] == 100.0
    assert row["px_1430"] == 101.0
    assert row["px_1500"] == 100.5
    # immediate reaction up -> dir 1, magnitude = |log(101/100)|
    assert row["dir_immediate"] == 1
    assert row["mag_immediate"] == pytest.approx(abs(np.log(101 / 100)))
    # delayed reaction down -> dir 0
    assert row["dir_delayed"] == 0
    assert row["ret_delayed"] == pytest.approx(np.log(100.5 / 101))
    # pre-window: 2 bars strictly before 14:00 (13:30, 13:45)
    assert row["n_pre_bars"] == 2


def test_pre_window_excludes_announcement_bar() -> None:
    bars = _full_day("2024-03-20", 100.0, 100.0, 101.0, 100.5)
    out = build_event_windows(bars).iloc[0]
    # the 14:00 bar must NOT be counted in the pre-window
    assert out["n_pre_bars"] == 2  # 13:30 and 13:45 only


def test_mark_uses_bar_open_so_announcement_minute_is_in_the_reaction() -> None:
    # The 14:00 bar holds the announcement jump: it OPENS at 100 (pre-reaction) and
    # CLOSES at 103 (post-jump). The reaction target must capture that jump (use the
    # open as the 14:00 mark), and the pre-window must NOT see the post-jump close.
    bars = pd.DataFrame(
        [
            {"event_date": "2024-03-20", "timestamp_et": "2024-03-20 13:30:00",
             "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0, "symbol": "SPY"},
            {"event_date": "2024-03-20", "timestamp_et": "2024-03-20 13:59:00",
             "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0, "symbol": "SPY"},
            # announcement minute: opens pre-reaction at 100, jumps to close 103
            {"event_date": "2024-03-20", "timestamp_et": "2024-03-20 14:00:00",
             "open": 100.0, "high": 103.0, "low": 100.0, "close": 103.0, "volume": 50.0, "symbol": "SPY"},
            {"event_date": "2024-03-20", "timestamp_et": "2024-03-20 14:30:00",
             "open": 104.0, "high": 104.0, "low": 104.0, "close": 104.0, "volume": 5.0, "symbol": "SPY"},
            {"event_date": "2024-03-20", "timestamp_et": "2024-03-20 15:00:00",
             "open": 104.0, "high": 104.0, "low": 104.0, "close": 104.0, "volume": 5.0, "symbol": "SPY"},
        ]
    )
    row = build_event_windows(bars).iloc[0]
    # 14:00 mark = OPEN of the 14:00 bar = 100 (pre-reaction), NOT the 103 close
    assert row["px_1400"] == 100.0
    # immediate reaction 100 -> 104 captures the full jump (would be ~0 if it used
    # the contaminated 103 close as the start)
    assert row["ret_immediate"] == pytest.approx(np.log(104 / 100))
    # pre-window return is flat (100/100) — the announcement jump did NOT leak in
    assert row["pre_ret"] == pytest.approx(0.0)


def test_cross_day_contamination_raises() -> None:
    # An event whose bars include a foreign calendar date is a real misalignment.
    bars = _full_day("2024-03-20", 100.0, 100.0, 101.0, 100.5)
    stray = bars.iloc[[0]].copy()
    stray["timestamp_et"] = "2024-03-21 14:00:00"  # wrong day, same event_date tag
    corrupted = pd.concat([bars, stray], ignore_index=True)
    with pytest.raises(ValueError, match="foreign dates"):
        build_event_windows(corrupted)


def test_missing_anchor_flagged_not_dropped() -> None:
    # A day missing the 14:30 anchor: row still produced, has_anchors=0, returns None.
    bars = _bars_for_day(
        "2024-03-20", {"13:30": 100.0, "13:45": 100.0, "14:00": 100.0, "15:00": 100.5}
    )
    out = build_event_windows(bars).iloc[0]
    assert out["has_anchors"] == 0
    assert out["px_1430"] is None
    assert out["ret_immediate"] is None


def test_join_statement_text_exact_date() -> None:
    events = pd.DataFrame({"event_date": ["2024-03-20", "2024-05-01"]})
    corpus = pd.DataFrame(
        {
            "doc_type": ["statement", "speech", "statement"],
            "date": ["2024-03-20", "2024-03-20", "2024-05-01"],
            "text": ["FOMC March statement", "a speech", "FOMC May statement"],
            "title": ["Statement Mar", "Speech", "Statement May"],
        }
    )
    merged = join_statement_text(events, corpus)
    assert merged.loc[merged["event_date"] == "2024-03-20", "text"].iloc[0] == (
        "FOMC March statement"
    )
    # the speech on the same date must NOT be chosen
    assert "speech" not in merged.loc[merged["event_date"] == "2024-03-20", "text"].iloc[0]
    assert merged.loc[merged["event_date"] == "2024-05-01", "text"].iloc[0] == (
        "FOMC May statement"
    )


def test_join_statement_text_flags_missing() -> None:
    events = pd.DataFrame({"event_date": ["2024-03-20", "2099-01-01"]})
    corpus = pd.DataFrame(
        {"doc_type": ["statement"], "date": ["2024-03-20"], "text": ["x"], "title": ["t"]}
    )
    merged = join_statement_text(events, corpus)
    assert merged["text"].isna().sum() == 1


def _sep_row(date: str, var: str, hor: str, **vals: float) -> dict[str, object]:
    base = {
        "meeting_date": date,
        "variable": var,
        "horizon": hor,
        "median": None,
        "central_low": None,
        "central_high": None,
        "range_low": None,
        "range_high": None,
    }
    base.update(vals)
    return base


def test_join_sep_features_availability_flag() -> None:
    events = pd.DataFrame({"event_date": ["2024-03-20", "2024-05-01"]})
    sep = pd.DataFrame(
        [
            _sep_row("2024-03-20", "ffr", "2024", median=4.6, central_low=4.4, central_high=4.9),
            _sep_row("2024-03-20", "gdp", "2024", median=2.1, central_low=2.0, central_high=2.4),
        ]
    )
    merged = join_sep_features(events, sep)
    assert merged.loc[merged["event_date"] == "2024-03-20", "sep_available"].iloc[0] == 1
    assert merged.loc[merged["event_date"] == "2024-05-01", "sep_available"].iloc[0] == 0
    row = merged.loc[merged["event_date"] == "2024-03-20"].iloc[0]
    assert row["sep_point_ffr_2024"] == 4.6
    # dispersion = central-tendency width
    assert row["sep_disp_ffr_2024"] == pytest.approx(0.5)


def test_join_sep_legacy_uses_central_tendency_midpoint() -> None:
    # 2013-2014 era: no median -> point falls back to CT midpoint, still available.
    events = pd.DataFrame({"event_date": ["2014-03-19"]})
    sep = pd.DataFrame(
        [_sep_row("2014-03-19", "gdp", "2014", central_low=2.8, central_high=3.0)]
    )
    merged = join_sep_features(events, sep)
    assert merged["sep_available"].iloc[0] == 1
    assert merged["sep_point_gdp_2014"].iloc[0] == pytest.approx(2.9)
