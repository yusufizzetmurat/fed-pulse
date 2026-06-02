"""Tests for the intraday event dataset builder."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from app.data import intraday_event_builder as ieb
from app.data.polygon_spx import PolygonBar


def _bar(ts: str, close: float, vol: float = 1.0) -> PolygonBar:
    return PolygonBar(timestamp_et=ts, open=close, high=close, low=close, close=close, volume=vol)


def _full_day(date: str, base: float = 100.0) -> list[PolygonBar]:
    # 91 bars 13:30..15:00, close rising by 0.1/min so all returns are positive.
    out = []
    t = datetime.datetime.fromisoformat(f"{date} 13:30:00")
    for i in range(91):
        ts = t.strftime("%Y-%m-%d %H:%M:00")
        out.append(_bar(ts, base + 0.1 * i, vol=10.0 + i))
        t += datetime.timedelta(minutes=1)
    return out


def test_anchor_close_reads_exact_minute() -> None:
    bars = _full_day("2024-01-31")
    assert ieb._anchor_close(bars, "2024-01-31 14:00:00") == pytest.approx(100.0 + 0.1 * 30)


def test_anchor_close_missing_returns_none() -> None:
    bars = _full_day("2024-01-31")
    assert ieb._anchor_close(bars, "2024-01-31 14:07:00") is not None  # present
    assert ieb._anchor_close([], "2024-01-31 14:00:00") is None


def test_compute_target_direction_and_magnitude() -> None:
    t = ieb._compute_target(100.0, 101.0)
    assert t == {"ret": pytest.approx(0.01), "dir": 1, "mag": pytest.approx(0.01)}
    t2 = ieb._compute_target(100.0, 99.0)
    assert t2["dir"] == 0 and t2["mag"] == pytest.approx(0.01)


def test_pre_window_excludes_announcement_and_after() -> None:
    bars = _full_day("2024-01-31")
    pre = ieb._pre_window_bars(bars, datetime.date(2024, 1, 31))
    assert pre[0].timestamp_et == "2024-01-31 13:30:00"
    assert pre[-1].timestamp_et == "2024-01-31 14:00:00"  # inclusive of the 14:00 boundary
    assert len(pre) == 31
    assert all(b.timestamp_et <= "2024-01-31 14:00:00" for b in pre)


def test_assert_no_leakage_flags_post_announcement_bar() -> None:
    leaky_pre = [_bar("2024-01-31 13:59:00", 100.0), _bar("2024-01-31 14:01:00", 101.0)]
    with pytest.raises(AssertionError, match="leak"):
        ieb._assert_no_leakage(leaky_pre)
    clean_pre = [_bar("2024-01-31 13:59:00", 100.0), _bar("2024-01-31 14:00:00", 100.5)]
    ieb._assert_no_leakage(clean_pre)  # no raise


def test_statement_text_by_date_dedups_horizons(tmp_path: Path) -> None:
    events = pd.DataFrame(
        {
            "event_date": ["2024-01-31"] * 4 + ["2024-03-20"],
            "event_kind": ["statement"] * 4 + ["minutes"],
            "horizon": [1, 5, 10, 30, 1],
            "text": ["hawkish text"] * 4 + ["minutes text"],
        }
    )
    p = tmp_path / "events.parquet"
    events.to_parquet(p, index=False)
    out = ieb.statement_text_by_date(p)
    assert out == {"2024-01-31": "hawkish text"}  # minutes excluded, horizons collapsed


def test_build_event_row_targets_and_sequence() -> None:
    bars = _full_day("2024-01-31", base=100.0)
    row = ieb._build_event_row("2024-01-31", bars, "hawkish text", symbol="SPY")
    assert row is not None
    assert row["n_pre_bars"] == 31
    assert row["close_1400"] == pytest.approx(100.0 + 0.1 * 30)  # 103.0
    assert row["close_1430"] == pytest.approx(100.0 + 0.1 * 60)  # 106.0
    assert row["close_1500"] == pytest.approx(100.0 + 0.1 * 90)  # 109.0
    assert row["dir_immediate"] == 1 and row["dir_delayed"] == 1
    assert row["ret_immediate"] == pytest.approx(106.0 / 103.0 - 1.0)
    assert row["text"] == "hawkish text"


def test_build_event_row_missing_anchor_returns_none() -> None:
    bars = [b for b in _full_day("2024-01-31") if b.timestamp_et != "2024-01-31 14:30:00"]
    assert ieb._build_event_row("2024-01-31", bars, "t", symbol="SPY") is None


def test_build_dataset_writes_parquet(tmp_path: Path) -> None:
    bars_by_date = {"2024-01-31": _full_day("2024-01-31")}
    texts = {"2024-01-31": "hawkish", "2024-03-20": "no bars"}
    out = tmp_path / "intraday_events.parquet"
    n = ieb.build_dataset(bars_by_date, texts, out_path=out, symbol="SPY")
    assert n == 1
    frame = pd.read_parquet(out)
    assert list(frame["event_date"]) == ["2024-01-31"]
    assert {"pre_close", "ret_immediate", "dir_delayed", "text"} <= set(frame.columns)
