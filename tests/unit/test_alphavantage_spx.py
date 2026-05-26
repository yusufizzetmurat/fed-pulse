"""Tests for the Alpha Vantage SPX intraday backfill module (#159)."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from app.data import alphavantage_spx
from app.data.alphavantage_spx import (
    FomcWindowReturn,
    IntradayBar,
    _bracket_window_returns,
    _months_covering,
    _parse_intraday_payload,
    compute_window_returns,
    load_window_returns,
)


def _synth_payload(date_iso: str, bars: list[dict[str, str]]) -> dict[str, Any]:
    series = {bar["t"]: {
        "1. open": bar["o"], "2. high": bar["h"], "3. low": bar["l"],
        "4. close": bar["c"], "5. volume": bar["v"],
    } for bar in bars}
    return {
        "Meta Data": {"1. Information": "test", "2. Symbol": "SPY"},
        "Time Series (1min)": series,
    }


def test_parse_intraday_payload_sorts_oldest_first() -> None:
    payload = _synth_payload(
        "2024-01-31",
        [
            {"t": "2024-01-31 14:30:00", "o": "1", "h": "1", "l": "1", "c": "473.5", "v": "100"},
            {"t": "2024-01-31 13:30:00", "o": "1", "h": "1", "l": "1", "c": "470.0", "v": "100"},
            {"t": "2024-01-31 14:00:00", "o": "1", "h": "1", "l": "1", "c": "471.2", "v": "100"},
        ],
    )
    bars = _parse_intraday_payload(payload)
    assert [b.timestamp_et for b in bars] == [
        "2024-01-31 13:30:00",
        "2024-01-31 14:00:00",
        "2024-01-31 14:30:00",
    ]
    assert bars[0].close == pytest.approx(470.0)


def test_parse_intraday_rejects_rate_limit_note() -> None:
    with pytest.raises(RuntimeError, match="rate limit"):
        _parse_intraday_payload(
            {"Note": "Thank you for using Alpha Vantage! The free tier..."}
        )


def test_parse_intraday_rejects_error_message() -> None:
    with pytest.raises(RuntimeError, match="Alpha Vantage error"):
        _parse_intraday_payload({"Error Message": "Invalid API call"})


def test_bracket_window_returns_picks_correct_pre_and_post() -> None:
    bars = [
        IntradayBar("2024-01-31 13:25:00", 470.0, 470.0, 470.0, 470.0, 0.0),
        IntradayBar("2024-01-31 13:30:00", 470.5, 470.5, 470.5, 470.5, 0.0),
        IntradayBar("2024-01-31 13:59:00", 472.0, 472.0, 472.0, 472.0, 0.0),
        IntradayBar("2024-01-31 14:00:00", 473.0, 473.0, 473.0, 473.0, 0.0),
        IntradayBar("2024-01-31 14:30:00", 475.0, 475.0, 475.0, 475.0, 0.0),
        IntradayBar("2024-01-31 14:35:00", 475.5, 475.5, 475.5, 475.5, 0.0),
    ]
    result = _bracket_window_returns(
        bars,
        datetime.date(2024, 1, 31),
        announcement_time=datetime.time(14, 0),
        window_minutes=30,
    )
    assert result is not None
    pre, post = result
    assert pre == pytest.approx(470.5)
    assert post == pytest.approx(475.0)


def test_bracket_window_returns_missing_side_returns_none() -> None:
    # Only post-window bars present; pre-window missing.
    bars = [
        IntradayBar("2024-01-31 14:30:00", 475.0, 475.0, 475.0, 475.0, 0.0),
        IntradayBar("2024-01-31 14:35:00", 475.5, 475.5, 475.5, 475.5, 0.0),
    ]
    result = _bracket_window_returns(
        bars,
        datetime.date(2024, 1, 31),
        announcement_time=datetime.time(14, 0),
        window_minutes=30,
    )
    assert result is None


def test_compute_window_returns_drops_uncovered_dates() -> None:
    bars = [
        IntradayBar("2024-01-31 13:30:00", 470.0, 470.0, 470.0, 470.0, 0.0),
        IntradayBar("2024-01-31 14:30:00", 475.0, 475.0, 475.0, 475.0, 0.0),
    ]
    rows = compute_window_returns(
        bars,
        [datetime.date(2024, 1, 31), datetime.date(2024, 3, 20)],
    )
    assert len(rows) == 1
    assert rows[0].event_date == "2024-01-31"
    assert rows[0].return_pct == pytest.approx((475.0 - 470.0) / 470.0)


def test_months_covering_dedupes_and_sorts() -> None:
    months = _months_covering(
        [
            datetime.date(2024, 1, 31),
            datetime.date(2024, 1, 31),
            datetime.date(2023, 12, 13),
            datetime.date(2024, 3, 20),
        ]
    )
    assert months == ["2023-12", "2024-01", "2024-03"]


def test_backfill_writes_parquet_and_sources_lock(tmp_path, monkeypatch) -> None:
    """End-to-end smoke: stubbed Alpha Vantage client, two events
    spanning two months. Cache + lock file land at the expected paths."""

    bars_jan = [
        IntradayBar("2024-01-31 13:30:00", 470.0, 470.0, 470.0, 470.0, 0.0),
        IntradayBar("2024-01-31 14:30:00", 472.5, 472.5, 472.5, 472.5, 0.0),
    ]
    bars_mar = [
        IntradayBar("2024-03-20 13:30:00", 510.0, 510.0, 510.0, 510.0, 0.0),
        IntradayBar("2024-03-20 14:30:00", 513.0, 513.0, 513.0, 513.0, 0.0),
    ]
    calls: list[str] = []

    def fake_fetch(*, api_key, symbol, interval, month, client):
        calls.append(month)
        return {"2024-01": bars_jan, "2024-03": bars_mar}[month]

    monkeypatch.setattr(alphavantage_spx, "fetch_intraday_minute_bars", fake_fetch)
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")
    sleep_calls: list[float] = []

    parquet_path = alphavantage_spx.backfill_fomc_days(
        fomc_dates=[datetime.date(2024, 1, 31), datetime.date(2024, 3, 20)],
        cache_dir=tmp_path,
        sleep_fn=lambda s: sleep_calls.append(s),
    )
    assert parquet_path == tmp_path / "spx_intraday_fomc_days.parquet"
    assert parquet_path.exists()
    assert calls == ["2024-01", "2024-03"]
    # One sleep between two months (the first call does not delay).
    assert len(sleep_calls) == 1

    frame = pd.read_parquet(parquet_path)
    assert len(frame) == 2
    assert set(frame["event_date"]) == {"2024-01-31", "2024-03-20"}

    lock_path = tmp_path / "SOURCES.lock"
    assert lock_path.exists()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    entry = lock["spx_intraday_fomc_days.parquet"]
    assert entry["rows"] == 2
    assert entry["months_fetched"] == ["2024-01", "2024-03"]
    assert entry["source"] == "alphavantage"


def test_load_window_returns_missing_cache_returns_empty(tmp_path) -> None:
    assert load_window_returns(tmp_path) == {}


def test_load_window_returns_round_trips_through_parquet(tmp_path) -> None:
    parquet_path = tmp_path / "spx_intraday_fomc_days.parquet"
    pd.DataFrame(
        [
            {
                "event_date": "2024-01-31",
                "pre_close": 470.0,
                "post_close": 472.5,
                "return_pct": (472.5 - 470.0) / 470.0,
                "window_minutes": 30,
                "symbol": "SPY",
                "fetched_at_utc": "2026-05-23T10:00:00Z",
            }
        ]
    ).to_parquet(parquet_path, index=False)
    returns = load_window_returns(tmp_path)
    assert returns == {"2024-01-31": pytest.approx((472.5 - 470.0) / 470.0)}
