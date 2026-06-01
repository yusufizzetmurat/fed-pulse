"""Tests for the Polygon.io SPX intraday backfill module."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from app.data import polygon_spx
from app.data.polygon_spx import (
    PolygonBar,
    _parse_aggs_payload,
    _ms_to_et_string,
    filter_window_bars,
    DEFAULT_SYMBOL,
)


def _ms(et_iso: str) -> int:
    from zoneinfo import ZoneInfo

    naive = datetime.datetime.fromisoformat(et_iso)
    aware = naive.replace(tzinfo=ZoneInfo("America/New_York"))
    return int(aware.timestamp() * 1000)


def _aggs_payload(bars: list[dict[str, Any]]) -> dict[str, Any]:
    return {"status": "OK", "ticker": "SPY", "resultsCount": len(bars), "results": bars}


def test_parse_aggs_payload_reads_ohlcv() -> None:
    payload = _aggs_payload(
        [{"t": _ms("2024-01-31 13:59:00"), "o": 1.0, "h": 2.0, "l": 0.5, "c": 470.0, "v": 100.0}]
    )
    bars = _parse_aggs_payload(payload)
    assert len(bars) == 1
    assert bars[0].close == pytest.approx(470.0)
    assert bars[0].timestamp_et == "2024-01-31 13:59:00"


def test_parse_aggs_payload_raises_on_error_status() -> None:
    with pytest.raises(RuntimeError, match="Polygon error"):
        _parse_aggs_payload({"status": "ERROR", "error": "Unknown API Key"})


def test_parse_aggs_payload_empty_results_returns_empty() -> None:
    assert _parse_aggs_payload({"status": "OK", "resultsCount": 0, "results": []}) == []


def test_filter_window_keeps_only_1330_to_1500_et() -> None:
    payload = _aggs_payload(
        [
            {"t": _ms("2024-01-31 09:30:00"), "o": 1, "h": 1, "l": 1, "c": 460.0, "v": 1},
            {"t": _ms("2024-01-31 13:30:00"), "o": 1, "h": 1, "l": 1, "c": 470.0, "v": 1},
            {"t": _ms("2024-01-31 14:00:00"), "o": 1, "h": 1, "l": 1, "c": 471.0, "v": 1},
            {"t": _ms("2024-01-31 15:00:00"), "o": 1, "h": 1, "l": 1, "c": 472.0, "v": 1},
            {"t": _ms("2024-01-31 15:30:00"), "o": 1, "h": 1, "l": 1, "c": 473.0, "v": 1},
        ]
    )
    bars = _parse_aggs_payload(payload)
    kept = filter_window_bars(bars, datetime.date(2024, 1, 31))
    closes = [b.close for b in kept]
    assert closes == [470.0, 471.0, 472.0]  # 13:30, 14:00, 15:00 inclusive; 09:30 + 15:30 dropped


import httpx


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{self.status_code}",
                request=httpx.Request("GET", "http://x"),
                response=self,  # type: ignore[arg-type]
            )

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    """Returns a per-date payload keyed on the date in the request URL.

    ``status_by_date`` overrides the HTTP status for specific dates so
    tests can exercise the 403 (out-of-plan-window) skip path.
    """

    def __init__(
        self,
        by_date: dict[str, dict[str, Any]],
        status_by_date: dict[str, int] | None = None,
    ) -> None:
        self._by_date = by_date
        self._status_by_date = status_by_date or {}
        self.calls: list[str] = []
        self.header_seen: dict[str, Any] | None = None

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
    ) -> _FakeResponse:
        date_iso = url.rstrip("/").split("/")[-1]
        self.calls.append(date_iso)
        self.header_seen = headers
        return _FakeResponse(
            self._by_date.get(date_iso, {"status": "OK", "results": []}),
            status_code=self._status_by_date.get(date_iso, 200),
        )

    def close(self) -> None:
        return None


def test_backfill_writes_raw_bars_parquet_and_lock(tmp_path: Path) -> None:
    by_date = {
        "2024-01-31": _aggs_payload(
            [
                {"t": _ms("2024-01-31 13:30:00"), "o": 1, "h": 1, "l": 1, "c": 470.0, "v": 1},
                {"t": _ms("2024-01-31 14:30:00"), "o": 1, "h": 1, "l": 1, "c": 472.0, "v": 1},
                {"t": _ms("2024-01-31 16:30:00"), "o": 1, "h": 1, "l": 1, "c": 480.0, "v": 1},
            ]
        )
    }
    client = _FakeClient(by_date)
    out = polygon_spx.backfill_fomc_days(
        fomc_dates=[datetime.date(2024, 1, 31)],
        cache_dir=tmp_path,
        api_key="test-key",
        sleep_fn=lambda _s: None,
        client=client,
    )
    frame = pd.read_parquet(out)
    # Only the two in-window bars (13:30, 14:30) survive; 16:30 dropped.
    assert len(frame) == 2
    assert set(frame["event_date"]) == {"2024-01-31"}
    assert sorted(frame["close"]) == [470.0, 472.0]
    lock = json.loads((tmp_path / "SOURCES.lock").read_text())
    assert lock[polygon_spx.INTRADAY_PARQUET]["rows"] == 2
    assert lock[polygon_spx.INTRADAY_PARQUET]["source"] == "polygon"


def test_backfill_empty_dates_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        polygon_spx.backfill_fomc_days(fomc_dates=[], cache_dir=tmp_path, api_key="k")


def test_backfill_skips_unauthorized_dates(tmp_path: Path) -> None:
    by_date = {
        "2024-01-31": _aggs_payload(
            [{"t": _ms("2024-01-31 14:00:00"), "o": 1, "h": 1, "l": 1, "c": 471.0, "v": 1}]
        ),
    }
    # 2010-01-27 predates the plan window -> 403; it must be skipped, not fatal.
    client = _FakeClient(by_date, status_by_date={"2010-01-27": 403})
    out = polygon_spx.backfill_fomc_days(
        fomc_dates=[datetime.date(2010, 1, 27), datetime.date(2024, 1, 31)],
        cache_dir=tmp_path,
        api_key="test-key",
        sleep_fn=lambda _s: None,
        client=client,
    )
    frame = pd.read_parquet(out)
    assert set(frame["event_date"]) == {"2024-01-31"}
    assert len(frame) == 1


def test_backfill_non_403_error_propagates(tmp_path: Path) -> None:
    client = _FakeClient({}, status_by_date={"2024-01-31": 500})
    with pytest.raises(httpx.HTTPStatusError):
        polygon_spx.backfill_fomc_days(
            fomc_dates=[datetime.date(2024, 1, 31)],
            cache_dir=tmp_path,
            api_key="test-key",
            sleep_fn=lambda _s: None,
            client=client,
        )


def test_fetch_uses_authorization_header_not_url(tmp_path: Path) -> None:
    client = _FakeClient(
        {"2024-01-31": _aggs_payload([])},
    )
    polygon_spx.fetch_day_minute_bars(
        api_key="secret-key", event_date=datetime.date(2024, 1, 31), client=client
    )
    assert client.header_seen == {"Authorization": "Bearer secret-key"}


def test_load_intraday_bars_groups_by_event_date(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "event_date": ["2024-01-31", "2024-01-31", "2024-03-20"],
            "timestamp_et": [
                "2024-01-31 13:30:00",
                "2024-01-31 13:31:00",
                "2024-03-20 13:30:00",
            ],
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [470.0, 470.5, 510.0],
            "volume": [1.0, 1.0, 1.0],
            "symbol": ["SPY", "SPY", "SPY"],
            "fetched_at_utc": ["x", "x", "x"],
        }
    )
    (tmp_path / polygon_spx.INTRADAY_PARQUET).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(tmp_path / polygon_spx.INTRADAY_PARQUET, index=False)
    grouped = polygon_spx.load_intraday_bars(cache_dir=tmp_path)
    assert set(grouped) == {"2024-01-31", "2024-03-20"}
    assert len(grouped["2024-01-31"]) == 2


def test_load_intraday_bars_missing_returns_empty(tmp_path: Path) -> None:
    assert polygon_spx.load_intraday_bars(cache_dir=tmp_path) == {}


def test_fomc_dates_from_events_parquet_defaults_to_statement(tmp_path: Path) -> None:
    events = pd.DataFrame(
        {
            "event_date": ["2024-01-31", "2024-01-31", "2024-03-20", "2024-02-21"],
            "event_kind": ["statement", "statement", "statement", "minutes"],
        }
    )
    p = tmp_path / "events.parquet"
    events.to_parquet(p, index=False)
    # Default kind is "statement"; the 2024-02-21 minutes row is excluded.
    dates = polygon_spx.fomc_dates_from_events_parquet(p)
    assert dates == [datetime.date(2024, 1, 31), datetime.date(2024, 3, 20)]


def test_fomc_dates_min_date_floor(tmp_path: Path) -> None:
    events = pd.DataFrame(
        {
            "event_date": ["2008-01-30", "2010-01-27", "2024-03-20"],
            "event_kind": ["statement", "statement", "statement"],
        }
    )
    p = tmp_path / "events.parquet"
    events.to_parquet(p, index=False)
    dates = polygon_spx.fomc_dates_from_events_parquet(p, min_date=datetime.date(2010, 1, 1))
    assert dates == [datetime.date(2010, 1, 27), datetime.date(2024, 3, 20)]
