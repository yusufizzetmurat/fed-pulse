"""Tests for the FRED client.

httpx is mocked via ``httpx.MockTransport`` so the tests run without network.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.services import fred_client


SAMPLE_PAYLOAD = {
    "realtime_start": "2026-05-15",
    "realtime_end": "2026-05-15",
    "observation_start": "2024-01-01",
    "observation_end": "2024-01-05",
    "count": 5,
    "observations": [
        {"date": "2024-01-02", "value": "5.33", "realtime_start": "2026-05-15", "realtime_end": "2026-05-15"},
        {"date": "2024-01-03", "value": "5.33", "realtime_start": "2026-05-15", "realtime_end": "2026-05-15"},
        {"date": "2024-01-04", "value": ".", "realtime_start": "2026-05-15", "realtime_end": "2026-05-15"},
        {"date": "2024-01-05", "value": "5.34", "realtime_start": "2026-05-15", "realtime_end": "2026-05-15"},
        {"date": "2024-01-06", "value": "5.33", "realtime_start": "2026-05-15", "realtime_end": "2026-05-15"},
    ],
}


def _mock_transport(captured: dict[str, dict[str, str]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        captured["url"] = str(request.url).split("?")[0]
        return httpx.Response(200, json=SAMPLE_PAYLOAD)

    return httpx.MockTransport(handler)


def test_fetch_fred_series_hits_api_and_caches(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    captured: dict[str, dict[str, str]] = {}

    response = fred_client.fetch_fred_series(
        "DFF",
        start="2024-01-01",
        end="2024-01-05",
        cache_dir=tmp_path,
        transport=_mock_transport(captured),
    )

    assert captured["url"] == fred_client.FRED_BASE_URL
    assert captured["params"]["series_id"] == "DFF"
    assert captured["params"]["api_key"] == "test-key"
    assert captured["params"]["observation_start"] == "2024-01-01"
    assert captured["params"]["observation_end"] == "2024-01-05"
    assert captured["params"]["file_type"] == "json"

    assert response.series_id == "DFF"
    assert response.count == 5
    assert len(response.observations) == 5
    values = [obs.value for obs in response.observations]
    # FRED encodes missing values as "."; the parser coerces those to None.
    assert values == [5.33, 5.33, None, 5.34, 5.33]

    cache_path = tmp_path / "DFF.json"
    lock_path = tmp_path / fred_client.SOURCES_LOCK_NAME
    assert cache_path.exists()
    assert lock_path.exists()

    lock = json.loads(lock_path.read_text())
    assert "DFF" in lock
    assert lock["DFF"]["count"] == 5
    assert len(lock["DFF"]["sha256"]) == 64


def test_fetch_fred_series_reuses_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    captured: dict[str, dict[str, str]] = {}

    fred_client.fetch_fred_series(
        "DFF",
        cache_dir=tmp_path,
        transport=_mock_transport(captured),
    )

    captured_second: dict[str, dict[str, str]] = {}

    def boom(_: httpx.Request) -> httpx.Response:
        captured_second["called"] = {"true": "true"}
        return httpx.Response(500)

    response = fred_client.fetch_fred_series(
        "DFF",
        cache_dir=tmp_path,
        transport=httpx.MockTransport(boom),
    )

    assert captured_second == {}  # cache hit; transport never called
    assert response.count == 5


def test_fetch_fred_series_force_refresh_overrides_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    captured: dict[str, dict[str, str]] = {}
    fred_client.fetch_fred_series(
        "DFF",
        cache_dir=tmp_path,
        transport=_mock_transport(captured),
    )

    refreshed_payload = {
        "realtime_start": "2026-05-16",
        "realtime_end": "2026-05-16",
        "observation_start": "2024-01-01",
        "observation_end": "2024-01-06",
        "count": 1,
        "observations": [
            {"date": "2024-01-06", "value": "5.40", "realtime_start": "2026-05-16", "realtime_end": "2026-05-16"},
        ],
    }

    def refreshed_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=refreshed_payload)

    response = fred_client.fetch_fred_series(
        "DFF",
        cache_dir=tmp_path,
        transport=httpx.MockTransport(refreshed_handler),
        force_refresh=True,
    )

    assert response.count == 1
    assert response.observations[0].value == 5.40


def test_missing_api_key_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("FRED_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        fred_client.fetch_fred_series(
            "DFF",
            cache_dir=tmp_path,
            transport=_mock_transport({}),
        )


def test_observation_value_parser_handles_missing_marker() -> None:
    assert fred_client._parse_observation_value(".") is None
    assert fred_client._parse_observation_value("") is None
    assert fred_client._parse_observation_value(None) is None
    assert fred_client._parse_observation_value("not-a-number") is None
    assert fred_client._parse_observation_value("5.33") == 5.33
    assert fred_client._parse_observation_value(0) == 0.0
