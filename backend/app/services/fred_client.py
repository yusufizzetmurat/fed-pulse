"""FRED time-series client for the credibility module.

Pulls per-series observations from the St. Louis Fed FRED REST API, caches
them as JSON under ``data/external/fred/<series_id>.json`` plus a
``SOURCES.lock`` entry recording the SHA-256 + retrieval timestamp.
Reproducibility: callers pin which observation_revision (FRED's vintage
mechanism) they're reading via the ``realtime_start`` / ``realtime_end``
fields on the response. First retrieval records the SHA in SOURCES.lock;
subsequent retrievals reuse the cache unless ``force_refresh=True``.

Series most relevant to the credibility module:
- ``DFF``    Federal funds effective rate (daily).
- ``DGS10``  10-year Treasury constant maturity (daily).
- ``GS3M``   3-month Treasury constant maturity (monthly avg).
- ``CPIAUCSL`` CPI all-urban consumers seasonally-adjusted (monthly).
- ``UNRATE`` Civilian unemployment rate (monthly).

Set ``FRED_API_KEY`` in the environment. Get a free key at
https://fred.stlouisfed.org/docs/api/api_key.html (10k requests/day).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from app.config import DATA_DIR

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_CACHE_DIR = DATA_DIR / "external" / "fred"
SOURCES_LOCK_NAME = "SOURCES.lock"
DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class FredObservation:
    date: str
    value: float | None
    realtime_start: str
    realtime_end: str


@dataclass(frozen=True)
class FredSeriesResponse:
    series_id: str
    realtime_start: str
    realtime_end: str
    observation_start: str
    observation_end: str
    count: int
    observations: list[FredObservation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "realtime_start": self.realtime_start,
            "realtime_end": self.realtime_end,
            "observation_start": self.observation_start,
            "observation_end": self.observation_end,
            "count": self.count,
            "observations": [asdict(obs) for obs in self.observations],
        }


def _resolve_api_key(api_key: str | None) -> str:
    if api_key:
        return api_key
    env_value = os.environ.get("FRED_API_KEY") or os.environ.get("FRED_TOKEN")
    if not env_value:
        raise RuntimeError(
            "FRED_API_KEY (or FRED_TOKEN) not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html."
        )
    return env_value


def _cache_paths(series_id: str, cache_dir: Path) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{series_id}.json", cache_dir / SOURCES_LOCK_NAME


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _read_lock(lock_path: Path) -> dict[str, Any]:
    if not lock_path.exists():
        return {}
    try:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_lock(lock_path: Path, payload: dict[str, Any]) -> None:
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _parse_observation_value(raw: Any) -> float | None:
    """FRED encodes missing values as the literal string '.'."""
    if raw is None or raw == "" or raw == ".":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _parse_observations(payload: dict[str, Any], series_id: str) -> FredSeriesResponse:
    obs = []
    for row in payload.get("observations", []) or []:
        obs.append(
            FredObservation(
                date=str(row.get("date") or ""),
                value=_parse_observation_value(row.get("value")),
                realtime_start=str(row.get("realtime_start") or ""),
                realtime_end=str(row.get("realtime_end") or ""),
            )
        )
    return FredSeriesResponse(
        series_id=series_id,
        realtime_start=str(payload.get("realtime_start") or ""),
        realtime_end=str(payload.get("realtime_end") or ""),
        observation_start=str(payload.get("observation_start") or ""),
        observation_end=str(payload.get("observation_end") or ""),
        count=int(payload.get("count") or len(obs)),
        observations=obs,
    )


def fetch_fred_series(
    series_id: str,
    *,
    start: str | None = None,
    end: str | None = None,
    api_key: str | None = None,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
    transport: httpx.BaseTransport | None = None,
) -> FredSeriesResponse:
    """Fetch a FRED series. Returns cached payload when present unless ``force_refresh=True``.

    The ``transport`` parameter is injected by tests with ``httpx.MockTransport``.
    Production callers leave it ``None`` so httpx uses the default transport.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_path, lock_path = _cache_paths(series_id, cache_dir)

    if cache_path.exists() and not force_refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return _parse_observations(payload, series_id)

    resolved_key = _resolve_api_key(api_key)
    params: dict[str, str] = {
        "series_id": series_id,
        "api_key": resolved_key,
        "file_type": "json",
    }
    if start:
        params["observation_start"] = start
    if end:
        params["observation_end"] = end

    client = httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS, transport=transport)
    try:
        resp = client.get(FRED_BASE_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()
    finally:
        client.close()

    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lock = _read_lock(lock_path)
    lock[series_id] = {
        "sha256": _sha256_of_file(cache_path),
        "observation_start": str(payload.get("observation_start") or ""),
        "observation_end": str(payload.get("observation_end") or ""),
        "count": int(payload.get("count") or 0),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_lock(lock_path, lock)

    return _parse_observations(payload, series_id)
