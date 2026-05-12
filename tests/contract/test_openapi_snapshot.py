from __future__ import annotations

import json
from pathlib import Path

import pytest

SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "snapshots" / "openapi.json"


@pytest.fixture(scope="module")
def live_schema() -> dict:
    pytest.importorskip("fastapi")
    pytest.importorskip("pydantic")
    from app.main import app

    return app.openapi()


def _strip_volatile_fields(schema: dict) -> dict:
    schema = dict(schema)
    info = dict(schema.get("info", {}))
    info.pop("version", None)
    schema["info"] = info
    return schema


def test_openapi_snapshot_matches(live_schema: dict) -> None:
    if not SNAPSHOT_PATH.exists():
        pytest.skip(f"snapshot missing at {SNAPSHOT_PATH}; run `make openapi-snapshot` to create it")
    stored = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert _strip_volatile_fields(live_schema) == _strip_volatile_fields(stored), (
        "OpenAPI schema drifted from snapshot. If the change is intentional, regenerate via "
        "`python -m tests.contract.regen_openapi_snapshot` and commit the result."
    )
