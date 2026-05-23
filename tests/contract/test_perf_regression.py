"""Latency regression on /analyze fast-mode (#101 part c).

Locks p50 / p95 of warm-cache fast-mode latency to a baseline checked
into ``tests/snapshots/perf_baseline.json``. Fails when p50 regresses
by more than 20% vs the baseline — the same threshold the issue spec
called for. The test skips when the baseline file is absent so a
fresh checkout doesn't fail before the baseline lands.

To regenerate the baseline after an intentional performance change:

    python tests/contract/test_perf_regression.py --regen

The regen path runs the same workload and writes the snapshot in
place.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

os.environ.setdefault("FED_PULSE_DISABLE_REDIS_POOL", "1")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "snapshots" / "perf_baseline.json"
WARMUP_CALLS = 2
TIMED_CALLS = 8
REGRESSION_THRESHOLD = 1.20  # fail when p50 exceeds baseline × 1.20

_PAYLOAD = {
    "text": (
        "The committee anticipates that the appropriate path of the "
        "federal funds rate will balance the dual mandate as inflation "
        "pressures subside. Recent labor market indicators suggest "
        "moderation; the committee remains data-dependent."
    ),
    "date": "2024-01-31",
    "symbol": "SPY",
    "horizon": "5d",
    "forecast_mode": "fast",
    "include_realized": False,
}


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = max(0, min(len(sorted_values) - 1, math.ceil(pct * len(sorted_values)) - 1))
    return sorted_values[rank]


def _measure(client: TestClient) -> dict[str, float]:
    # Warm-up: first call may bootstrap the checkpoint; we are timing
    # warm-cache latency.
    for _ in range(WARMUP_CALLS):
        response = client.post("/analyze", json=_PAYLOAD)
        assert response.status_code == 200, response.text

    samples: list[float] = []
    for _ in range(TIMED_CALLS):
        start = time.perf_counter()
        response = client.post("/analyze", json=_PAYLOAD)
        elapsed = time.perf_counter() - start
        assert response.status_code == 200, response.text
        samples.append(elapsed)

    return {
        "p50_seconds": statistics.median(samples),
        "p95_seconds": _percentile(samples, 0.95),
        "samples": samples,
    }


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    """TestClient with a tmp checkpoint dir — same redirect rationale as
    ``test_diff_modes.py`` so a stale on-disk head shape cannot crash
    the timed calls."""

    from app.models import config as model_config
    from app.services import forecaster as forecaster_module

    fresh_dir = tmp_path_factory.mktemp("forecaster_models")
    fresh_path = fresh_dir / "forecaster_best.pt"
    original = model_config.BEST_MODEL_PATH
    model_config.BEST_MODEL_PATH = fresh_path
    forecaster_module.BEST_MODEL_PATH = fresh_path
    forecaster_module._model = None
    forecaster_module._model_artifact_metadata = None

    try:
        yield TestClient(app)
    finally:
        model_config.BEST_MODEL_PATH = original
        forecaster_module.BEST_MODEL_PATH = original
        forecaster_module._model = None
        forecaster_module._model_artifact_metadata = None


def test_analyze_fast_mode_latency_within_baseline(client: TestClient) -> None:
    if not SNAPSHOT_PATH.exists():
        pytest.skip(
            f"perf baseline missing at {SNAPSHOT_PATH}; "
            "regenerate via `python tests/contract/test_perf_regression.py --regen`"
        )

    baseline = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    measured = _measure(client)

    ceiling = float(baseline["p50_seconds"]) * REGRESSION_THRESHOLD
    assert measured["p50_seconds"] <= ceiling, (
        f"/analyze fast-mode p50 regressed: "
        f"measured {measured['p50_seconds']:.3f}s, "
        f"baseline {baseline['p50_seconds']:.3f}s × {REGRESSION_THRESHOLD} "
        f"ceiling = {ceiling:.3f}s; samples={measured['samples']}"
    )


def _regenerate() -> int:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Reuse the same client fixture, manually, outside pytest.
    from app.models import config as model_config
    from app.services import forecaster as forecaster_module

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        fresh_path = Path(tmp) / "forecaster_best.pt"
        original = model_config.BEST_MODEL_PATH
        model_config.BEST_MODEL_PATH = fresh_path
        forecaster_module.BEST_MODEL_PATH = fresh_path
        forecaster_module._model = None
        forecaster_module._model_artifact_metadata = None
        try:
            client = TestClient(app)
            measured = _measure(client)
        finally:
            model_config.BEST_MODEL_PATH = original
            forecaster_module.BEST_MODEL_PATH = original
            forecaster_module._model = None
            forecaster_module._model_artifact_metadata = None

    SNAPSHOT_PATH.write_text(
        json.dumps(
            {
                "p50_seconds": measured["p50_seconds"],
                "p95_seconds": measured["p95_seconds"],
                "warmup_calls": WARMUP_CALLS,
                "timed_calls": TIMED_CALLS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote baseline → {SNAPSHOT_PATH}")
    print(f"  p50 = {measured['p50_seconds']:.3f}s")
    print(f"  p95 = {measured['p95_seconds']:.3f}s")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--regen":
        raise SystemExit(_regenerate())
    raise SystemExit("Use --regen to recompute the perf baseline.")
