"""Differential contract across /analyze forecast_mode (#101 part b).

For the same input, every synchronous mode (`fast`, `quick_train`) must
produce a response whose:

- ``prediction.volatility`` is non-negative,
- ``series.forecast_volatility`` is non-negative pointwise,
- close-band width (``upper - lower``) is non-decreasing as the
  horizon advances — uncertainty cannot shrink with time,
- response contains no NaN or infinity in any numeric field.

`real_train` is async and returns ``TrainJobAcceptedResponse``; the
test asserts the queue accepted the request and returns a job id.
"""

from __future__ import annotations

import math
import os
from typing import Any

import pytest

pytest.importorskip("fastapi")

os.environ.setdefault("FED_PULSE_DISABLE_REDIS_POOL", "1")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

_BASE_PAYLOAD = {
    "text": (
        "The committee continues to anticipate that the appropriate path "
        "of the federal funds rate will balance the dual mandate as "
        "inflation pressures subside. Recent labor market indicators "
        "suggest moderation; the committee remains data-dependent."
    ),
    "date": "2024-01-31",
    "symbol": "SPY",
    "horizon": "5d",
    "include_realized": False,
}


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    """TestClient with the model singleton redirected to an empty path.

    The repo's local ``backend/models/forecaster_best.pt`` is whatever
    the most recent training run produced — its head shape may not
    match the current ``ForecasterModel`` definition. Rebinding
    ``BEST_MODEL_PATH`` inside ``app.services.forecaster`` (and
    resetting the cached singleton) sends the next ``_get_model()``
    call to a non-existent file; the loader falls through to a
    default-init model rather than raising on a head-shape mismatch.

    This does NOT trigger the cold-start bootstrap path — that branch
    keys off ``checkpoint_exists()`` whose default arg is bound at
    import time and is not reached from here. A fresh-init model is
    sufficient for shape-invariant contract tests; the goal is a
    deterministic, crash-free inference path, not a fitted forecast.
    """

    from app.services import forecaster as forecaster_module

    fresh_dir = tmp_path_factory.mktemp("forecaster_models")
    fresh_path = fresh_dir / "forecaster_best.pt"

    original = forecaster_module.BEST_MODEL_PATH
    forecaster_module.BEST_MODEL_PATH = fresh_path
    forecaster_module._model = None
    forecaster_module._model_artifact_metadata = None

    try:
        yield TestClient(app)
    finally:
        forecaster_module.BEST_MODEL_PATH = original
        forecaster_module._model = None
        forecaster_module._model_artifact_metadata = None


@pytest.fixture(scope="module")
def responses(client: TestClient) -> dict[str, dict[str, Any]]:
    """Call /analyze once per sync mode and cache the response.

    Each invariant assertion below reads from this cache instead of
    re-POSTing — ``quick_train`` runs the adaptation training on every
    call, so a per-test post-and-train would multiply CI time and add
    flake surface.
    """

    cached: dict[str, dict[str, Any]] = {}
    for mode in ("fast", "quick_train"):
        response = client.post("/analyze", json={**_BASE_PAYLOAD, "forecast_mode": mode})
        assert response.status_code == 200, response.text
        cached[mode] = response.json()
    return cached


def _walk_check_finite(payload: Any, path: str = "$") -> list[str]:
    """Return a list of paths whose float value is NaN or infinite."""

    findings: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            findings.extend(_walk_check_finite(value, f"{path}.{key}"))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            findings.extend(_walk_check_finite(value, f"{path}[{index}]"))
    elif isinstance(payload, float) and not math.isfinite(payload):
        findings.append(path)
    return findings


@pytest.mark.parametrize("mode", ["fast", "quick_train"])
def test_sync_mode_volatility_is_non_negative(
    responses: dict[str, dict[str, Any]], mode: str
) -> None:
    body = responses[mode]
    assert body["prediction"]["volatility"] >= 0.0
    for value in body["series"]["forecast_volatility"]:
        assert value >= 0.0


@pytest.mark.parametrize("mode", ["fast", "quick_train"])
def test_sync_mode_band_width_is_non_decreasing(
    responses: dict[str, dict[str, Any]], mode: str
) -> None:
    series = responses[mode]["series"]
    widths = [
        upper - lower
        for upper, lower in zip(
            series["forecast_close_upper"], series["forecast_close_lower"]
        )
    ]
    # Uncertainty must not shrink as the horizon advances. A small
    # numerical wobble is tolerated (1e-9) but a real reduction is not.
    for previous, current in zip(widths[:-1], widths[1:]):
        assert current + 1e-9 >= previous, (
            f"{mode}: close band narrowed from {previous} to {current} "
            "across consecutive horizon steps"
        )


@pytest.mark.parametrize("mode", ["fast", "quick_train"])
def test_sync_mode_response_has_no_nan(
    responses: dict[str, dict[str, Any]], mode: str
) -> None:
    bad_paths = _walk_check_finite(responses[mode])
    assert not bad_paths, (
        f"{mode}: response contains non-finite float at: {bad_paths[:5]}"
    )


def test_real_train_mode_enqueues_job(client: TestClient) -> None:
    response = client.post(
        "/analyze", json={**_BASE_PAYLOAD, "forecast_mode": "real_train"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body.get("status") == "queued"
    assert isinstance(body.get("job_id"), str) and body["job_id"]
