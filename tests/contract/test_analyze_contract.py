"""Single-mode /analyze contract.

Asserts the synchronous fast-mode response shape: prediction.volatility
is non-negative, series.forecast_volatility is non-negative pointwise,
close-band width is non-decreasing as the horizon advances, and the
full response carries no NaN or infinity in any numeric field.

The deprecated quick_train / real_train branches were retired in #265
Phase 2; the test surface collapsed to a single contract at the same
time as the API.
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

_PAYLOAD = {
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
def response_body(client: TestClient) -> dict[str, Any]:
    """One /analyze call cached for every invariant assertion below."""

    response = client.post("/analyze", json=_PAYLOAD)
    assert response.status_code == 200, response.text
    return response.json()


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


def test_volatility_is_non_negative(response_body: dict[str, Any]) -> None:
    assert response_body["prediction"]["volatility"] >= 0.0
    for value in response_body["series"]["forecast_volatility"]:
        assert value >= 0.0


def test_close_band_width_is_non_decreasing(response_body: dict[str, Any]) -> None:
    series = response_body["series"]
    widths = [
        upper - lower
        for upper, lower in zip(
            series["forecast_close_upper"], series["forecast_close_lower"]
        )
    ]
    for previous, current in zip(widths[:-1], widths[1:]):
        assert current + 1e-9 >= previous, (
            f"close band narrowed from {previous} to {current} "
            "across consecutive horizon steps"
        )


def test_response_has_no_nan(response_body: dict[str, Any]) -> None:
    bad_paths = _walk_check_finite(response_body)
    assert not bad_paths, (
        f"response contains non-finite float at: {bad_paths[:5]}"
    )
