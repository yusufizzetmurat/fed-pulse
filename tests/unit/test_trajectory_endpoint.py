"""Endpoint tests for POST /analyze/trajectory (#296)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")

from fastapi.testclient import TestClient  # noqa: E402

import app.main as main_mod  # noqa: E402
from app.services import trajectory as trajectory_service  # noqa: E402
from app.trajectory import model as traj_model  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(main_mod.app)


@pytest.fixture(autouse=True)
def _reset_trajectory_singleton():
    trajectory_service.reset_state()
    yield
    trajectory_service.reset_state()


def _toy_metadata(dates: list[str]) -> pd.DataFrame:
    rows = []
    stance_cycle = ["dovish", "hawkish", "neutral"]
    for idx, dt in enumerate(dates):
        rows.append(
            {
                "event_date": dt,
                "axis_stance": stance_cycle[idx % 3],
                "embedding_2d_x": float(idx) * 0.1,
                "embedding_2d_y": float(idx) * -0.2,
                "pre_meeting_trailing_2y_yield_change_5d_bps": float(idx),
                "vix_close": 15.0 + float(idx),
            }
        )
    return pd.DataFrame(rows)


def _install_fake_trajectory_state() -> None:
    dates = [
        "2010-03-16",
        "2012-06-20",
        "2015-03-18",
        "2017-06-13",
        "2019-07-30",
        "2022-06-15",
    ]
    metadata = _toy_metadata(dates)
    embedding_dim = 6
    rng = np.random.default_rng(11)
    embeddings = rng.normal(size=(len(dates), embedding_dim)).astype(np.float32)
    config = traj_model.TrajectoryConfig(
        architecture="lstm", embedding_dim=embedding_dim, history_length=4
    )
    model = traj_model.build_model(config)
    state = trajectory_service.build_state_for_tests(
        model=model,
        config=config,
        embeddings=embeddings,
        metadata=metadata,
        encoder_alias="test_trajectory_encoder",
        train_end="2025-01-01",
        architecture="lstm",
        conformal_quantile=0.7,
        conformal_alpha=0.2,
    )
    trajectory_service.install_state(state)


def test_analyze_trajectory_returns_empty_when_bundle_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FED_PULSE_TRAJECTORY_DIR", str(tmp_path / "does-not-exist"))
    response = client.post(
        "/analyze/trajectory",
        json={"as_of_date": "2024-01-01", "history_length": 12},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["history"] == []
    assert body["projected_next"] is None
    assert body["encoder_alias"] == "finbert_fed_adjacent_xbank_dapt"


def test_analyze_trajectory_returns_history_and_projection(
    client: TestClient,
) -> None:
    _install_fake_trajectory_state()
    response = client.post(
        "/analyze/trajectory",
        json={"as_of_date": "2023-01-01", "history_length": 4},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["architecture"] == "lstm"
    assert body["encoder_alias"] == "test_trajectory_encoder"
    history = body["history"]
    assert 1 <= len(history) <= 4
    # Strictly backward: every history marker is on or before as_of.
    for marker in history:
        assert marker["event_date"] <= "2023-01-01"
        assert isinstance(marker["embedding_2d"], list)
        assert len(marker["embedding_2d"]) == 2
    projection = body["projected_next"]
    assert projection is not None
    assert projection["predicted_stance"] in {"hawkish", "dovish", "neutral"}
    probs = projection["class_probs"]
    assert set(probs.keys()) == {"hawkish", "dovish", "neutral"}
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)


def test_analyze_trajectory_confidence_band_present_when_conformal_quantile_set(
    client: TestClient,
) -> None:
    _install_fake_trajectory_state()
    response = client.post(
        "/analyze/trajectory",
        json={"as_of_date": "2023-01-01", "history_length": 6},
    )
    assert response.status_code == 200
    band = response.json()["projected_next"]["confidence_band"]
    # A conformal_quantile of 0.7 → keep classes with prob >= 0.3, plus
    # the argmax-fallback. Either way the band is a non-empty list of
    # stance labels.
    assert isinstance(band, list)
    assert all(label in {"hawkish", "dovish", "neutral"} for label in band)


def test_analyze_trajectory_503_does_not_leak_internal_errors(
    client: TestClient,
) -> None:
    _install_fake_trajectory_state()

    secret_path = "/internal/secret/trajectory/path"

    def _explode(*_args, **_kwargs):
        raise RuntimeError(f"boom at {secret_path}")

    # Replace the projection callable on the imported module so the
    # endpoint hits the sanitised exception arm.
    original = trajectory_service.project_trajectory
    try:
        trajectory_service.project_trajectory = _explode  # type: ignore[assignment]
        response = client.post(
            "/analyze/trajectory",
            json={"as_of_date": "2023-01-01", "history_length": 4},
        )
        assert response.status_code == 503
        body = response.json()
        assert body.get("detail") == "Trajectory projection unavailable"
        assert secret_path not in response.text
        assert "RuntimeError" not in response.text
        assert "Traceback" not in response.text
    finally:
        trajectory_service.project_trajectory = original  # type: ignore[assignment]


def test_analyze_trajectory_validates_history_length_lower_bound(client: TestClient) -> None:
    response = client.post(
        "/analyze/trajectory",
        json={"as_of_date": "2023-01-01", "history_length": 0},
    )
    assert response.status_code == 422


def test_analyze_trajectory_validates_history_length_upper_bound(client: TestClient) -> None:
    response = client.post(
        "/analyze/trajectory",
        json={"as_of_date": "2023-01-01", "history_length": 999},
    )
    assert response.status_code == 422


def test_analyze_trajectory_validates_missing_as_of_date(client: TestClient) -> None:
    response = client.post(
        "/analyze/trajectory",
        json={"history_length": 12},
    )
    assert response.status_code == 422
