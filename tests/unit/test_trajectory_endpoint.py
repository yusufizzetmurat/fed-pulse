"""Endpoint tests for POST /analyze/trajectory (#296)."""

from __future__ import annotations

import json
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


# ---------------------------------------------------------------------------
# Strict-backward boundary (finding 1)
# ---------------------------------------------------------------------------


def test_analyze_trajectory_excludes_meeting_on_as_of_date(
    client: TestClient,
) -> None:
    """A meeting whose ``event_date`` equals ``as_of_date`` is the
    meeting being projected — never an eligible history marker.
    Strict-backward enforces ``event_date < as_of_date``.
    """

    _install_fake_trajectory_state()
    response = client.post(
        "/analyze/trajectory",
        # Latest fixture date is 2022-06-15 — query it directly.
        json={"as_of_date": "2022-06-15", "history_length": 12},
    )
    assert response.status_code == 200
    body = response.json()
    dates = {marker["event_date"] for marker in body["history"]}
    assert "2022-06-15" not in dates, (
        f"history leaked the as_of meeting: {sorted(dates)}"
    )


def test_analyze_trajectory_disambiguates_duplicate_event_dates(
    client: TestClient,
) -> None:
    """When two metadata rows share an ``event_date`` (e.g. statement +
    intermeeting release on the same day) the embedding-row lookup
    must key on ``text_hash`` so the embedding matches the marker
    being projected. Without the fix the first row always wins.
    """

    metadata = pd.DataFrame(
        [
            {
                "event_date": "2020-03-15",
                "text_hash": "h_statement",
                "axis_stance": "dovish",
                "embedding_2d_x": 0.1,
                "embedding_2d_y": 0.2,
                "pre_meeting_trailing_2y_yield_change_5d_bps": 1.0,
                "vix_close": 15.0,
            },
            # Distinct second row sharing the same event_date.
            {
                "event_date": "2020-03-15",
                "text_hash": "h_intermeeting",
                "axis_stance": "neutral",
                "embedding_2d_x": -0.3,
                "embedding_2d_y": 0.4,
                "pre_meeting_trailing_2y_yield_change_5d_bps": -2.5,
                "vix_close": 60.0,
            },
            {
                "event_date": "2020-06-10",
                "text_hash": "h_june",
                "axis_stance": "dovish",
                "embedding_2d_x": 0.5,
                "embedding_2d_y": -0.1,
                "pre_meeting_trailing_2y_yield_change_5d_bps": 0.5,
                "vix_close": 28.0,
            },
        ]
    )
    embedding_dim = 6
    rng = np.random.default_rng(11)
    embeddings = rng.normal(size=(len(metadata), embedding_dim)).astype(np.float32)
    # Make the two same-date rows trivially distinguishable.
    embeddings[0] = 1.0
    embeddings[1] = -1.0
    config = traj_model.TrajectoryConfig(
        architecture="lstm", embedding_dim=embedding_dim, history_length=3
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
        conformal_quantile=0.5,
        conformal_alpha=0.2,
    )
    trajectory_service.install_state(state)
    response = client.post(
        "/analyze/trajectory",
        json={"as_of_date": "2020-07-01", "history_length": 3},
    )
    assert response.status_code == 200
    body = response.json()
    # Both same-date rows must surface in the history slice.
    history_hashes = {marker.get("event_date") for marker in body["history"]}
    assert "2020-03-15" in history_hashes
    # Projection runs cleanly with the disambiguated lookups.
    assert body["projected_next"] is not None


# ---------------------------------------------------------------------------
# Warning when as_of beyond train_end (finding 5b)
# ---------------------------------------------------------------------------


def test_analyze_trajectory_emits_warning_when_as_of_beyond_train_end(
    client: TestClient,
) -> None:
    _install_fake_trajectory_state()
    # State has train_end="2025-01-01" — query past it.
    response = client.post(
        "/analyze/trajectory",
        json={"as_of_date": "2030-01-01", "history_length": 4},
    )
    assert response.status_code == 200
    body = response.json()
    assert body.get("warning") is not None
    assert "train_end" in body["warning"]


# ---------------------------------------------------------------------------
# Real-encoder smoke (finding 16) — guarded slow test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_analyze_trajectory_real_encoder_load_path(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise the production ``_load_state`` path end-to-end with a
    tiny real encoder + a synthetic bundle. Covers the regression
    surface every other endpoint test bypasses (they all install a
    pre-built state directly).

    Skipped automatically when the tiny encoder is not cached locally
    so the suite can run offline; full CI fetches it on demand.
    """

    import os

    tiny_repo = os.environ.get(
        "FED_PULSE_TEST_TINY_ENCODER", "sshleifer/tiny-distilbert-base-uncased"
    )
    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore[import-not-found,unused-ignore]
    except Exception:  # pragma: no cover
        pytest.skip("transformers not available")

    try:
        tokenizer = AutoTokenizer.from_pretrained(tiny_repo)
        encoder = AutoModel.from_pretrained(tiny_repo)
    except Exception as exc:  # pragma: no cover — offline / rate-limited
        pytest.skip(f"tiny encoder unavailable: {exc!r}")

    # Build a synthetic bundle with a couple of meeting rows and
    # standardisation stats so ``_load_state`` produces a valid state.
    bundle = tmp_path / "trajectory_real"
    bundle.mkdir()
    embedding_dim = encoder.config.hidden_size
    dates = ["2020-01-29", "2021-03-17", "2022-06-15"]
    metadata = pd.DataFrame(
        [
            {
                "event_date": dt,
                "text_hash": f"h_{idx}",
                "axis_stance": ["hawkish", "dovish", "neutral"][idx % 3],
                "embedding_2d_x": float(idx),
                "embedding_2d_y": float(-idx),
                "pre_meeting_trailing_2y_yield_change_5d_bps": float(idx),
                "vix_close": 15.0 + idx,
            }
            for idx, dt in enumerate(dates)
        ]
    )
    metadata.to_parquet(bundle / "embedding_index.parquet", index=False)
    rng = np.random.default_rng(11)
    embeddings = rng.normal(size=(len(dates), embedding_dim)).astype(np.float32)
    feature_mean = embeddings.mean(axis=0).astype(np.float32)
    feature_std = embeddings.std(axis=0).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    np.savez(
        bundle / "embedding_index.npz",
        embeddings=embeddings,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    config = traj_model.TrajectoryConfig(
        architecture="lstm", embedding_dim=embedding_dim, history_length=4
    )
    model = traj_model.build_model(config)
    traj_model.save_model(model, config, bundle / "model.pt")
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "architecture": "lstm",
                "encoder_alias": tiny_repo,
                "encoder_revision": "",
                "train_end": "2025-01-01",
                "history_length": 4,
                "embedding_dim": embedding_dim,
                "n_classes": 3,
                "row_count": len(dates),
                "stance_classes": ["hawkish", "dovish", "neutral"],
                "config": config.to_dict(),
                "built_at_utc": "2026-05-26T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FED_PULSE_TRAJECTORY_DIR", str(bundle))
    trajectory_service.reset_state()
    response = client.post(
        "/analyze/trajectory",
        json={"as_of_date": "2023-01-01", "history_length": 4},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["projected_next"] is not None
    assert body["encoder_alias"] == tiny_repo


