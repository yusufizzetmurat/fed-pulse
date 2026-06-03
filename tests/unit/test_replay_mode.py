"""Replay-mode (time-machine) coverage.

Covers two surfaces:

1. :func:`app.services.replay.resolve_fold_for_date` — picks the right
   walk-forward fold for a given as-of date, returns a structured
   ``FoldRef.unavailable(...)`` when the manifest / checkpoint is
   missing on disk.

2. The /analyze API path under ``as_of_date`` — emits the ``replay``
   block + ``realised_outcome`` reveal on a happy path, and surfaces a
   422 when the fold is unavailable.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("torch")
pytest.importorskip("transformers")

from fastapi.testclient import TestClient  # noqa: E402

import app.main as main_mod  # noqa: E402
from app.services import replay as replay_service  # noqa: E402


# ---------------------------------------------------------------------------
# resolve_fold_for_date
# ---------------------------------------------------------------------------


def _write_manifest(
    tmp_path: Path, folds: list[dict], checkpoint_dir: Path | None = None
) -> Path:
    if checkpoint_dir is not None:
        for fold in folds:
            fold.setdefault("checkpoint_dir", str(checkpoint_dir))
    manifest = tmp_path / "fold_manifest.json"
    manifest.write_text(
        json.dumps({"training_package_id": "canonical", "folds": folds}),
        encoding="utf-8",
    )
    return manifest


def _seed_fold_checkpoint(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "forecaster_best.pt").write_bytes(b"stub")
    return root


def test_resolve_returns_unavailable_when_manifest_missing(tmp_path):
    ref = replay_service.resolve_fold_for_date(
        date(2024, 1, 5), manifest_path=tmp_path / "missing.json"
    )
    assert ref.available is False
    assert ref.reason == "fold_manifest_missing"


def test_resolve_returns_unavailable_when_no_fold_predates_as_of(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "fold_id": "wf_fold_1",
                "train_end": "2024-12-31",
                "test_start": "2025-01-02",
                "test_end": "2025-06-30",
            }
        ],
    )
    ref = replay_service.resolve_fold_for_date(
        date(2024, 6, 1), manifest_path=manifest
    )
    assert ref.available is False
    assert ref.reason == "no_fold_before_as_of"


def test_resolve_returns_unavailable_when_checkpoint_file_missing(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "fold_id": "wf_fold_1",
                "train_end": "2023-12-31",
                "test_start": "2024-01-02",
                "test_end": "2024-06-30",
                "checkpoint_dir": str(tmp_path / "nope"),
            }
        ],
    )
    ref = replay_service.resolve_fold_for_date(
        date(2024, 3, 1), manifest_path=manifest
    )
    assert ref.available is False
    assert ref.reason == "fold_checkpoint_missing"


def test_resolve_picks_latest_fold_whose_train_end_precedes_as_of(tmp_path):
    ckpt_a = _seed_fold_checkpoint(tmp_path / "fold_a")
    ckpt_b = _seed_fold_checkpoint(tmp_path / "fold_b")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "training_package_id": "canonical",
                "folds": [
                    {
                        "fold_id": "wf_fold_1",
                        "train_end": "2022-12-31",
                        "test_start": "2023-01-02",
                        "test_end": "2023-06-30",
                        "checkpoint_dir": str(ckpt_a),
                    },
                    {
                        "fold_id": "wf_fold_2",
                        "train_end": "2023-06-30",
                        "test_start": "2023-07-03",
                        "test_end": "2023-12-31",
                        "checkpoint_dir": str(ckpt_b),
                    },
                    {
                        "fold_id": "wf_fold_3",
                        "train_end": "2024-06-30",
                        "test_start": "2024-07-01",
                        "test_end": "2024-12-31",
                        "checkpoint_dir": str(tmp_path / "no_ckpt"),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    # ``as_of`` = 2024-01-15 sits inside wf_fold_2's test window; the
    # right serving fold has train_end < 2024-01-15, which is wf_fold_2
    # itself (train_end 2023-06-30, strictly before 2024-01-15).
    ref = replay_service.resolve_fold_for_date(
        date(2024, 1, 15), manifest_path=manifest
    )
    assert ref.available is True
    assert ref.fold_id == "wf_fold_2"
    assert ref.train_end == date(2023, 6, 30)
    assert ref.forecaster_checkpoint == ckpt_b / "forecaster_best.pt"


# ---------------------------------------------------------------------------
# /analyze API path
# ---------------------------------------------------------------------------


def _stub_market_path(monkeypatch):
    monkeypatch.setattr(
        main_mod,
        "analyze_text",
        lambda _: {
            "label": "HAWKISH",
            "score": 0.62,
            "raw": [{"label": "HAWKISH", "score": 0.62}],
        },
    )
    monkeypatch.setattr(
        main_mod,
        "fetch_market_snapshot",
        lambda **_: {
            "symbol": "^GSPC",
            "requested_date": "2024-03-15",
            "date_used": "2024-03-15",
            "lookback_days": 5,
            "close": 5000.0,
            "volatility_5d": 0.01,
        },
    )
    monkeypatch.setattr(
        main_mod,
        "fetch_market_history",
        lambda **_: [
            {"date": "2024-03-12", "close": 4980.0, "volatility_5d": 0.011},
            {"date": "2024-03-13", "close": 5000.0, "volatility_5d": 0.010},
        ],
    )
    monkeypatch.setattr(main_mod, "parse_horizon_steps", lambda _: 3)
    monkeypatch.setattr(
        main_mod,
        "fetch_forward_trading_dates",
        lambda **_: ["2024-03-18", "2024-03-19", "2024-03-20"],
    )
    monkeypatch.setattr(
        main_mod,
        "forecast_quantitative_series",
        lambda **_: {
            "prediction": {"close": 5050.0, "volatility": 0.012, "horizon": "3d"},
            "model": {
                "checkpoint_path": "backend/models/forecaster_best.pt",
                "checkpoint_exists": True,
                "checkpoint_loaded": True,
                "runtime_mode": "fast",
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.15,
                "head_hidden_size": 32,
                "close_scale": 10000.0,
                "sequence_length": 5,
            },
            "series": {
                "timestamps": ["2024-03-12", "2024-03-13"],
                "history_close": [4980.0, 5000.0],
                "history_volatility": [0.011, 0.01],
                "forecast_timestamps": ["2024-03-18", "2024-03-19", "2024-03-20"],
                "forecast_close": [5020.0, 5040.0, 5050.0],
                "forecast_close_lower": [5000.0, 5015.0, 5020.0],
                "forecast_close_upper": [5040.0, 5060.0, 5080.0],
                "forecast_volatility": [0.011, 0.012, 0.012],
                "forecast_volatility_lower": [0.009, 0.010, 0.010],
                "forecast_volatility_upper": [0.013, 0.014, 0.015],
                "forecast_confidence_level": 0.8,
                "volatility_scale": {"suggested_ymin": 0.0, "suggested_ymax": 0.02},
            },
        },
    )
    monkeypatch.setattr(main_mod, "checkpoint_exists", lambda: True)


def test_replay_mode_returns_422_when_per_fold_checkpoints_missing(monkeypatch):
    _stub_market_path(monkeypatch)
    client = TestClient(main_mod.app)

    response = client.post(
        "/analyze",
        json={
            "text": "Recent indicators…",
            "date": "2024-03-15",
            "symbol": "^GSPC",
            "horizon": "3d",
            "include_realized": False,
            "as_of_date": "2024-03-15",
        },
    )
    assert response.status_code == 422
    assert "replay_unavailable" in response.json()["detail"]


def test_replay_mode_emits_replay_and_realised_blocks_when_fold_resolves(
    monkeypatch, tmp_path
):
    _stub_market_path(monkeypatch)
    ckpt = tmp_path / "fold"
    ckpt.mkdir(parents=True)
    (ckpt / "forecaster_best.pt").write_bytes(b"stub")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "training_package_id": "canonical",
                "folds": [
                    {
                        "fold_id": "wf_fold_2",
                        "train_end": "2023-12-31",
                        "test_start": "2024-01-02",
                        "test_end": "2024-06-30",
                        "checkpoint_dir": str(ckpt),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(replay_service, "_DEFAULT_MANIFEST_PATH", manifest)
    monkeypatch.setattr(
        replay_service,
        "realised_outcome",
        lambda as_of, symbol="^GSPC": {
            "as_of_date": as_of.isoformat(),
            "symbol": symbol,
            "horizons": [
                {
                    "horizon": 1,
                    "log_return": 0.01,
                    "realised_volatility_5d": 0.005,
                    "close": 5050.0,
                    "date": "2024-03-18",
                },
                {
                    "horizon": 5,
                    "log_return": 0.02,
                    "realised_volatility_5d": 0.007,
                    "close": 5100.0,
                    "date": "2024-03-22",
                },
                {
                    "horizon": 10,
                    "log_return": None,
                    "realised_volatility_5d": None,
                    "close": None,
                    "date": None,
                },
            ],
        },
    )

    client = TestClient(main_mod.app)
    response = client.post(
        "/analyze",
        json={
            "text": "Recent indicators…",
            "date": "2024-03-15",
            "symbol": "^GSPC",
            "horizon": "3d",
            "include_realized": False,
            "as_of_date": "2024-03-15",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["replay"] is not None
    assert body["replay"]["as_of_date"] == "2024-03-15"
    assert body["replay"]["fold_id"] == "wf_fold_2"
    assert body["replay"]["train_end"] == "2023-12-31"
    assert body["replay"]["classifier_rewind"] is False
    assert any("classifier rewind" in note.lower() for note in body["replay"]["notes"])
    assert body["realised_outcome"] is not None
    horizons = {h["horizon"]: h for h in body["realised_outcome"]["horizons"]}
    assert horizons[1]["log_return"] == pytest.approx(0.01)
    assert horizons[10]["log_return"] is None


def test_live_mode_payload_is_unchanged_when_as_of_date_omitted(monkeypatch):
    _stub_market_path(monkeypatch)
    client = TestClient(main_mod.app)
    response = client.post(
        "/analyze",
        json={
            "text": "Recent indicators…",
            "date": "2026-03-15",
            "symbol": "^GSPC",
            "horizon": "3d",
            "include_realized": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body.get("replay") is None
    assert body.get("realised_outcome") is None
