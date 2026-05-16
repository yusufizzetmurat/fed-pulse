"""Tests for /research/artifacts and /train-jobs (multi-page expansion #150)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")
pytest.importorskip("transformers")

from fastapi.testclient import TestClient  # noqa: E402

import app.main as main_mod  # noqa: E402
from app.services import research_artifacts  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_train_jobs(monkeypatch):
    """Each test starts with an empty in-memory train-jobs map and the
    arq Redis pool disabled so the legacy in-memory path is exercised."""

    monkeypatch.setenv("DISABLE_REDIS_POOL", "1")
    if getattr(main_mod.app.state, "redis_pool", None) is not None:
        main_mod.app.state.redis_pool = None
    with main_mod._train_jobs_lock:
        main_mod._train_jobs.clear()
    yield
    with main_mod._train_jobs_lock:
        main_mod._train_jobs.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(main_mod.app)


def _write_phase3_aggregate(root: Path, *, name: str, by_encoder: dict) -> Path:
    target = root / "phase3" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "by_encoder": by_encoder,
        "coverage": 0.95,
        "seed": 11,
        "block_size": 1,
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def _write_cross_bank_matrix(root: Path, *, payload: dict, name: str = "transfer_matrix.json") -> Path:
    target = root / "cross_bank" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def test_research_artifacts_empty_state(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main_mod, "DATA_DIR", tmp_path)
    response = client.get("/research/artifacts")
    assert response.status_code == 200
    body = response.json()
    assert body["artifacts_root"].endswith("artifacts")
    assert body["encoder_bakeoff"]["available"] is False
    assert body["encoder_bakeoff"]["rows"] == []
    assert body["cross_bank_transfer"]["available"] is False
    for section in ("phase3", "cross_bank", "cross_asset", "next_fomc"):
        assert body["sections"][section] == []


def test_research_artifacts_with_bakeoff_and_transfer(
    client, monkeypatch, tmp_path
):
    artifacts_root = tmp_path / "artifacts"
    _write_phase3_aggregate(
        artifacts_root,
        name="run-a/aggregate.json",
        by_encoder={
            "bert-base-uncased": {
                "checkpoint": "bert-base-uncased",
                "per_seed": {
                    "11": {"macro_f1": 0.55, "weighted_f1": 0.58, "accuracy": 0.60},
                    "29": {"macro_f1": 0.57, "weighted_f1": 0.59, "accuracy": 0.61},
                },
                "macro_f1_ci": {"low": 0.52, "high": 0.60},
            },
            "fomc-roberta": {
                "checkpoint": "yiyanghkust/finbert-tone",
                "per_seed": {
                    "11": {"macro_f1": 0.61, "weighted_f1": 0.62, "accuracy": 0.65},
                },
            },
        },
    )
    _write_cross_bank_matrix(
        artifacts_root,
        payload={
            "metric_name": "macro_f1",
            "sources": ["fed", "ecb"],
            "targets": ["fed", "ecb"],
            "cells": [
                {"source": "fed", "target": "fed", "metric": 0.63},
                {"source": "fed", "target": "ecb", "metric": 0.41},
                {"source": "ecb", "target": "fed", "metric": 0.45},
                {"source": "ecb", "target": "ecb", "metric": 0.58},
            ],
        },
    )
    monkeypatch.setattr(main_mod, "DATA_DIR", tmp_path)

    response = client.get("/research/artifacts")
    assert response.status_code == 200
    body = response.json()
    bakeoff = body["encoder_bakeoff"]
    assert bakeoff["available"] is True
    encoder_keys = [r["encoder_key"] for r in bakeoff["rows"]]
    assert encoder_keys == ["bert-base-uncased", "fomc-roberta"]
    bert_row = next(r for r in bakeoff["rows"] if r["encoder_key"] == "bert-base-uncased")
    assert bert_row["macro_f1_values"] == [0.55, 0.57]
    assert pytest.approx(bert_row["macro_f1_mean"], rel=1e-4) == 0.56
    assert bert_row["macro_f1_ci_low"] == 0.52
    assert bert_row["macro_f1_ci_high"] == 0.60

    transfer = body["cross_bank_transfer"]
    assert transfer["available"] is True
    assert transfer["metric_name"] == "macro_f1"
    assert sorted(transfer["sources"]) == ["ecb", "fed"]
    cell_keys = {(c["source"], c["target"]) for c in transfer["cells"]}
    assert ("fed", "ecb") in cell_keys

    # Files are surfaced for the explorer side panel.
    assert any(f["relative_path"].endswith("aggregate.json") for f in body["sections"]["phase3"])
    assert any(
        f["relative_path"].endswith("transfer_matrix.json")
        for f in body["sections"]["cross_bank"]
    )


def test_research_artifacts_accepts_matrix_shape(client, monkeypatch, tmp_path):
    artifacts_root = tmp_path / "artifacts"
    _write_cross_bank_matrix(
        artifacts_root,
        payload={
            "metric_name": "macro_f1",
            "matrix": {"fed": {"fed": 0.63, "ecb": 0.41}, "ecb": {"fed": 0.45, "ecb": 0.58}},
        },
    )
    monkeypatch.setattr(main_mod, "DATA_DIR", tmp_path)

    response = client.get("/research/artifacts")
    body = response.json()
    transfer = body["cross_bank_transfer"]
    assert transfer["available"] is True
    cells = {(c["source"], c["target"]): c["metric"] for c in transfer["cells"]}
    assert cells[("fed", "fed")] == 0.63
    assert cells[("ecb", "ecb")] == 0.58


def test_train_jobs_listing_filters_and_pagination(client):
    now = datetime.now(timezone.utc).isoformat()
    jobs = [
        {"job_id": "a", "status": "running", "symbol": "^GSPC", "created_at": now},
        {"job_id": "b", "status": "queued", "symbol": "^GSPC", "created_at": now},
        {"job_id": "c", "status": "succeeded", "symbol": "QQQ", "created_at": now},
        {"job_id": "d", "status": "failed", "symbol": "^GSPC", "created_at": now, "error": "boom"},
    ]
    with main_mod._train_jobs_lock:
        for job in jobs:
            main_mod._train_jobs[job["job_id"]] = job

    response = client.get("/train-jobs")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 4
    # Running first, then queued, failed, succeeded -- per _TRAIN_JOB_SORT_RANK.
    assert [item["job_id"] for item in body["items"]] == ["a", "b", "d", "c"]

    filtered = client.get("/train-jobs", params={"status": "failed"}).json()
    assert filtered["total"] == 1
    assert filtered["items"][0]["error"] == "boom"

    paged = client.get("/train-jobs", params={"limit": 2, "offset": 1}).json()
    assert paged["limit"] == 2
    assert paged["offset"] == 1
    assert len(paged["items"]) == 2


def test_research_artifacts_section_skips_dotfiles(tmp_path):
    section_dir = tmp_path / "artifacts" / "phase3"
    section_dir.mkdir(parents=True)
    (section_dir / "aggregate.json").write_text("{}", encoding="utf-8")
    (section_dir / ".hidden").write_text("nope", encoding="utf-8")
    infos = research_artifacts.list_section_files(tmp_path / "artifacts", "phase3")
    names = {Path(info.relative_path).name for info in infos}
    assert "aggregate.json" in names
    assert ".hidden" not in names
