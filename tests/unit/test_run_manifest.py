from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def test_write_run_manifest_creates_file(tmp_path: Path) -> None:
    from app.training.manifest import MANIFEST_FILENAME, write_run_manifest

    out = write_run_manifest(
        tmp_path,
        run_id="run_test_seed11",
        version_ids={"model_version": "mv_test_v1", "training_package_id": "tp_test"},
        seeds=[11],
        hyperparameters={"epochs": 3, "lr": 2e-5},
    )
    assert out == tmp_path / MANIFEST_FILENAME
    assert out.exists()


def test_manifest_payload_contains_required_keys(tmp_path: Path) -> None:
    from app.training.manifest import MANIFEST_VERSION, write_run_manifest

    out = write_run_manifest(
        tmp_path,
        run_id="run_payload",
        seeds=[11, 29],
        hyperparameters={"batch_size": 16},
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["manifest_version"] == MANIFEST_VERSION
    assert payload["run_id"] == "run_payload"
    assert payload["hostname"]
    assert payload["python_version"]
    assert payload["platform"]
    assert "cli_argv" in payload
    assert payload["seeds"] == [11, 29]
    assert payload["hyperparameters"]["batch_size"] == 16
    assert "library_versions" in payload
    assert "input_sha256" in payload


def test_manifest_hashes_existing_input_file(tmp_path: Path) -> None:
    from app.training.manifest import write_run_manifest

    payload_file = tmp_path / "fixture.txt"
    payload_file.write_text("fed-pulse-test", encoding="utf-8")
    expected = hashlib.sha256(b"fed-pulse-test").hexdigest()

    out = write_run_manifest(
        tmp_path,
        run_id="run_with_inputs",
        inputs=[payload_file],
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["input_sha256"][str(payload_file)] == expected


def test_manifest_skips_missing_input_files(tmp_path: Path) -> None:
    from app.training.manifest import write_run_manifest

    missing = tmp_path / "not-here.parquet"
    out = write_run_manifest(
        tmp_path,
        run_id="run_missing_input",
        inputs=[missing],
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["input_sha256"] == {}


def test_manifest_extra_field_round_trips(tmp_path: Path) -> None:
    from app.training.manifest import write_run_manifest

    out = write_run_manifest(
        tmp_path,
        run_id="run_extra",
        extra={"note": "smoke run", "owner": "examiner"},
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["extra"] == {"note": "smoke run", "owner": "examiner"}
