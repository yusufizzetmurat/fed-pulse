"""Tests for the training-package manifest sidecar (#97 alternative)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from app.data.manifest_sha import (
    ManifestShaMismatch,
    compute_manifest_sha,
    verify_manifest_sha,
    write_manifest_sha,
)


def _make_package(tmp_path: Path) -> Path:
    package_dir = tmp_path / "tp_test_v1_v1"
    package_dir.mkdir()
    (package_dir / "dataset_metadata.json").write_text(
        json.dumps({"training_package_id": "tp_test_v1_v1", "input_rows": 42}),
        encoding="utf-8",
    )
    return package_dir


def test_write_then_verify_roundtrips(tmp_path: Path) -> None:
    package_dir = _make_package(tmp_path)
    digest = write_manifest_sha(package_dir)

    assert len(digest) == 64  # SHA-256 hex length
    assert (package_dir / "dataset_metadata.sha256").exists()
    assert verify_manifest_sha(package_dir) is True


def test_verify_detects_tampered_manifest(tmp_path: Path) -> None:
    package_dir = _make_package(tmp_path)
    write_manifest_sha(package_dir)

    # Silently replace the manifest contents (this is the scenario the
    # benchmark policy forbids — same name, different contents).
    (package_dir / "dataset_metadata.json").write_text(
        json.dumps({"training_package_id": "tp_test_v1_v1", "input_rows": 99}),
        encoding="utf-8",
    )

    with pytest.raises(ManifestShaMismatch) as exc:
        verify_manifest_sha(package_dir)
    assert "mismatch" in str(exc.value).lower()


def test_verify_returns_false_and_warns_when_sidecar_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    package_dir = _make_package(tmp_path)
    # No write_manifest_sha call → no sidecar.

    with caplog.at_level(logging.WARNING):
        assert verify_manifest_sha(package_dir) is False
    assert any(
        "manifest_sha_sidecar_missing" in record.message
        for record in caplog.records
    )


def test_compute_raises_when_manifest_missing(tmp_path: Path) -> None:
    package_dir = tmp_path / "empty_pkg"
    package_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        compute_manifest_sha(package_dir)


def test_write_is_idempotent_for_unchanged_manifest(tmp_path: Path) -> None:
    package_dir = _make_package(tmp_path)

    first = write_manifest_sha(package_dir)
    second = write_manifest_sha(package_dir)
    assert first == second
