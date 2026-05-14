from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from app.data.build_training_package import (
    _build_folds,
    _source_distribution,
    _source_drift_max,
    _source_shares,
)


def _write_quality_passed_fixture(target: Path) -> None:
    rows = []
    for idx in range(40):
        rows.append(
            {
                "record_id": f"r{idx:03d}",
                "source": "scraped_fed",
                "source_record_id": f"src:{idx}",
                "source_type": "fomc_minutes" if idx % 2 == 0 else "fomc_statement",
                "document_type": "minutes" if idx % 2 == 0 else "statement",
                "event_date": f"2024-{(idx % 12) + 1:02d}-15",
                "title": f"FOMC doc {idx}",
                "text": f"Document body {idx} hawkish dovish",
                "text_hash": f"h{idx:03d}",
                "label": "hawkish" if idx % 3 == 0 else "dovish",
                "label_origin": "human",
                "license_scope": "public_source_scrape_terms_required",
                "citation_ref": "federalreserve_primary_source",
                "ingested_at_utc": "2024-01-01T00:00:00+00:00",
                "mapped_label": "hawkish" if idx % 3 == 0 else "dovish",
                "label_map_version": "label_map_v1.0",
                "label_taxonomy": "hawkish_dovish_neutral",
            }
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_training_package_metadata_includes_source_type_counts(tmp_path: Path) -> None:
    input_path = tmp_path / "registry_quality_passed.jsonl"
    _write_quality_passed_fixture(input_path)

    cmd = [
        "python",
        "-m",
        "app.data.build_training_package",
        "--input",
        str(input_path),
        "--quality-report-dir",
        str(tmp_path / "quality_reports"),
        "--processed-root",
        str(tmp_path / "processed"),
        "--dataset-version",
        "test_ds_v0",
        "--feature-version",
        "test_fv_v0",
        "--training-package-id",
        "tp_test_v0",
    ]
    repo_root = Path(__file__).resolve().parents[2]
    backend_root = repo_root / "backend"
    cwd = "/app" if Path("/app").exists() else str(backend_root)
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=cwd)
    assert result.returncode == 0, result.stderr

    metadata_path = tmp_path / "processed" / "tp_test_v0" / "dataset_metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert "source_type_counts" in payload
    assert payload["source_type_counts"]["fomc_minutes"] > 0
    assert payload["source_type_counts"]["fomc_statement"] > 0


def test_source_distribution_and_shares_round_trip() -> None:
    rows = [
        {"source": "trillion_dollar_words"},
        {"source": "trillion_dollar_words"},
        {"source": "op_fed"},
        {"source": ""},  # ignored — empty source
        {},  # ignored — missing source
    ]
    dist = _source_distribution(rows)
    assert dist == {"trillion_dollar_words": 2, "op_fed": 1}
    shares = _source_shares(dist)
    assert shares["trillion_dollar_words"] == pytest.approx(2 / 3)
    assert shares["op_fed"] == pytest.approx(1 / 3)


def test_source_drift_max_is_zero_when_distributions_match() -> None:
    train = {"trillion_dollar_words": 70, "op_fed": 30}
    val = {"trillion_dollar_words": 7, "op_fed": 3}
    assert _source_drift_max(train, val) == pytest.approx(0.0)


def test_source_drift_max_measures_share_point_gap() -> None:
    train = {"trillion_dollar_words": 100, "op_fed": 0}  # 100% TDW
    val = {"trillion_dollar_words": 50, "op_fed": 50}  # 50% / 50%
    # On the TDW key the gap is 1.00 - 0.50 = 0.50; symmetric on op_fed.
    assert _source_drift_max(train, val) == pytest.approx(0.5)


def test_source_drift_max_returns_zero_when_either_side_is_empty() -> None:
    assert _source_drift_max({}, {"x": 1}) == 0.0
    assert _source_drift_max({"x": 1}, {}) == 0.0
    assert _source_drift_max({}, {}) == 0.0


def test_build_folds_persists_source_distribution_and_drift_per_fold() -> None:
    # Build 12 dates × mixed sources so we can drive a multi-fold walk-forward.
    sources = [
        "trillion_dollar_words",
        "trillion_dollar_words",
        "op_fed",
        "op_fed",
    ]
    rows = []
    for idx in range(24):
        rows.append(
            {
                "record_id": f"r{idx:03d}",
                "source": sources[idx % len(sources)],
                "event_date": f"2024-{(idx // 2) + 1:02d}-15",
                "mapped_label": "hawkish" if idx % 2 == 0 else "dovish",
            }
        )
    folds = _build_folds(rows, min_train_ratio=0.5, fold_count=3)
    assert folds, "expected at least one fold to be built"
    for fold in folds:
        assert fold.train_source_distribution  # populated
        assert isinstance(fold.source_drift_max, float)
        assert 0.0 <= fold.source_drift_max <= 1.0


def test_build_training_package_emits_source_drift_metadata_and_per_fold_distributions(
    tmp_path: Path,
) -> None:
    """End-to-end: registry with mixed sources should produce fold-level source
    distributions in fold_manifest_*.json and a drift summary in dataset_metadata.json."""

    input_path = tmp_path / "registry_quality_passed.jsonl"
    rows = []
    for idx in range(60):
        # Heavy TDW concentration in first half of the year, Op-Fed in the second.
        # Walk-forward folds will see real drift between splits.
        month = (idx % 12) + 1
        src = "trillion_dollar_words" if month <= 6 else "op_fed"
        rows.append(
            {
                "record_id": f"r{idx:03d}",
                "source": src,
                "source_record_id": f"src:{idx}",
                "source_type": "fomc_minutes",
                "document_type": "minutes",
                "event_date": f"2024-{month:02d}-{(idx % 28) + 1:02d}",
                "title": f"doc {idx}",
                "text": f"body {idx}",
                "text_hash": f"h{idx:03d}",
                "label": "hawkish" if idx % 3 == 0 else "dovish",
                "label_origin": "human",
                "license_scope": "public_source_scrape_terms_required",
                "citation_ref": "federalreserve_primary_source",
                "ingested_at_utc": "2024-01-01T00:00:00+00:00",
                "mapped_label": "hawkish" if idx % 3 == 0 else "dovish",
                "label_map_version": "label_map_v1.0",
                "label_taxonomy": "hawkish_dovish_neutral",
            }
        )
    input_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    repo_root = Path(__file__).resolve().parents[2]
    backend_root = repo_root / "backend"
    cwd = "/app" if Path("/app").exists() else str(backend_root)
    cmd = [
        "python",
        "-m",
        "app.data.build_training_package",
        "--input",
        str(input_path),
        "--quality-report-dir",
        str(tmp_path / "quality_reports"),
        "--processed-root",
        str(tmp_path / "processed"),
        "--dataset-version",
        "stratify_ds_v0",
        "--feature-version",
        "stratify_fv_v0",
        "--training-package-id",
        "tp_stratify_v0",
        "--fold-count",
        "3",
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=cwd)
    assert result.returncode == 0, result.stderr

    package = tmp_path / "processed" / "tp_stratify_v0"
    fold_manifest = json.loads(
        (package / "fold_manifest_expanding_walk_forward.json").read_text(encoding="utf-8")
    )
    assert fold_manifest["source_drift_tolerance"] == 0.0
    for fold in fold_manifest["folds"]:
        assert "train_source_distribution" in fold
        assert "val_source_distribution" in fold
        assert "test_source_distribution" in fold
        assert "source_drift_max" in fold

    metadata = json.loads((package / "dataset_metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_drift_tolerance"] == 0.0
    assert "source_drift_per_fold" in metadata
    assert metadata["source_drift_max_across_folds"] >= 0.0
    assert metadata["source_drift_max_across_folds"] <= 1.0


def test_build_training_package_aborts_when_drift_tolerance_exceeded(tmp_path: Path) -> None:
    """A pathological mix where each fold's val/test slice is 100% from a different
    source than train should trip the tolerance check and exit non-zero."""

    input_path = tmp_path / "registry_quality_passed.jsonl"
    rows = []
    for idx in range(60):
        month = (idx % 12) + 1
        src = "trillion_dollar_words" if month <= 6 else "op_fed"
        rows.append(
            {
                "record_id": f"r{idx:03d}",
                "source": src,
                "source_record_id": f"src:{idx}",
                "source_type": "fomc_minutes",
                "document_type": "minutes",
                "event_date": f"2024-{month:02d}-{(idx % 28) + 1:02d}",
                "title": f"doc {idx}",
                "text": f"body {idx}",
                "text_hash": f"h{idx:03d}",
                "label": "hawkish" if idx % 3 == 0 else "dovish",
                "label_origin": "human",
                "license_scope": "public_source_scrape_terms_required",
                "citation_ref": "federalreserve_primary_source",
                "ingested_at_utc": "2024-01-01T00:00:00+00:00",
                "mapped_label": "hawkish" if idx % 3 == 0 else "dovish",
                "label_map_version": "label_map_v1.0",
                "label_taxonomy": "hawkish_dovish_neutral",
            }
        )
    input_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    repo_root = Path(__file__).resolve().parents[2]
    backend_root = repo_root / "backend"
    cwd = "/app" if Path("/app").exists() else str(backend_root)
    cmd = [
        "python",
        "-m",
        "app.data.build_training_package",
        "--input",
        str(input_path),
        "--quality-report-dir",
        str(tmp_path / "quality_reports"),
        "--processed-root",
        str(tmp_path / "processed"),
        "--dataset-version",
        "strict_ds_v0",
        "--feature-version",
        "strict_fv_v0",
        "--training-package-id",
        "tp_strict_v0",
        "--fold-count",
        "3",
        "--source-drift-tolerance",
        "0.05",  # tight gate, the synthetic fixture should trip it
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=cwd)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "source-drift" in result.stdout
