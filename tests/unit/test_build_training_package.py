from __future__ import annotations

import json
import subprocess
from pathlib import Path


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
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd="/app")
    assert result.returncode == 0, result.stderr

    metadata_path = tmp_path / "processed" / "tp_test_v0" / "dataset_metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert "source_type_counts" in payload
    assert payload["source_type_counts"]["fomc_minutes"] > 0
    assert payload["source_type_counts"]["fomc_statement"] > 0
