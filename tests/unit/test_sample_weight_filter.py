"""Regression test: cross-bank rows (sample_weight=0) must never enter NLP training.

The cross-bank ingestors tag rows with ``sample_weight=0`` (in
``data/schema/labels.yaml``). The fine-tune pilot's ``_load_registry_rows``
honours that flag and drops zero-weight rows by default; the cross-CB eval
harness opts back in via ``include_zero_weight=True``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.data.finetune_pilot import _load_registry_rows


def _write_registry(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


@pytest.fixture
def mixed_registry(tmp_path: Path) -> Path:
    package_dir = tmp_path / "processed" / "tp_v1"
    _write_registry(
        package_dir / "registry_normalized.jsonl",
        [
            {
                "record_id": "fomc_1",
                "text": "the committee judges policy appropriate",
                "event_date": "2024-01-01",
                "mapped_label": "neutral",
                "source": "hf_fomc_communication",
                "provenance": "peer_reviewed",
                "sample_weight": 1.0,
            },
            {
                "record_id": "ecb_1",
                "text": "rate path firm",
                "event_date": "2024-02-01",
                "mapped_label": "hawkish",
                "source": "gtfintechlab_european_central_bank",
                "provenance": "peer_reviewed_cross_bank",
                "sample_weight": 0.0,
            },
            {
                "record_id": "scraped_1",
                "text": "auxiliary text",
                "event_date": "2024-02-15",
                "mapped_label": "neutral",
                "source": "scraped_fed",
                "provenance": "scraped",
                "sample_weight": 0.0,
            },
        ],
    )
    return package_dir


def test_default_excludes_zero_weight_rows(mixed_registry: Path) -> None:
    rows = _load_registry_rows(mixed_registry)
    assert {r.record_id for r in rows} == {"fomc_1"}
    assert all(r.sample_weight > 0 for r in rows)


def test_include_zero_weight_opts_cross_bank_back_in(mixed_registry: Path) -> None:
    rows = _load_registry_rows(mixed_registry, include_zero_weight=True)
    assert {r.record_id for r in rows} == {"fomc_1", "ecb_1", "scraped_1"}


def test_loaded_row_carries_source_and_provenance(mixed_registry: Path) -> None:
    rows = _load_registry_rows(mixed_registry, include_zero_weight=True)
    by_id = {row.record_id: row for row in rows}
    assert by_id["ecb_1"].source == "gtfintechlab_european_central_bank"
    assert by_id["ecb_1"].provenance == "peer_reviewed_cross_bank"
    assert by_id["fomc_1"].provenance == "peer_reviewed"


def test_invalid_sample_weight_falls_back_to_unit(tmp_path: Path) -> None:
    package_dir = tmp_path / "processed" / "tp_v1"
    _write_registry(
        package_dir / "registry_normalized.jsonl",
        [
            {
                "record_id": "x",
                "text": "speech text",
                "event_date": "2024-01-01",
                "mapped_label": "neutral",
                "source": "hf_fomc_communication",
                "sample_weight": "not-a-number",
            }
        ],
    )
    rows = _load_registry_rows(package_dir)
    assert len(rows) == 1
    assert rows[0].sample_weight == 1.0
