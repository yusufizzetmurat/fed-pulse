from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.data import ingest_sources
from app.data.source_type import (
    SOURCE_TYPE_FOMC_MINUTES,
    SOURCE_TYPE_FOMC_STATEMENT,
    SOURCE_TYPE_VALUES,
)


def _write_scraped_fixture(data_dir: Path) -> None:
    minutes = [
        {
            "date": "2024-01-31",
            "title": "FOMC Meeting Minutes January 31, 2024",
            "text": "Some minutes text",
            "document_type": "minutes",
        }
    ]
    statements = [
        {
            "date": "2024-03-20",
            "title": "FOMC statement",
            "text": "Some statement text",
            "document_type": "statement",
        }
    ]
    (data_dir / "fomc_minutes.json").write_text(json.dumps(minutes), encoding="utf-8")
    (data_dir / "fomc_statements.json").write_text(json.dumps(statements), encoding="utf-8")


def test_build_registry_record_carries_source_type() -> None:
    record = ingest_sources._build_registry_record(
        source="scraped_fed",
        source_record_id="sample:0",
        event_date="2024-01-31",
        document_type="minutes",
        title="FOMC Meeting Minutes January 31, 2024",
        text="hello world",
        label="",
        license_scope="public_source_scrape_terms_required",
        citation_ref="federalreserve_primary_source",
    )

    assert record is not None
    assert record["source_type"] == SOURCE_TYPE_FOMC_MINUTES
    assert record["source_type"] in SOURCE_TYPE_VALUES


def test_iter_scraped_records_assigns_source_type_per_filename(tmp_path: Path) -> None:
    _write_scraped_fixture(tmp_path)
    records = ingest_sources._iter_scraped_records(tmp_path)

    by_type = {r["source_type"] for r in records}
    assert SOURCE_TYPE_FOMC_MINUTES in by_type
    assert SOURCE_TYPE_FOMC_STATEMENT in by_type


def test_write_summary_includes_source_type_counts(tmp_path: Path) -> None:
    rows = [
        {"source": "scraped_fed", "source_type": SOURCE_TYPE_FOMC_MINUTES, "label": ""},
        {"source": "scraped_fed", "source_type": SOURCE_TYPE_FOMC_MINUTES, "label": ""},
        {"source": "scraped_fed", "source_type": SOURCE_TYPE_FOMC_STATEMENT, "label": "hawkish"},
    ]
    summary_path = tmp_path / "ingestion_summary.json"
    ingest_sources._write_summary(summary_path, rows)

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["source_type_counts"] == {
        SOURCE_TYPE_FOMC_MINUTES: 2,
        SOURCE_TYPE_FOMC_STATEMENT: 1,
    }
