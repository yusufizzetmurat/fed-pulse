from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.data import ingest_sources
from app.data.source_type import (
    SOURCE_TYPE_BEIGE_BOOK,
    SOURCE_TYPE_CHAIR_SPEECH,
    SOURCE_TYPE_CONGRESSIONAL_TESTIMONY,
    SOURCE_TYPE_FOMC_MINUTES,
    SOURCE_TYPE_FOMC_STATEMENT,
    SOURCE_TYPE_GOVERNOR_SPEECH,
    SOURCE_TYPE_PRESS_CONFERENCE,
    SOURCE_TYPE_REGIONAL_RESEARCH,
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


def test_iter_scraped_records_assigns_chair_speech_source_type(tmp_path: Path) -> None:
    speeches = [
        {
            "date": "2024-01-31",
            "title": "Chair Powell on inflation",
            "text": "Full speech text",
            "document_type": "chair_speech",
            "url": "https://www.federalreserve.gov/newsevents/speech/powell20240131a.htm",
            "scraped_at_utc": "2024-01-31T12:00:00+00:00",
        }
    ]
    (tmp_path / "chair_speeches.json").write_text(json.dumps(speeches), encoding="utf-8")

    records = ingest_sources._iter_scraped_records(tmp_path)

    chair_records = [r for r in records if r["source_type"] == SOURCE_TYPE_CHAIR_SPEECH]
    assert len(chair_records) == 1
    assert chair_records[0]["source"] == "scraped_fed"
    assert chair_records[0]["title"] == "Chair Powell on inflation"
    assert chair_records[0]["event_date"] == "2024-01-31"


def test_iter_scraped_records_assigns_governor_speech_source_type(tmp_path: Path) -> None:
    speeches = [
        {
            "date": "2024-02-15",
            "title": "Governor Waller on the labor market",
            "text": "Full speech text",
            "document_type": "governor_speech",
            "url": "https://www.federalreserve.gov/newsevents/speech/waller20240215a.htm",
            "scraped_at_utc": "2024-02-15T12:00:00+00:00",
        }
    ]
    (tmp_path / "governor_speeches.json").write_text(json.dumps(speeches), encoding="utf-8")

    records = ingest_sources._iter_scraped_records(tmp_path)

    governor_records = [r for r in records if r["source_type"] == SOURCE_TYPE_GOVERNOR_SPEECH]
    assert len(governor_records) == 1
    assert governor_records[0]["source"] == "scraped_fed"
    assert governor_records[0]["title"] == "Governor Waller on the labor market"
    assert governor_records[0]["event_date"] == "2024-02-15"


def test_iter_scraped_records_assigns_congressional_testimony_source_type(tmp_path: Path) -> None:
    rows = [
        {
            "date": "2024-03-06",
            "title": "Semiannual Monetary Policy Report",
            "text": "Full text",
            "document_type": "congressional_testimony",
            "url": "https://www.federalreserve.gov/newsevents/testimony/powell20240306a.htm",
            "scraped_at_utc": "2024-03-06T12:00:00+00:00",
        }
    ]
    (tmp_path / "congressional_testimonies.json").write_text(json.dumps(rows), encoding="utf-8")

    records = ingest_sources._iter_scraped_records(tmp_path)

    test_records = [r for r in records if r["source_type"] == SOURCE_TYPE_CONGRESSIONAL_TESTIMONY]
    assert len(test_records) == 1
    assert test_records[0]["source"] == "scraped_fed"
    assert test_records[0]["title"] == "Semiannual Monetary Policy Report"


def test_iter_scraped_records_assigns_press_conference_source_type(tmp_path: Path) -> None:
    rows = [
        {
            "date": "2024-03-20",
            "title": "FOMC Press Conference",
            "text": "Full transcript",
            "document_type": "press_conference",
            "url": "https://www.federalreserve.gov/monetarypolicy/fomcpresconf20240320.htm",
            "scraped_at_utc": "2024-03-20T18:00:00+00:00",
        }
    ]
    (tmp_path / "press_conferences.json").write_text(json.dumps(rows), encoding="utf-8")

    records = ingest_sources._iter_scraped_records(tmp_path)

    pc_records = [r for r in records if r["source_type"] == SOURCE_TYPE_PRESS_CONFERENCE]
    assert len(pc_records) == 1
    assert pc_records[0]["source"] == "scraped_fed"
    assert pc_records[0]["title"] == "FOMC Press Conference"


def test_iter_scraped_records_assigns_beige_book_source_type(tmp_path: Path) -> None:
    rows = [
        {
            "date": "2024-03-01",
            "title": "Beige Book - March 2024",
            "text": "Full report body",
            "document_type": "beige_book",
            "url": "https://www.federalreserve.gov/monetarypolicy/beigebook202403-summary.htm",
            "scraped_at_utc": "2024-03-01T12:00:00+00:00",
        }
    ]
    (tmp_path / "beige_book.json").write_text(json.dumps(rows), encoding="utf-8")

    records = ingest_sources._iter_scraped_records(tmp_path)

    bb_records = [r for r in records if r["source_type"] == SOURCE_TYPE_BEIGE_BOOK]
    assert len(bb_records) == 1
    assert bb_records[0]["source"] == "scraped_fed"
    assert bb_records[0]["title"] == "Beige Book - March 2024"


def test_iter_scraped_records_assigns_regional_research_source_type(tmp_path: Path) -> None:
    rows = [
        {
            "date": "2024-03-01",
            "title": "A Liberty Street Post",
            "text": "Body of the post",
            "document_type": "regional_research",
            "url": "https://libertystreeteconomics.newyorkfed.org/2024/03/sample-post/",
            "source_bank": "ny_fed",
            "scraped_at_utc": "2024-03-01T12:00:00+00:00",
        }
    ]
    (tmp_path / "regional_research.json").write_text(json.dumps(rows), encoding="utf-8")

    records = ingest_sources._iter_scraped_records(tmp_path)

    rr_records = [r for r in records if r["source_type"] == SOURCE_TYPE_REGIONAL_RESEARCH]
    assert len(rr_records) == 1
    assert rr_records[0]["source"] == "scraped_fed"
    assert rr_records[0]["title"] == "A Liberty Street Post"
