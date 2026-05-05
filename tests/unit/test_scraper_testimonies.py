from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.scraper_testimonies import (
    TestimonyListingEntry,
    ParsedTestimony,
    extract_testimony_listing,
    parse_testimony_page,
    write_testimonies_json,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_extract_testimony_listing_returns_entries_from_real_archive_page() -> None:
    html = (FIXTURES / "fed_testimony_archive.html").read_text(encoding="utf-8")

    entries = extract_testimony_listing(html)

    # Archive should list at least a few testimonies
    assert len(entries) >= 3
    for entry in entries:
        assert isinstance(entry, TestimonyListingEntry)
        assert entry.url.startswith("https://www.federalreserve.gov/newsevents/testimony/")
        assert entry.url.endswith(".htm")
        assert entry.date  # ISO yyyy-mm-dd
        assert entry.title


def test_extract_testimony_listing_returns_empty_on_unrelated_html() -> None:
    assert extract_testimony_listing("<html><body>nothing here</body></html>") == []


def test_extract_testimony_listing_deduplicates_repeated_urls() -> None:
    repeated_url = "/newsevents/testimony/powell20240131a.htm"
    html = f'<html><body><a href="{repeated_url}">A</a><a href="{repeated_url}">B</a></body></html>'
    entries = extract_testimony_listing(html)
    assert len(entries) == 1
    assert entries[0].url.endswith(repeated_url)


def test_parse_testimony_page_extracts_speaker_date_and_body() -> None:
    html = (FIXTURES / "fed_testimony_sample.html").read_text(encoding="utf-8")
    # Use any testimony URL — the actual fixture's URL is what matters for date inference
    # Pick one that grep'd from the archive
    archive_html = (FIXTURES / "fed_testimony_archive.html").read_text(encoding="utf-8")
    import re
    match = re.search(r'/newsevents/testimony/[a-z]+[0-9]{8}[a-z]\.htm', archive_html)
    assert match, "No testimony URL found in archive fixture"
    source_url = "https://www.federalreserve.gov" + match.group(0)

    parsed = parse_testimony_page(html, source_url=source_url)

    assert isinstance(parsed, ParsedTestimony)
    assert parsed.speaker  # non-empty
    assert parsed.date.startswith("20")  # ISO date
    assert len(parsed.text) > 200
    assert parsed.title
    assert parsed.url == source_url


def test_write_testimonies_json_emits_one_row_per_parsed(tmp_path: Path) -> None:
    parsed = [
        ParsedTestimony(
            date="2024-03-06",
            speaker="Chair Powell",
            title="Semiannual Monetary Policy Report",
            text="Full body of the testimony " * 30,
            url="https://www.federalreserve.gov/newsevents/testimony/powell20240306a.htm",
        ),
        ParsedTestimony(
            date="",
            speaker="Governor Waller",
            title="Empty date",
            text="something",
            url="https://www.federalreserve.gov/newsevents/testimony/waller20240501a.htm",
        ),
    ]

    output = tmp_path / "congressional_testimonies.json"
    written = write_testimonies_json(parsed, output)

    assert written == 1  # the empty-date row is skipped
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["document_type"] == "congressional_testimony"
    assert payload[0]["title"] == "Semiannual Monetary Policy Report"
