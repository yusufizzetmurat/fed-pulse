from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.services.scraper_beige_book import (
    BeigeBookListingEntry,
    ParsedBeigeBook,
    extract_beige_book_listing,
    parse_beige_book_page,
    write_beige_book_json,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_extract_beige_book_listing_returns_entries() -> None:
    """The listing has both base and summary URLs; adapter returns deduplicated entries."""

    html = (FIXTURES / "fed_beige_book_listing.html").read_text(encoding="utf-8")
    entries = extract_beige_book_listing(html)

    assert len(entries) >= 5
    for entry in entries:
        assert isinstance(entry, BeigeBookListingEntry)
        assert entry.url.startswith("https://www.federalreserve.gov/monetarypolicy/beigebook")
        assert entry.url.endswith(".htm")
        assert entry.date  # ISO yyyy-mm-dd derived from URL


def test_extract_beige_book_listing_returns_empty_on_unrelated_html() -> None:
    assert extract_beige_book_listing("<html><body>nothing</body></html>") == []


def test_extract_beige_book_listing_deduplicates_repeated_urls() -> None:
    repeated = "/monetarypolicy/beigebook202401-summary.htm"
    html = f'<html><body><a href="{repeated}">A</a><a href="{repeated}">B</a></body></html>'
    entries = extract_beige_book_listing(html)
    assert len(entries) == 1


def test_parse_beige_book_page_extracts_substantive_text() -> None:
    html = (FIXTURES / "fed_beige_book_sample.html").read_text(encoding="utf-8")
    # The sample fixture is the January 2026 National Summary
    source_url = "https://www.federalreserve.gov/monetarypolicy/beigebook202601-summary.htm"

    parsed = parse_beige_book_page(html, source_url=source_url)

    assert isinstance(parsed, ParsedBeigeBook)
    assert parsed.date.startswith("20")
    # Full Beige Book national summary is substantial — at least 5k chars of economic content
    assert len(parsed.text) > 5000
    assert parsed.url == source_url


def test_write_beige_book_json_emits_one_row_per_parsed(tmp_path: Path) -> None:
    parsed = [
        ParsedBeigeBook(
            date="2024-03-01",
            title="Beige Book - March 2024",
            text="Full body of the report " * 50,
            url="https://www.federalreserve.gov/monetarypolicy/beigebook202403-summary.htm",
        ),
        ParsedBeigeBook(
            date="",
            title="Empty date",
            text="something",
            url="https://www.federalreserve.gov/monetarypolicy/beigebook202404-summary.htm",
        ),
    ]

    output = tmp_path / "beige_book.json"
    written = write_beige_book_json(parsed, output)

    assert written == 1  # empty-date row is skipped
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["document_type"] == "beige_book"
