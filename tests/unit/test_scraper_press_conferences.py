from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.services.scraper_press_conferences import (
    PressConferenceListingEntry,
    ParsedPressConference,
    extract_press_conference_listing,
    parse_press_conference_page,
    write_press_conferences_json,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_extract_press_conference_listing_returns_entries_from_calendar() -> None:
    html = (FIXTURES / "fed_fomc_calendar.html").read_text(encoding="utf-8")

    entries = extract_press_conference_listing(html)

    # Calendar typically lists 8 per year for the past 2-3 years; expect at least 5 total
    assert len(entries) >= 5
    for entry in entries:
        assert isinstance(entry, PressConferenceListingEntry)
        assert entry.url.startswith("https://www.federalreserve.gov/monetarypolicy/fomcpresconf")
        assert entry.url.endswith(".htm")
        assert entry.date  # ISO yyyy-mm-dd derived from URL


def test_extract_press_conference_listing_returns_empty_on_unrelated_html() -> None:
    assert extract_press_conference_listing("<html><body>nothing</body></html>") == []


def test_extract_press_conference_listing_deduplicates_repeated_urls() -> None:
    repeated = "/monetarypolicy/fomcpresconf20240131.htm"
    html = f'<html><body><a href="{repeated}">A</a><a href="{repeated}">B</a></body></html>'
    entries = extract_press_conference_listing(html)
    assert len(entries) == 1


def test_parse_press_conference_page_extracts_date_and_body() -> None:
    sample_html = (FIXTURES / "fed_press_conference_sample.html").read_text(encoding="utf-8")
    calendar_html = (FIXTURES / "fed_fomc_calendar.html").read_text(encoding="utf-8")
    match = re.search(r'/monetarypolicy/fomcpresconf202[45][0-9]{4}\.htm', calendar_html)
    assert match
    source_url = "https://www.federalreserve.gov" + match.group(0)

    parsed = parse_press_conference_page(sample_html, source_url=source_url)

    assert isinstance(parsed, ParsedPressConference)
    assert parsed.date.startswith("20")
    assert len(parsed.text) > 200
    assert parsed.url == source_url


def test_write_press_conferences_json_emits_one_row_per_parsed(tmp_path: Path) -> None:
    parsed = [
        ParsedPressConference(
            date="2024-03-20",
            title="FOMC Press Conference",
            text="Full transcript " * 50,
            url="https://www.federalreserve.gov/monetarypolicy/fomcpresconf20240320.htm",
        ),
        ParsedPressConference(
            date="",
            title="missing date",
            text="something",
            url="https://www.federalreserve.gov/monetarypolicy/fomcpresconf20240501.htm",
        ),
    ]

    output = tmp_path / "press_conferences.json"
    written = write_press_conferences_json(parsed, output)

    assert written == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["document_type"] == "press_conference"
    assert payload[0]["date"] == "2024-03-20"
