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


def test_parse_press_conference_page_extracts_transcript_from_pdf(tmp_path: Path, monkeypatch) -> None:
    """The press conference HTML page is a video-only landing; the
    transcript lives in a sibling PDF. parse_press_conference_page
    must download and extract the PDF text."""

    sample_html = (FIXTURES / "fed_press_conference_sample.html").read_text(encoding="utf-8")
    pdf_bytes = (FIXTURES / "fed_press_conference_sample.pdf").read_bytes()

    calendar_html = (FIXTURES / "fed_fomc_calendar.html").read_text(encoding="utf-8")
    match = re.search(r'/monetarypolicy/fomcpresconf202[45][0-9]{4}\.htm', calendar_html)
    assert match
    source_url = "https://www.federalreserve.gov" + match.group(0)

    # Stub the PDF download
    class _StubResponse:
        def __init__(self, content):
            self.content = content
            self.status_code = 200

        def raise_for_status(self):
            pass

    def fake_get(url, *args, **kwargs):
        # Verify the PDF URL is constructed correctly
        assert "FOMCpresconf" in url
        assert url.endswith(".pdf")
        return _StubResponse(pdf_bytes)

    monkeypatch.setattr("app.services.scraper_press_conferences.requests.get", fake_get)

    parsed = parse_press_conference_page(sample_html, source_url=source_url)

    assert isinstance(parsed, ParsedPressConference)
    assert parsed.date.startswith("20")
    # Real transcript should be substantial (thousands of chars), not 600 chars of boilerplate
    assert len(parsed.text) > 5000
    assert parsed.url == source_url
    # Powell's prepared remarks should mention the FOMC at minimum
    assert "FOMC" in parsed.text or "Federal" in parsed.text


def test_parse_press_conference_page_falls_back_when_pdf_unavailable(monkeypatch) -> None:
    """If the PDF download fails (404, network), return empty text rather than raising."""

    sample_html = (FIXTURES / "fed_press_conference_sample.html").read_text(encoding="utf-8")
    source_url = "https://www.federalreserve.gov/monetarypolicy/fomcpresconf20240131.htm"

    def fake_get(url, *args, **kwargs):
        raise Exception("simulated network failure")

    monkeypatch.setattr("app.services.scraper_press_conferences.requests.get", fake_get)

    parsed = parse_press_conference_page(sample_html, source_url=source_url)
    assert parsed.text == ""
    assert parsed.date.startswith("2024")


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
