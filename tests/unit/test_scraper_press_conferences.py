from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.services.scraper_press_conferences import (
    PressConferenceListingEntry,
    ParsedPressConference,
    build_qa_lookup,
    extract_press_conference_listing,
    parse_press_conference_page,
    split_prepared_remarks_and_qa,
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
            prepared_remarks_text="Opening remarks " * 10,
            qa_text="Q. and A. " * 30,
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
    # #214: Q&A and prepared remarks are persisted alongside the legacy
    # full-transcript text so downstream lookups can address either slice.
    assert "qa_text" in payload[0]
    assert "prepared_remarks_text" in payload[0]
    assert payload[0]["qa_text"].startswith("Q. and A.")


def test_split_prepared_remarks_and_qa_handles_real_transcript() -> None:
    """The real sample PDF must split into a small remarks slice and a
    much larger Q&A slice (Q&A is the high-information portion). The
    boundary anchors on "I look forward to your questions"."""

    from pypdf import PdfReader

    pdf = PdfReader(FIXTURES / "fed_press_conference_sample.pdf")
    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    prepared, qa = split_prepared_remarks_and_qa(text)

    assert prepared, "prepared remarks must be non-empty on a valid transcript"
    assert qa, "Q&A must be non-empty on a valid transcript"
    # Sanity floor: prepared remarks usually run 5-10% of the transcript;
    # Q&A is the bulk of the text. A real Powell press conference has at
    # least 30k chars of Q&A.
    assert len(qa) > 30_000
    assert len(prepared) < len(qa) // 3
    # The hand-off phrase ends the prepared remarks.
    assert "look forward to your questions" in prepared.lower()
    # Q&A must contain at least one reporter speaker turn.
    assert "STEVE LIESMAN" in qa or "MICHELLE SMITH" in qa


def test_split_prepared_remarks_and_qa_empty_on_missing_text() -> None:
    assert split_prepared_remarks_and_qa("") == ("", "")


def test_split_prepared_remarks_and_qa_returns_remarks_on_missing_boundary() -> None:
    """When neither the hand-off phrase nor a moderator turn appears the
    whole text is returned as prepared remarks and Q&A is empty — the
    caller flips ``has_press_conf`` based on whether ``text`` is
    populated, not on whether Q&A landed."""

    text = "Some short statement. " * 50
    prepared, qa = split_prepared_remarks_and_qa(text)
    assert prepared.strip()
    assert qa == ""


def test_build_qa_lookup_keys_on_event_date() -> None:
    parsed = [
        ParsedPressConference(
            date="2024-03-20",
            title="FOMC Press Conference",
            text="full transcript",
            url="https://www.federalreserve.gov/monetarypolicy/fomcpresconf20240320.htm",
            prepared_remarks_text="opening",
            qa_text="reporter and chair exchange",
        ),
        ParsedPressConference(
            date="2024-05-01",
            title="FOMC Press Conference",
            text="",  # download failed; skipped
            url="https://www.federalreserve.gov/monetarypolicy/fomcpresconf20240501.htm",
        ),
    ]

    lookup = build_qa_lookup(parsed)
    assert set(lookup.keys()) == {"2024-03-20"}
    assert lookup["2024-03-20"]["qa_text"].startswith("reporter and chair")
    assert lookup["2024-03-20"]["has_press_conf"] == "1"


def test_parse_press_conference_page_caches_pdf_to_disk(tmp_path: Path, monkeypatch) -> None:
    """The cache_pdf_dir kwarg must persist the fetched PDF locally so
    a second call short-circuits the network fetch (#214)."""

    sample_html = (FIXTURES / "fed_press_conference_sample.html").read_text(encoding="utf-8")
    pdf_bytes = (FIXTURES / "fed_press_conference_sample.pdf").read_bytes()
    source_url = "https://www.federalreserve.gov/monetarypolicy/fomcpresconf20250129.htm"

    call_count = {"n": 0}

    class _StubResponse:
        def __init__(self, content):
            self.content = content
            self.status_code = 200

        def raise_for_status(self):
            pass

    def fake_get(url, *args, **kwargs):
        call_count["n"] += 1
        return _StubResponse(pdf_bytes)

    monkeypatch.setattr("app.services.scraper_press_conferences.requests.get", fake_get)

    cache_dir = tmp_path / "press_conf_cache"
    parsed_first = parse_press_conference_page(
        sample_html, source_url=source_url, cache_pdf_dir=cache_dir
    )
    parsed_second = parse_press_conference_page(
        sample_html, source_url=source_url, cache_pdf_dir=cache_dir
    )

    assert call_count["n"] == 1  # second call short-circuits to cache
    cached_pdf = cache_dir / "20250129.pdf"
    assert cached_pdf.exists()
    assert cached_pdf.read_bytes() == pdf_bytes
    assert parsed_first.text == parsed_second.text
    # Both calls must produce the same Q&A split.
    assert parsed_first.qa_text == parsed_second.qa_text
    assert parsed_first.qa_text
    assert parsed_first.prepared_remarks_text
