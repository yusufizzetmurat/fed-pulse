from __future__ import annotations

from pathlib import Path

import pytest

from app.services.scraper_speeches import (
    SpeechListingEntry,
    extract_speech_listing,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_extract_speech_listing_returns_entries_from_real_archive_page() -> None:
    html = (FIXTURES / "fed_speech_archive_2024.html").read_text(encoding="utf-8")

    entries = extract_speech_listing(html)

    # The archive lists at least 10 speeches in 2024.
    assert len(entries) >= 10
    for entry in entries:
        assert isinstance(entry, SpeechListingEntry)
        assert entry.url.startswith("https://www.federalreserve.gov/newsevents/speech/")
        assert entry.url.endswith(".htm")
        assert entry.date  # ISO yyyy-mm-dd
        assert entry.title  # non-empty


def test_extract_speech_listing_returns_empty_on_unrelated_html() -> None:
    assert extract_speech_listing("<html><body>nothing here</body></html>") == []


def test_extract_speech_listing_deduplicates_repeated_urls() -> None:
    """Same URL listed twice in the archive must collapse to one entry."""
    repeated_url = "/newsevents/speech/powell20240131a.htm"
    html = f"""
    <html><body>
      <a href="{repeated_url}">Speech on inflation</a>
      <a href="{repeated_url}">Speech on inflation (duplicate)</a>
    </body></html>
    """
    entries = extract_speech_listing(html)
    assert len(entries) == 1
    assert entries[0].url.endswith(repeated_url)


from app.services.scraper_speeches import ParsedSpeech, parse_speech_page


def test_parse_speech_page_extracts_speaker_date_and_body() -> None:
    html = (FIXTURES / "fed_speech_powell_2024_sample.html").read_text(encoding="utf-8")
    source_url = "https://www.federalreserve.gov/newsevents/speech/powell20241114a.htm"

    parsed = parse_speech_page(html, source_url=source_url)

    assert isinstance(parsed, ParsedSpeech)
    assert "Powell" in parsed.speaker
    assert parsed.date == "2024-11-14"  # date is derivable from the URL
    assert len(parsed.text) > 500
    assert parsed.title  # non-empty
    assert parsed.url == source_url
