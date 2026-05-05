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


import json

from app.services.scraper_speeches import (
    is_chair_speech,
    write_chair_speeches_json,
)


@pytest.mark.parametrize(
    "speaker,expected",
    [
        ("Chair Powell", True),
        ("Chairman Bernanke", True),
        ("Chair Jerome H. Powell", True),
        ("Chair Yellen", True),
        ("Chairwoman Yellen", True),
        ("Vice Chair Brainard", False),
        ("Vice Chairman Clarida", False),
        ("Governor Waller", False),
        ("Governor Bowman", False),
        ("", False),
    ],
)
def test_is_chair_speech_classifies_speaker_correctly(speaker: str, expected: bool) -> None:
    assert is_chair_speech(speaker) == expected


def test_write_chair_speeches_json_emits_one_row_per_chair_speech(tmp_path: Path) -> None:
    parsed = [
        ParsedSpeech(
            date="2024-01-31",
            speaker="Chair Powell",
            title="Speech on inflation",
            text="Full body of the speech " * 30,
            url="https://www.federalreserve.gov/newsevents/speech/powell20240131a.htm",
        ),
        ParsedSpeech(
            date="2024-02-15",
            speaker="Governor Waller",
            title="Speech on the labor market",
            text="Full body " * 30,
            url="https://www.federalreserve.gov/newsevents/speech/waller20240215a.htm",
        ),
    ]

    output = tmp_path / "chair_speeches.json"
    written = write_chair_speeches_json(parsed, output)

    assert written == 1

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) == 1
    row = payload[0]
    assert row["title"] == "Speech on inflation"
    assert row["date"] == "2024-01-31"
    assert row["text"].startswith("Full body of the speech")
    assert row["document_type"] == "chair_speech"
    assert row["url"].endswith("powell20240131a.htm")
    assert "scraped_at_utc" in row


def test_write_chair_speeches_json_skips_rows_with_empty_body(tmp_path: Path) -> None:
    parsed = [
        ParsedSpeech(
            date="2024-01-31",
            speaker="Chair Powell",
            title="Empty speech",
            text="",
            url="https://www.federalreserve.gov/newsevents/speech/powell20240131a.htm",
        )
    ]
    output = tmp_path / "chair_speeches.json"
    written = write_chair_speeches_json(parsed, output)
    assert written == 0
    assert output.read_text(encoding="utf-8") == "[]"
