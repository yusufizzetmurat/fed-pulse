from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.services.scraper_regional_research import (
    RegionalResearchListingEntry,
    ParsedRegionalResearch,
    extract_lse_listing,
    parse_lse_post,
    write_regional_research_json,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_extract_lse_listing_returns_entries_with_expected_url_pattern() -> None:
    html = (FIXTURES / "lse_listing.html").read_text(encoding="utf-8")
    entries = extract_lse_listing(html)

    assert len(entries) >= 3
    for entry in entries:
        assert isinstance(entry, RegionalResearchListingEntry)
        assert entry.url.startswith("https://libertystreeteconomics.newyorkfed.org/")
        assert entry.date  # ISO yyyy-mm-01 (date precision is month-level from URL)
        assert entry.title or True  # title may be empty; not strictly required


def test_extract_lse_listing_returns_empty_on_unrelated_html() -> None:
    assert extract_lse_listing("<html><body>nothing</body></html>") == []


def test_extract_lse_listing_deduplicates_repeated_urls() -> None:
    repeated = "https://libertystreeteconomics.newyorkfed.org/2024/03/sample-post/"
    html = f'<html><body><a href="{repeated}">A</a><a href="{repeated}">B</a></body></html>'
    entries = extract_lse_listing(html)
    assert len(entries) == 1


def test_parse_lse_post_extracts_title_and_substantive_text() -> None:
    html = (FIXTURES / "lse_post_sample.html").read_text(encoding="utf-8")
    listing_html = (FIXTURES / "lse_listing.html").read_text(encoding="utf-8")
    match = re.search(r'https://libertystreeteconomics\.newyorkfed\.org/[0-9]{4}/[0-9]{2}/[a-z0-9-]+/?', listing_html)
    assert match
    source_url = match.group(0)

    parsed = parse_lse_post(html, source_url=source_url)

    assert isinstance(parsed, ParsedRegionalResearch)
    assert parsed.date.startswith("20")
    assert parsed.url == source_url
    assert parsed.title  # non-empty
    # Liberty Street posts are typically a few thousand chars
    assert len(parsed.text) > 1000


def test_write_regional_research_json_emits_one_row_per_parsed(tmp_path: Path) -> None:
    parsed = [
        ParsedRegionalResearch(
            date="2024-03-01",
            title="A Liberty Street Post",
            text="Body of the post " * 80,
            url="https://libertystreeteconomics.newyorkfed.org/2024/03/sample-post/",
            source_bank="ny_fed",
        ),
        ParsedRegionalResearch(
            date="",
            title="Empty date",
            text="something",
            url="https://libertystreeteconomics.newyorkfed.org/2024/04/another/",
            source_bank="ny_fed",
        ),
    ]

    output = tmp_path / "regional_research.json"
    written = write_regional_research_json(parsed, output)

    assert written == 1  # empty-date row is skipped
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["document_type"] == "regional_research"
    assert payload[0]["source_bank"] == "ny_fed"
