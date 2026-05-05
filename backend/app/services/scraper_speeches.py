"""Fed speech archive scraper.

Two responsibilities (Task 2 lands the first; Task 3 lands the second):
1. List speeches from the annual archive page (extract_speech_listing).
2. Parse a single speech page into a structured row (parse_speech_page).

Output rows match the schema used by services/scraper.py so the existing
ingestion pipeline picks them up without changes when wired through
ingest_sources.SCRAPED_FILES.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urljoin

from bs4 import BeautifulSoup

ARCHIVE_BASE_URL = "https://www.federalreserve.gov"
SPEECH_URL_PATTERN = re.compile(r"^/newsevents/speech/[a-z]+(\d{8})[a-z]\.htm$")
DATE_FROM_URL_PATTERN = re.compile(r"/speech/[a-z]+(\d{8})[a-z]\.htm$")

_MONTH_PATTERN = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s+(\d{4})",
    flags=re.IGNORECASE,
)
# Numeric date as used in the archive time tags: m/d/yyyy or mm/dd/yyyy
_NUMERIC_DATE_PATTERN = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")


@dataclass(frozen=True)
class SpeechListingEntry:
    date: str  # ISO yyyy-mm-dd
    speaker: str
    title: str
    url: str  # absolute


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def _date_from_url(url: str) -> str:
    matched = DATE_FROM_URL_PATTERN.search(url)
    if not matched:
        return ""
    return datetime.strptime(matched.group(1), "%Y%m%d").date().isoformat()


def _coerce_date(value: str) -> str:
    """Parse a date string that may be in month-name or numeric m/d/yyyy form."""
    text = (value or "").strip()

    # Try numeric m/d/yyyy first (used in archive <time> tags)
    m = _NUMERIC_DATE_PATTERN.match(text)
    if m:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(year, month, day).date().isoformat()
        except ValueError:
            pass

    # Fall back to written-out month name
    matched = _MONTH_PATTERN.search(text)
    if not matched:
        return ""
    return datetime.strptime(matched.group(0), "%B %d, %Y").date().isoformat()


def extract_speech_listing(html: str) -> list[SpeechListingEntry]:
    """Parse a federalreserve.gov annual speech archive page.

    Returns one SpeechListingEntry per linked speech. Duplicate URLs
    collapse to the first occurrence; non-speech anchors are skipped.
    """

    soup = BeautifulSoup(html, "html.parser")
    entries: list[SpeechListingEntry] = []
    seen_urls: set[str] = set()

    for anchor in soup.select("a[href]"):
        href = (anchor.get("href") or "").strip()
        if not SPEECH_URL_PATTERN.match(href):
            continue
        absolute = urljoin(ARCHIVE_BASE_URL, href)
        if absolute in seen_urls:
            continue
        seen_urls.add(absolute)

        title = _clean_text(anchor.get_text(" ", strip=True))
        if not title:
            continue

        speaker = ""
        explicit_date = ""
        # Walk up to a row-level container that holds both the speaker and date.
        # The immediate parent of the <a> is a <p>; we want the enclosing .row
        # or the closest div/tr/li that is not the direct <p> wrapper.
        row = anchor.find_parent(class_=re.compile(r"row|eventlist__event"))
        if row is None:
            row = anchor.find_parent(["tr", "li"])
        if row is not None:
            speaker_node = row.select_one(".speaker, .news__speaker")
            if speaker_node is not None:
                speaker = _clean_text(speaker_node.get_text(" ", strip=True))
            time_node = row.select_one("time, .news__date, .eventlist__time")
            if time_node is not None:
                explicit_date = _clean_text(time_node.get_text(" ", strip=True))

        date_iso = _date_from_url(absolute) or _coerce_date(explicit_date)
        if not date_iso:
            continue

        entries.append(
            SpeechListingEntry(
                date=date_iso,
                speaker=speaker,
                title=title,
                url=absolute,
            )
        )
    return entries
