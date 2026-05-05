"""Fed congressional testimony archive scraper.

Mirrors scraper_speeches.py for the testimony URL pattern:
  /newsevents/testimony/{lastname}{date}{letter}.htm

Unlike speeches, there is no speaker filter — the entire testimony
archive is congressional testimony by definition, so write_testimonies_json
emits every row with a valid date and non-empty text, tagged
document_type='congressional_testimony'.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

from bs4 import BeautifulSoup

ARCHIVE_BASE_URL = "https://www.federalreserve.gov"
TESTIMONY_URL_PATTERN = re.compile(r"^/newsevents/testimony/[a-z]+(\d{8})[a-z]\.htm$")
DATE_FROM_URL_PATTERN = re.compile(r"/testimony/[a-z]+(\d{8})[a-z]\.htm$")

_MONTH_PATTERN = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s+(\d{4})",
    flags=re.IGNORECASE,
)
# Numeric date as used in the archive time tags: m/d/yyyy or mm/dd/yyyy
_NUMERIC_DATE_PATTERN = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")


@dataclass(frozen=True)
class TestimonyListingEntry:
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


@dataclass(frozen=True)
class ParsedTestimony:
    date: str  # ISO yyyy-mm-dd
    speaker: str
    title: str
    text: str
    url: str


_TITLE_TAIL_PATTERN = re.compile(r"\s*-\s*Federal Reserve Board\s*$", flags=re.IGNORECASE)


def _extract_title(soup: BeautifulSoup) -> str:
    # Prefer the og:title meta when present. Fall back to <title> with the
    # tail stripped, then to any in-content h3/h2.
    og = soup.find("meta", attrs={"property": "og:title"})
    if og is not None and og.get("content"):
        return _clean_text(og["content"])
    title_tag = soup.find("title")
    if title_tag is not None:
        return _TITLE_TAIL_PATTERN.sub("", _clean_text(title_tag.get_text(" ", strip=True)))
    h3 = soup.select_one("h3.title, h3, h2")
    if h3 is not None:
        return _clean_text(h3.get_text(" ", strip=True))
    return ""


_SPEAKER_TITLE_PATTERN = re.compile(
    r"\b(Chair|Chairman|Chairwoman|Vice\s+Chair|Vice\s+Chairman|Governor)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
)


def _extract_speaker(soup: BeautifulSoup, title: str) -> str:
    # Try the explicit speaker selectors first (newer pages use these).
    for selector in ("p.speaker", ".speaker", ".news__speaker", "span.speaker"):
        node = soup.select_one(selector)
        if node is not None:
            text = _clean_text(node.get_text(" ", strip=True))
            if text:
                return text
    # Fall back to extracting "Chair Powell" / "Governor Bowman" out of the title.
    matched = _SPEAKER_TITLE_PATTERN.search(title)
    if matched:
        return _clean_text(matched.group(0))
    return ""


def _extract_body(soup: BeautifulSoup) -> str:
    selectors = [
        "div.col-xs-12.col-sm-8.col-md-8 p",
        "div#article p",
        "article p",
        "main p",
    ]
    for selector in selectors:
        nodes = soup.select(selector)
        if nodes:
            chunks = [_clean_text(node.get_text(" ", strip=True)) for node in nodes]
            return "\n".join(chunk for chunk in chunks if chunk)
    return ""


def parse_testimony_page(html: str, *, source_url: str) -> ParsedTestimony:
    """Parse a single federalreserve.gov testimony page into a ParsedTestimony.

    Falls back gracefully when individual fields are missing on the page;
    callers decide whether to keep or discard rows with empty fields.
    """

    soup = BeautifulSoup(html, "html.parser")
    title = _extract_title(soup)
    speaker = _extract_speaker(soup, title)
    body = _extract_body(soup)

    article_time = soup.select_one("p.article__time, .article__time, time")
    date_text = _clean_text(article_time.get_text(" ", strip=True)) if article_time else ""
    date_iso = _date_from_url(source_url) or _coerce_date(date_text)

    return ParsedTestimony(
        date=date_iso,
        speaker=speaker,
        title=title,
        text=body,
        url=source_url,
    )


def extract_testimony_listing(html: str) -> list[TestimonyListingEntry]:
    """Parse a federalreserve.gov annual testimony archive page.

    Returns one TestimonyListingEntry per linked testimony. Duplicate URLs
    collapse to the first occurrence; non-testimony anchors are skipped.
    """

    soup = BeautifulSoup(html, "html.parser")
    entries: list[TestimonyListingEntry] = []
    seen_urls: set[str] = set()

    for anchor in soup.select("a[href]"):
        href = (anchor.get("href") or "").strip()
        if not TESTIMONY_URL_PATTERN.match(href):
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
            TestimonyListingEntry(
                date=date_iso,
                speaker=speaker,
                title=title,
                url=absolute,
            )
        )
    return entries


def write_testimonies_json(parsed: Iterable[ParsedTestimony], output_path: Path) -> int:
    """Write all valid testimony rows to output_path as a JSON list.

    Unlike the speech writers, there is no speaker filter — every testimony
    entry is congressional testimony by definition. Rows with empty text or
    date are skipped. Returns the number of rows written.

    Each row matches the schema consumed by ingest_sources._iter_scraped_records:
    date, title, text, document_type ('congressional_testimony'), url, scraped_at_utc.
    """

    rows: list[dict[str, str]] = []
    scraped_at = datetime.now(timezone.utc).isoformat()
    for entry in parsed:
        if not entry.text or not entry.date:
            continue
        rows.append(
            {
                "date": entry.date,
                "title": entry.title,
                "text": entry.text,
                "document_type": "congressional_testimony",
                "url": entry.url,
                "scraped_at_utc": scraped_at,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return len(rows)
