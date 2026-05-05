"""FOMC press-conference scraper.

Same FRB site template as speech / testimony pages, so the same body
selectors apply. URL pattern is /monetarypolicy/fomcpresconf{YYYYMMDD}.htm
(no speaker letter; the Chair always leads). The press-conference
archive lives on the FOMC calendar at /monetarypolicy/fomccalendars.htm
alongside meeting-minutes URLs. There is no speaker filter — every
entry is a Chair-led press conference.

Structural note: as of 2024-2025, the individual press-conference pages are
video-landing pages. The actual transcript text is published only as a PDF
(linked from the page as "Press Conference Transcript (PDF)"). The
div#article p selector therefore captures the video accessibility boilerplate
and the Related Information links rather than the speech transcript body.
PDF parsing is left as a future enhancement. Body text extracted here is
sufficient to register the event in the source registry but should not be
used for NLP training without the PDF transcript fallback.
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
PRESS_CONF_URL_PATTERN = re.compile(r"^/monetarypolicy/fomcpresconf(\d{8})\.htm$")
DATE_FROM_URL_PATTERN = re.compile(r"/fomcpresconf(\d{8})\.htm$")


@dataclass(frozen=True)
class PressConferenceListingEntry:
    date: str
    url: str


@dataclass(frozen=True)
class ParsedPressConference:
    date: str
    title: str
    text: str
    url: str


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def _date_from_url(url: str) -> str:
    matched = DATE_FROM_URL_PATTERN.search(url)
    if not matched:
        return ""
    return datetime.strptime(matched.group(1), "%Y%m%d").date().isoformat()


def extract_press_conference_listing(html: str) -> list[PressConferenceListingEntry]:
    """Parse FOMC calendar HTML for press-conference URLs.

    Returns one entry per unique URL. Entries with an unparseable date
    (URL doesn't match the pattern) are skipped.
    """

    soup = BeautifulSoup(html, "html.parser")
    entries: list[PressConferenceListingEntry] = []
    seen: set[str] = set()

    for anchor in soup.select("a[href]"):
        href = (anchor.get("href") or "").strip()
        if not PRESS_CONF_URL_PATTERN.match(href):
            continue
        absolute = urljoin(ARCHIVE_BASE_URL, href)
        if absolute in seen:
            continue
        seen.add(absolute)

        date_iso = _date_from_url(absolute)
        if not date_iso:
            continue

        entries.append(PressConferenceListingEntry(date=date_iso, url=absolute))
    return entries


def parse_press_conference_page(html: str, *, source_url: str) -> ParsedPressConference:
    """Parse a single press-conference page into a ParsedPressConference.

    Reuses the same FRB body selectors that work for speeches / testimony.
    The Chair is always the speaker; we don't extract a speaker field
    separately because all rows share it.
    """

    soup = BeautifulSoup(html, "html.parser")

    # Title (usually "FOMC Press Conference" or similar)
    title = ""
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        title = _clean_text(og["content"])
    if not title:
        title_tag = soup.find("title")
        if title_tag:
            title = _clean_text(title_tag.get_text(" ", strip=True))
            title = re.sub(r"\s*-\s*Federal Reserve Board\s*$", "", title, flags=re.IGNORECASE)
    if not title:
        h3 = soup.select_one("h3, h2")
        if h3:
            title = _clean_text(h3.get_text(" ", strip=True))

    # Body — same selector chain as speeches/testimony.
    # Note: as of 2024-2025, press-conference pages are video-only landing pages;
    # the selectors capture video accessibility boilerplate + metadata links.
    # The actual transcript text is in the linked PDF (future enhancement).
    body_chunks: list[str] = []
    for selector in [
        "div.col-xs-12.col-sm-8.col-md-8 p",
        "div#article p",
        "article p",
        "main p",
    ]:
        nodes = soup.select(selector)
        if nodes:
            body_chunks = [_clean_text(node.get_text(" ", strip=True)) for node in nodes]
            break
    body = "\n".join(c for c in body_chunks if c)

    date_iso = _date_from_url(source_url)

    return ParsedPressConference(
        date=date_iso,
        title=title or "FOMC Press Conference",
        text=body,
        url=source_url,
    )


def write_press_conferences_json(
    parsed: Iterable[ParsedPressConference], output_path: Path
) -> int:
    """Write parsed press conferences to output_path as a JSON list.

    Skips rows with empty text or missing date. Tags every row with
    document_type='press_conference'.
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
                "document_type": "press_conference",
                "url": entry.url,
                "scraped_at_utc": scraped_at,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return len(rows)
