"""FOMC press-conference scraper.

The /monetarypolicy/fomcpresconf{YYYYMMDD}.htm page is a video-only
landing page with no substantive transcript text. The actual transcript
is published as a PDF at /mediacenter/files/FOMCpresconf{date}.pdf.
parse_press_conference_page constructs the PDF URL from the HTML URL,
downloads it, extracts text via pypdf, and returns the transcript.
On download or extraction failure the text is empty — the caller
(write_press_conferences_json) skips empty-text rows.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
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
        raw_href = anchor.get("href") or ""
        href = (raw_href if isinstance(raw_href, str) else " ".join(map(str, raw_href))).strip()
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


def _pdf_url_from_html_url(source_url: str) -> str:
    """Convert /monetarypolicy/fomcpresconf{date}.htm to /mediacenter/files/FOMCpresconf{date}.pdf."""

    match = DATE_FROM_URL_PATTERN.search(source_url)
    if not match:
        return ""
    date_yyyymmdd = match.group(1)
    return f"{ARCHIVE_BASE_URL}/mediacenter/files/FOMCpresconf{date_yyyymmdd}.pdf"


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract concatenated text from a PDF byte stream via pypdf.

    Returns empty string on any extraction failure — the caller decides
    whether to keep the row.
    """

    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(pdf_bytes))
        chunks: list[str] = []
        for page in reader.pages:
            try:
                chunks.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(c for c in chunks if c).strip()
    except Exception:
        return ""


def parse_press_conference_page(html: str, *, source_url: str) -> ParsedPressConference:
    """Parse a press-conference page.

    The HTML page is a video-only landing page; the substantive
    transcript lives in a sibling PDF at /mediacenter/files/FOMCpresconf{date}.pdf.
    This function constructs the PDF URL, downloads it, extracts text
    via pypdf, and returns the extracted text as `parsed.text`. On any
    download / extraction failure, returns empty text — the caller can
    decide whether to keep the row.
    """

    soup = BeautifulSoup(html, "html.parser")

    # Title (best-effort from HTML)
    title = ""
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        content_attr = og["content"]
        title = _clean_text(
            content_attr if isinstance(content_attr, str) else " ".join(map(str, content_attr))
        )
    if not title:
        title_tag = soup.find("title")
        if title_tag:
            title = _clean_text(title_tag.get_text(" ", strip=True))
            title = re.sub(r"\s*-\s*Federal Reserve Board\s*$", "", title, flags=re.IGNORECASE)
    if not title:
        title = "FOMC Press Conference"

    date_iso = _date_from_url(source_url)

    # Fetch and extract the PDF transcript
    text = ""
    pdf_url = _pdf_url_from_html_url(source_url)
    if pdf_url:
        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            text = _extract_pdf_text(response.content)
        except Exception:
            text = ""

    return ParsedPressConference(
        date=date_iso,
        title=title,
        text=text,
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
