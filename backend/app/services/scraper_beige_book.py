"""Federal Reserve Beige Book scraper.

The Beige Book is a regional economic survey published 8 times a year
(one per FOMC meeting). Each issue contains a national summary plus
12 district reports. The federalreserve.gov URL scheme uses base landing
pages at /monetarypolicy/beigebook{YYYYMM}.htm (or date variants like
beigebook20230531.htm) which contain only boilerplate "About This
Publication" text. The substantive content lives at:
  - /monetarypolicy/beigebook{YYYYMM}-summary.htm  (national summary)
  - /monetarypolicy/beigebook{YYYYMM}-{district}.htm  (district reports)

This adapter reads the archive listing at
/monetarypolicy/publications/beige-book-default.htm, discovers issue
base URLs (from both direct <a href> links and the commented-out sitemap
list embedded in the page), and derives the summary URL for each issue.
The national summary is used as the canonical content page — it contains
the cross-district synthesis document which is the most analytically
relevant single page for sentiment analysis.
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

# Matches any beigebook URL that looks like an issue page:
#   beigebook202401.htm           (6-digit YYYYMM)
#   beigebook20230531.htm         (8-digit YYYYMMDD, older convention)
#   beigebook202401-summary.htm   (national summary direct link from listing)
# Deliberately excludes:
#   beige-book-faqs.htm, beige-book-archive.htm  (informational pages)
#   beigebook202401-boston.htm  (district sub-pages — derived, not listed)
BEIGE_BOOK_ISSUE_PATTERN = re.compile(
    r"^/monetarypolicy/beigebook(\d{6}|\d{8})(-summary)?\.htm$"
)

# For extracting date from any beigebook URL variant
DATE_FROM_URL_PATTERN = re.compile(r"/beigebook(\d{6,8})(?:-[a-z-]+)?\.htm")


@dataclass(frozen=True)
class BeigeBookListingEntry:
    date: str  # ISO yyyy-mm-dd
    url: str   # URL of the national summary page for this issue


@dataclass(frozen=True)
class ParsedBeigeBook:
    date: str
    title: str
    text: str
    url: str


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def _date_from_url(url: str) -> str:
    """Extract an ISO date string from any beigebook URL variant.

    Handles:
      beigebook202401.htm          → 2024-01-01
      beigebook202401-summary.htm  → 2024-01-01
      beigebook20230531.htm        → 2023-05-31
    """
    matched = DATE_FROM_URL_PATTERN.search(url)
    if not matched:
        return ""
    digits = matched.group(1)
    if len(digits) == 6:
        return f"{digits[:4]}-{digits[4:6]}-01"
    if len(digits) == 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return ""


def _base_to_summary_url(base_url: str) -> str:
    """Convert a beigebook base URL to its national summary URL.

    beigebook202401.htm          → beigebook202401-summary.htm
    beigebook202401-summary.htm  → beigebook202401-summary.htm  (idempotent)
    """
    if base_url.endswith("-summary.htm"):
        return base_url
    # Strip .htm and append -summary.htm
    return base_url[:-4] + "-summary.htm"


def extract_beige_book_listing(html: str) -> list[BeigeBookListingEntry]:
    """Parse the Beige Book archive page and return one entry per issue.

    Discovers issue URLs from two sources in the listing HTML:
      1. Direct <a href> anchor tags (recent issues)
      2. Commented-out sitemap list embedded by the CMS (full archive)

    Returns entries whose URL points to the national summary page for each
    issue. Entries without a parseable date are silently skipped.
    Duplicate issues (same date/URL) are deduplicated.
    """
    soup = BeautifulSoup(html, "html.parser")
    seen_dates: set[str] = set()
    entries: list[BeigeBookListingEntry] = []

    def _add(href: str) -> None:
        href = href.strip()
        if not BEIGE_BOOK_ISSUE_PATTERN.match(href):
            return
        summary_path = _base_to_summary_url(href)
        absolute = urljoin(ARCHIVE_BASE_URL, summary_path)
        date_iso = _date_from_url(absolute)
        if not date_iso or date_iso in seen_dates:
            return
        seen_dates.add(date_iso)
        entries.append(BeigeBookListingEntry(date=date_iso, url=absolute))

    # Source 1: explicit <a href> links
    for anchor in soup.select("a[href]"):
        _add(anchor.get("href") or "")

    # Source 2: commented-out sitemap list embedded by the CMS
    # The listing page contains HTML comments like:
    #   <!--<ul><li>/monetarypolicy/beigebook202604.htm</li>...-->
    raw_html = html
    for comment_match in re.finditer(r"<!--(.*?)-->", raw_html, re.DOTALL):
        comment_body = comment_match.group(1)
        for path_match in re.finditer(r"/monetarypolicy/beigebook[^<\"'\s]+\.htm", comment_body):
            _add(path_match.group(0))

    return entries


def parse_beige_book_page(html: str, *, source_url: str) -> ParsedBeigeBook:
    """Parse a Beige Book national summary (or district) page.

    Extracts the page title and substantive body text from the FRB content
    area (div#article). The national summary page contains the cross-district
    synthesis which is the most analytically relevant content for sentiment
    analysis. District pages follow the same HTML structure.
    """
    soup = BeautifulSoup(html, "html.parser")

    # --- Title ---
    title = ""
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        title = _clean_text(str(og["content"]))
    if not title:
        title_tag = soup.find("title")
        if title_tag:
            title = _clean_text(title_tag.get_text(" ", strip=True))
            title = re.sub(
                r"\s*-\s*Federal Reserve Board\s*$", "", title, flags=re.IGNORECASE
            )
            title = re.sub(
                r"^The Fed\s*-\s*", "", title, flags=re.IGNORECASE
            )
    if not title:
        h_tag = soup.select_one("h1, h2, h3")
        if h_tag:
            title = _clean_text(h_tag.get_text(" ", strip=True))
    if not title:
        title = "Beige Book"

    # Prefix with "Beige Book" if not already present
    if "beige book" not in title.lower() and "beigebook" not in title.lower():
        title = f"Beige Book - {title}"

    # --- Body text ---
    # Try progressively broader selectors; stop when we have substantive content
    body_chunks: list[str] = []

    for selector in [
        "div#article p",
        "div.col-xs-12.col-md-9 p",
        "article p",
        "main p",
    ]:
        nodes = soup.select(selector)
        chunks = [_clean_text(n.get_text(" ", strip=True)) for n in nodes]
        chunks = [c for c in chunks if c and len(c) > 20]
        if len(chunks) >= 3:
            body_chunks = chunks
            break

    if not body_chunks:
        # Last resort: every <p> in the document
        all_p = soup.select("p")
        body_chunks = [
            _clean_text(n.get_text(" ", strip=True))
            for n in all_p
            if _clean_text(n.get_text(" ", strip=True)) and len(_clean_text(n.get_text(" ", strip=True))) > 20
        ]

    body = "\n".join(body_chunks)
    date_iso = _date_from_url(source_url)

    return ParsedBeigeBook(
        date=date_iso,
        title=title,
        text=body,
        url=source_url,
    )


def write_beige_book_json(parsed: Iterable[ParsedBeigeBook], output_path: Path) -> int:
    """Write parsed Beige Book entries to a JSON file.

    Rows with empty date or text are skipped. Returns the number of rows
    written. The output format mirrors other scraped JSON files so that
    ingest_sources._iter_scraped_records can ingest it unchanged.
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
                "document_type": "beige_book",
                "url": entry.url,
                "scraped_at_utc": scraped_at,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return len(rows)
