"""Federal Reserve regional research scraper.

The 12 Federal Reserve regional banks each publish research material
in different formats. This adapter ships NY Fed Liberty Street Economics
as the primary representative (most market-relevant; clean WordPress-
style blog format with predictable URLs).

URL pattern: https://libertystreeteconomics.newyorkfed.org/{YYYY}/{MM}/{slug}/
The listing is the homepage. Each post has a title, author(s), date,
and body content. Output rows tag source_bank='ny_fed' so a future
extension to other banks (St. Louis Research, Atlanta Macroblog, etc.)
can use the same source_type='regional_research' with different
source_bank values. Other 11 banks are deferred as future work.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup


LSE_BASE_URL = "https://libertystreeteconomics.newyorkfed.org"
LSE_POST_URL_PATTERN = re.compile(
    r"^https://libertystreeteconomics\.newyorkfed\.org/(\d{4})/(\d{2})/[a-z0-9-]+/?$"
)
DATE_FROM_URL_PATTERN = re.compile(
    r"libertystreeteconomics\.newyorkfed\.org/(\d{4})/(\d{2})/"
)


@dataclass(frozen=True)
class RegionalResearchListingEntry:
    date: str  # ISO yyyy-mm-01
    title: str
    url: str


@dataclass(frozen=True)
class ParsedRegionalResearch:
    date: str
    title: str
    text: str
    url: str
    source_bank: str  # 'ny_fed' for Liberty Street; future: 'stlouis_fed', 'atlanta_fed', etc.


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def _date_from_url(url: str) -> str:
    matched = DATE_FROM_URL_PATTERN.search(url)
    if not matched:
        return ""
    return f"{matched.group(1)}-{matched.group(2)}-01"


def extract_lse_listing(html: str) -> list[RegionalResearchListingEntry]:
    """Parse the Liberty Street homepage / archive HTML for post URLs.

    Returns one entry per unique post URL. Entries with non-matching
    URLs (nav, sidebar, etc.) are skipped.
    """

    soup = BeautifulSoup(html, "html.parser")
    entries: list[RegionalResearchListingEntry] = []
    seen: set[str] = set()

    for anchor in soup.select("a[href]"):
        href = (anchor.get("href") or "").strip().rstrip("/")
        if not href:
            continue
        # Normalize trailing slash for the pattern check
        candidate = href if href.endswith("/") else href + "/"
        if not LSE_POST_URL_PATTERN.match(candidate):
            continue
        if candidate in seen:
            continue
        seen.add(candidate)

        title = _clean_text(anchor.get_text(" ", strip=True))
        date_iso = _date_from_url(candidate)
        if not date_iso:
            continue

        entries.append(
            RegionalResearchListingEntry(date=date_iso, title=title, url=candidate)
        )
    return entries


def parse_lse_post(html: str, *, source_url: str) -> ParsedRegionalResearch:
    """Parse a Liberty Street post into a ParsedRegionalResearch.

    The site uses a custom WordPress theme. Title is in
    `<h1 class="ts-blog-article-title">` or `<meta property="og:title">`.
    Body lives in `<div class="ts-article-text">`. Fallbacks include
    standard WordPress `<div class="entry-content">` and `<article>` tags.
    """

    soup = BeautifulSoup(html, "html.parser")

    # --- Title extraction ---
    title = ""
    h1 = soup.select_one("h1.ts-blog-article-title")
    if h1:
        title = _clean_text(h1.get_text(" ", strip=True))
    if not title:
        og = soup.find("meta", attrs={"property": "og:title"})
        if og and og.get("content"):
            title = _clean_text(og["content"])
            title = re.sub(r"\s*[-|]\s*Liberty Street.*$", "", title, flags=re.IGNORECASE)
    if not title:
        h1_generic = soup.select_one("h1.entry-title, h1.post-title, h1")
        if h1_generic:
            title = _clean_text(h1_generic.get_text(" ", strip=True))
    if not title:
        title_tag = soup.find("title")
        if title_tag:
            title = _clean_text(title_tag.get_text(" ", strip=True))
            title = re.sub(r"\s*[-|]\s*Liberty Street.*$", "", title, flags=re.IGNORECASE)
    if not title:
        title = "Liberty Street Economics post"

    # --- Body extraction ---
    body_chunks: list[str] = []
    for selector in [
        "div.ts-article-text p",
        "div.entry-content p",
        "div.post-content p",
        "article p",
        "main p",
    ]:
        nodes = soup.select(selector)
        if len(nodes) >= 3:
            body_chunks = [_clean_text(node.get_text(" ", strip=True)) for node in nodes]
            break

    if not body_chunks:
        # Last resort: all <p> tags
        nodes = soup.select("p")
        body_chunks = [_clean_text(node.get_text(" ", strip=True)) for node in nodes]

    body = "\n".join(c for c in body_chunks if c and len(c) > 20)

    date_iso = _date_from_url(source_url)

    return ParsedRegionalResearch(
        date=date_iso,
        title=title,
        text=body,
        url=source_url,
        source_bank="ny_fed",
    )


def write_regional_research_json(
    parsed: Iterable[ParsedRegionalResearch], output_path: Path
) -> int:
    """Write parsed regional research entries to output_path as a JSON list."""

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
                "document_type": "regional_research",
                "source_bank": entry.source_bank,
                "url": entry.url,
                "scraped_at_utc": scraped_at,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return len(rows)
