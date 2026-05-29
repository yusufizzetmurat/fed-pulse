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
import time
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup


LSE_BASE_URL = "https://libertystreeteconomics.newyorkfed.org"
ARCHIVE_LISTING_URL = LSE_BASE_URL + "/"
OUTPUT_FILENAME = "regional_research.json"

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
        raw_href = anchor.get("href") or ""
        href = (
            raw_href if isinstance(raw_href, str) else " ".join(map(str, raw_href))
        ).strip().rstrip("/")
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
            content_attr = og["content"]
            title = _clean_text(
                content_attr if isinstance(content_attr, str) else " ".join(map(str, content_attr))
            )
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


# Liberty Street Economics is fronted by a CDN that 403s the stdlib
# default ``Python-urllib/x.y`` UA on some edge nodes. Identifying the
# project in the UA keeps the traffic auditable on the upstream's side
# and matches the convention the federalreserve.gov scrapers use.
_USER_AGENT = (
    "fed-pulse-data-ingester/1.0 "
    "(+https://github.com/yusufizzetmurat/fed-pulse)"
)


def _http_get_text(url: str, *, timeout: float) -> str:
    """Fetch ``url`` and return the response body as decoded text.

    Wraps stdlib ``urllib.request.urlopen`` so HTTP non-2xx surfaces as
    ``RuntimeError`` (the upstream ``HTTPError`` is re-raised with the URL
    in the message so the caller logs see which post page failed).
    """

    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} from {url}") from exc
    with response:
        body: bytes = response.read()
    return body.decode("utf-8", errors="replace")


def pull_regional_research_archive(  # noqa: PLR0913
    target_path: Path,
    *,
    force: bool = False,
    archive_url: str = ARCHIVE_LISTING_URL,
    limit: int | None = None,
    timeout: float = 30.0,
    delay_seconds: float = 0.5,
) -> int:
    """Walk the Liberty Street Economics archive and write parsed rows.

    Fetches the archive listing (defaults to the site homepage, which
    surfaces the most recent posts), discovers every post URL matching
    the canonical ``/{YYYY}/{MM}/{slug}/`` pattern, then fetches and
    parses each post page. Parsed rows are written to ``target_path``
    via :func:`write_regional_research_json` -- the same JSON shape
    ``ingest_sources._iter_scraped_records`` consumes.

    Idempotent: when ``target_path`` already exists with a non-empty
    JSON list and ``force`` is False, the existing row count is
    returned without any HTTP traffic. A corrupt or empty file forces
    a re-pull.

    Per-post failures (HTTP error on a single post page, parse miss)
    are logged via :mod:`warnings` and the walk continues -- one bad
    post page must not drop the whole pull. ``limit`` caps the walk at
    the first ``N`` discovered posts, useful for smoke-testing.
    """

    if target_path.exists() and not force:
        try:
            cached = json.loads(target_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cached = None
        if isinstance(cached, list) and len(cached) > 0:
            return len(cached)

    listing_html = _http_get_text(archive_url, timeout=timeout)
    entries = extract_lse_listing(listing_html)
    if limit is not None:
        entries = entries[:limit]

    parsed: list[ParsedRegionalResearch] = []
    for i, entry in enumerate(entries):
        try:
            page_html = _http_get_text(entry.url, timeout=timeout)
            parsed.append(parse_lse_post(page_html, source_url=entry.url))
        except Exception as exc:
            warnings.warn(
                f"Regional research fetch failed for {entry.url}: {exc}",
                stacklevel=2,
            )
            continue
        # Polite delay between page fetches so a long walk does not
        # trigger an upstream throttle. Skipped after the last entry.
        if delay_seconds > 0 and i + 1 < len(entries):
            time.sleep(delay_seconds)

    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    try:
        written = write_regional_research_json(parsed, tmp_path)
        if written == 0:
            raise RuntimeError(
                f"Regional research pull from {archive_url} produced zero rows"
            )
        tmp_path.replace(target_path)
        return written
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


if __name__ == "__main__":
    import argparse

    from app.config import DATA_DIR as _DEFAULT_DATA_DIR

    parser = argparse.ArgumentParser(
        description=(
            "Walk the Liberty Street Economics archive and write "
            f"{OUTPUT_FILENAME} into the data directory. The resulting "
            "JSON is picked up unchanged by "
            "`python -m app.data.ingest_sources --include-scraped`."
        )
    )
    parser.add_argument(
        "--data-dir",
        default=str(_DEFAULT_DATA_DIR),
        help="Base data directory (default: app.config.DATA_DIR).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-pull even if the cache file already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after fetching N posts (smoke testing).",
    )
    parser.add_argument(
        "--archive-url",
        default=ARCHIVE_LISTING_URL,
        help="Override the archive listing URL.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.5,
        help="Polite delay between page fetches (default: 0.5s).",
    )
    ns = parser.parse_args()
    target = Path(ns.data_dir) / OUTPUT_FILENAME
    rows = pull_regional_research_archive(
        target,
        force=ns.force,
        archive_url=ns.archive_url,
        limit=ns.limit,
        delay_seconds=ns.delay_seconds,
    )
    print(f"Regional research cache at {target} (rows: {rows})")
