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
import time
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

from bs4 import BeautifulSoup

ARCHIVE_BASE_URL = "https://www.federalreserve.gov"
TESTIMONY_URL_PATTERN = re.compile(r"^/newsevents/testimony/[a-z]+(\d{8})[a-z]\.htm$")
DATE_FROM_URL_PATTERN = re.compile(r"/testimony/[a-z]+(\d{8})[a-z]\.htm$")

OUTPUT_FILENAME = "congressional_testimonies.json"
ARCHIVE_LISTING_URL_TEMPLATE = (
    ARCHIVE_BASE_URL + "/newsevents/testimony/{year}-testimony.htm"
)
ARCHIVE_LISTING_URL = ARCHIVE_LISTING_URL_TEMPLATE.format(
    year=datetime.now(timezone.utc).year
)
# Default historical window when ``--years`` is not specified on the CLI.
# The Fed's annual testimony archives go back to ~1996 but the pre-2006
# pages have looser HTML; the canonical FOMC corpus this project trains
# on starts in 2006 so we use that as the default lower bound.
_DEFAULT_YEAR_LOWER = 2006

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
        content_attr = og["content"]
        return _clean_text(
            content_attr if isinstance(content_attr, str) else " ".join(map(str, content_attr))
        )
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
        raw_href = anchor.get("href") or ""
        href = (raw_href if isinstance(raw_href, str) else " ".join(map(str, raw_href))).strip()
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


# federalreserve.gov 403s the stdlib default ``Python-urllib/x.y`` UA,
# so every request needs a real-browser-ish header. Identifying the
# project in the UA keeps the traffic auditable on the upstream's side.
_USER_AGENT = (
    "fed-pulse-data-ingester/1.0 "
    "(+https://github.com/yusufizzetmurat/fed-pulse)"
)


def _http_get_text(url: str, *, timeout: float) -> str:
    """Fetch ``url`` and return the response body as decoded text.

    Wraps stdlib ``urllib.request.urlopen`` so HTTP non-2xx surfaces as
    ``RuntimeError`` (the upstream ``HTTPError`` is re-raised with the
    URL in the message so the caller logs see which page failed). A
    User-Agent header is set explicitly because the federalreserve.gov
    edge rejects requests without one.
    """

    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} from {url}") from exc
    with response:
        body = response.read()
    return body.decode("utf-8", errors="replace")


def _default_years() -> list[int]:
    """Return the default per-year archive coverage window.

    Walks from ``_DEFAULT_YEAR_LOWER`` to the current calendar year
    inclusive. Operators can override with ``--years`` on the CLI.
    """

    current = datetime.now(timezone.utc).year
    return list(range(_DEFAULT_YEAR_LOWER, current + 1))


def pull_testimonies_archive(  # noqa: PLR0913
    target_path: Path,
    *,
    force: bool = False,
    archive_url: str | None = None,
    years: list[int] | None = None,
    limit: int | None = None,
    timeout: float = 30.0,
    delay_seconds: float = 0.5,
) -> int:
    """Walk the federalreserve.gov testimony archives and write parsed rows.

    The Fed publishes one annual archive page per year at
    ``/newsevents/testimony/{year}-testimony.htm``. This orchestrator
    walks every year in ``years`` (defaults to 2006 through the current
    calendar year), discovers individual testimony URLs from each
    annual page, then fetches and parses each testimony. Parsed rows
    are written to ``target_path`` via :func:`write_testimonies_json`
    -- the same JSON shape ``ingest_sources._iter_scraped_records``
    consumes.

    ``archive_url`` short-circuits the year walk: when set, the
    orchestrator treats it as the single listing URL to walk (useful
    for testing). When unset, the per-year template is used.

    Idempotent: when ``target_path`` already exists with a non-empty
    JSON list and ``force`` is False, the existing row count is
    returned without any HTTP traffic. A corrupt or empty file forces
    a re-pull.

    Per-page failures (HTTP error on a single annual archive or
    testimony page, parse miss) are logged via :mod:`warnings` and the
    walk continues -- one bad page must not drop the whole pull.
    ``limit`` caps the walk at the first ``N`` discovered testimonies,
    useful for smoke-testing. A walk that produces zero rows raises
    ``RuntimeError`` so a broken default surfaces loudly.
    """

    if target_path.exists() and not force:
        try:
            cached = json.loads(target_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cached = None
        if isinstance(cached, list) and len(cached) > 0:
            return len(cached)

    if archive_url is not None:
        listing_urls = [archive_url]
    else:
        year_list = years if years is not None else _default_years()
        listing_urls = [
            ARCHIVE_LISTING_URL_TEMPLATE.format(year=y) for y in year_list
        ]

    entries: list[TestimonyListingEntry] = []
    seen_urls: set[str] = set()
    for listing_url in listing_urls:
        try:
            listing_html = _http_get_text(listing_url, timeout=timeout)
        except Exception as exc:
            # Listing-page failures degrade gracefully -- one missing
            # year (e.g. a future year before the page is generated)
            # must not abort the walk. The aggregate zero-rows check
            # below still catches a fully broken default.
            warnings.warn(
                f"Testimony listing fetch failed for {listing_url}: {exc}",
                stacklevel=2,
            )
            continue
        for entry in extract_testimony_listing(listing_html):
            if entry.url in seen_urls:
                continue
            seen_urls.add(entry.url)
            entries.append(entry)

    if limit is not None:
        entries = entries[:limit]

    parsed: list[ParsedTestimony] = []
    for i, entry in enumerate(entries):
        try:
            page_html = _http_get_text(entry.url, timeout=timeout)
            parsed.append(parse_testimony_page(page_html, source_url=entry.url))
        except Exception as exc:
            warnings.warn(
                f"Testimony fetch failed for {entry.url}: {exc}",
                stacklevel=2,
            )
            continue
        if delay_seconds > 0 and i + 1 < len(entries):
            time.sleep(delay_seconds)

    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    try:
        written = write_testimonies_json(parsed, tmp_path)
        if written == 0:
            raise RuntimeError(
                f"Testimonies pull produced zero rows (listings tried: "
                f"{len(listing_urls)})"
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
            "Walk the federalreserve.gov testimony archives and write "
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
        help="Stop after fetching N testimonies (smoke testing).",
    )
    parser.add_argument(
        "--archive-url",
        default=None,
        help=(
            "Override the listing URL. When set, the orchestrator walks "
            "this single URL instead of the per-year template."
        ),
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Years to walk via the per-year template (default: 2006 "
            "through the current calendar year)."
        ),
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.5,
        help="Polite delay between page fetches (default: 0.5s).",
    )
    ns = parser.parse_args()
    target = Path(ns.data_dir) / OUTPUT_FILENAME
    rows = pull_testimonies_archive(
        target,
        force=ns.force,
        archive_url=ns.archive_url,
        years=ns.years,
        limit=ns.limit,
        delay_seconds=ns.delay_seconds,
    )
    print(f"Congressional testimonies cache at {target} (rows: {rows})")
