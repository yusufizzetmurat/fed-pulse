"""Fed speech archive scraper.

Two responsibilities (Task 2 lands the first; Task 3 lands the second):
1. List speeches from the annual archive page (extract_speech_listing).
2. Parse a single speech page into a structured row (parse_speech_page).

Output rows match the schema used by services/scraper.py so the existing
ingestion pipeline picks them up without changes when wired through
ingest_sources.SCRAPED_FILES.
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
SPEECH_URL_PATTERN = re.compile(r"^/newsevents/speech/[a-z]+(\d{8})[a-z]\.htm$")
DATE_FROM_URL_PATTERN = re.compile(r"/speech/[a-z]+(\d{8})[a-z]\.htm$")

# Two distinct output files: one for chair speeches, one for governor
# speeches. The same parsed-entry list is filtered through both writers,
# which key on the speaker's title to assign each entry to the right
# sub-corpus. ``ingest_sources.SCRAPED_FILES`` lists both filenames so
# the downstream record iterator picks them up unchanged.
CHAIR_OUTPUT_FILENAME = "chair_speeches.json"
GOVERNOR_OUTPUT_FILENAME = "governor_speeches.json"
ARCHIVE_LISTING_URL_TEMPLATE = (
    ARCHIVE_BASE_URL + "/newsevents/speech/{year}-speeches.htm"
)
ARCHIVE_LISTING_URL = ARCHIVE_LISTING_URL_TEMPLATE.format(
    year=datetime.now(timezone.utc).year
)
# Default historical window when ``--years`` is not specified on the
# CLI. The Fed's annual speech archives go back to 1996; the canonical
# FOMC corpus this project trains on starts in 2006 so we use that as
# the default lower bound. The window collapses to a single year via
# ``--years 2025``.
_DEFAULT_YEAR_LOWER = 2006

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


@dataclass(frozen=True)
class ParsedSpeech:
    date: str  # ISO yyyy-mm-dd
    speaker: str
    title: str
    text: str
    url: str


_TITLE_TAIL_PATTERN = re.compile(r"\s*-\s*Federal Reserve Board\s*$", flags=re.IGNORECASE)


def _extract_title(soup: BeautifulSoup) -> str:
    # Prefer the og:title meta when present (the rendered page title without
    # the " - Federal Reserve Board" tail). Fall back to <title> with the
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
    # Fall back to extracting "Chair Powell" / "Governor Waller" out of the title.
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


def parse_speech_page(html: str, *, source_url: str) -> ParsedSpeech:
    """Parse a single federalreserve.gov speech page into a ParsedSpeech.

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

    return ParsedSpeech(
        date=date_iso,
        speaker=speaker,
        title=title,
        text=body,
        url=source_url,
    )


def extract_speech_listing(html: str) -> list[SpeechListingEntry]:
    """Parse a federalreserve.gov annual speech archive page.

    Returns one SpeechListingEntry per linked speech. Duplicate URLs
    collapse to the first occurrence; non-speech anchors are skipped.
    """

    soup = BeautifulSoup(html, "html.parser")
    entries: list[SpeechListingEntry] = []
    seen_urls: set[str] = set()

    for anchor in soup.select("a[href]"):
        raw_href = anchor.get("href") or ""
        href = (raw_href if isinstance(raw_href, str) else " ".join(map(str, raw_href))).strip()
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


_CHAIR_PATTERN = re.compile(r"\b(chair|chairman|chairwoman)\b", flags=re.IGNORECASE)
_VICE_CHAIR_PATTERN = re.compile(r"\bvice\s+(chair|chairman|chairwoman)\b", flags=re.IGNORECASE)
_GOVERNOR_PATTERN = re.compile(r"\bgovernor\b", flags=re.IGNORECASE)


def is_chair_speech(speaker: str) -> bool:
    """Classify a speaker as the Chair/Chairman/Chairwoman (excluding Vice Chair).

    Returns True if the speaker string contains chair/chairman/chairwoman and
    does NOT contain "vice chair/chairman/chairwoman".
    """
    if not speaker:
        return False
    if _VICE_CHAIR_PATTERN.search(speaker):
        return False
    return bool(_CHAIR_PATTERN.search(speaker))


def write_chair_speeches_json(parsed: Iterable[ParsedSpeech], output_path: Path) -> int:
    """Write only chair speeches to output_path as a JSON list.

    Returns the number of rows written. Each row matches the schema
    consumed by ingest_sources._iter_scraped_records: date, title, text,
    document_type ('chair_speech'), url, scraped_at_utc.
    """

    rows: list[dict[str, str]] = []
    scraped_at = datetime.now(timezone.utc).isoformat()
    for entry in parsed:
        if not is_chair_speech(entry.speaker):
            continue
        if not entry.text or not entry.date:
            continue
        rows.append(
            {
                "date": entry.date,
                "title": entry.title,
                "text": entry.text,
                "document_type": "chair_speech",
                "url": entry.url,
                "scraped_at_utc": scraped_at,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return len(rows)


def is_governor_speech(speaker: str) -> bool:
    """True iff the speaker is a Fed Governor and NOT a Chair or Vice Chair.

    This filters the speech archive by speaker type. Vice-Chair-rank governors
    (e.g. Vice Chair Brainard) are excluded — they belong to neither the
    chair nor the governor slot for a clean three-way split (chair / governor /
    vice-chair). The vice-chair set is small and lands as a follow-up source
    if the project document needs to distinguish it.
    """

    if not speaker:
        return False
    if _CHAIR_PATTERN.search(speaker):
        return False  # excludes Chair / Chairman / Chairwoman / Vice Chair / Vice Chairman / Vice Chairwoman
    return bool(_GOVERNOR_PATTERN.search(speaker))


def write_governor_speeches_json(parsed: Iterable[ParsedSpeech], output_path: Path) -> int:
    """Write only governor speeches to output_path as a JSON list.

    Mirrors write_chair_speeches_json but filters via is_governor_speech and
    tags rows with document_type='governor_speech'.
    """

    rows: list[dict[str, str]] = []
    scraped_at = datetime.now(timezone.utc).isoformat()
    for entry in parsed:
        if not is_governor_speech(entry.speaker):
            continue
        if not entry.text or not entry.date:
            continue
        rows.append(
            {
                "date": entry.date,
                "title": entry.title,
                "text": entry.text,
                "document_type": "governor_speech",
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


def pull_speeches_archive(  # noqa: PLR0913
    chair_target_path: Path,
    governor_target_path: Path,
    *,
    force: bool = False,
    archive_url: str | None = None,
    years: list[int] | None = None,
    limit: int | None = None,
    timeout: float = 30.0,
    delay_seconds: float = 0.5,
) -> int:
    """Walk the federalreserve.gov speech archives and write the chair + governor JSONs.

    The Fed publishes one annual archive page per year at
    ``/newsevents/speech/{year}-speeches.htm`` (the master
    ``/newsevents/speeches.htm`` is a 301 redirect to the latest year,
    so we generate year URLs over a range rather than scraping it).
    This orchestrator walks every year in ``years`` (defaults to 2006
    through the current calendar year), discovers individual speech
    URLs from each annual page, then fetches and parses each speech.

    Because the speech corpus is split into two output files keyed by
    the speaker's title (chair vs governor), the same parsed-entry
    list is funnelled through BOTH
    :func:`write_chair_speeches_json` and
    :func:`write_governor_speeches_json` -- two JSON files land in the
    operator's data directory. The function returns the total row
    count across both files so the CLI can print one summary line.

    Idempotency check looks at BOTH cache files: when both exist with
    non-empty JSON lists and ``force`` is False, the combined row
    count is returned without any HTTP traffic. A corrupt or empty
    file on either side forces a re-pull of both.

    ``archive_url`` short-circuits the year walk: when set, the
    orchestrator treats it as the single listing URL to walk (useful
    for testing). Per-page failures (listing or detail) are logged via
    :mod:`warnings` and the walk continues; a walk producing zero
    total rows raises ``RuntimeError``.
    """

    if (
        chair_target_path.exists()
        and governor_target_path.exists()
        and not force
    ):
        try:
            chair_cached = json.loads(chair_target_path.read_text(encoding="utf-8"))
            gov_cached = json.loads(governor_target_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            chair_cached = None
            gov_cached = None
        if (
            isinstance(chair_cached, list)
            and isinstance(gov_cached, list)
            and (len(chair_cached) + len(gov_cached)) > 0
        ):
            return len(chair_cached) + len(gov_cached)

    if archive_url is not None:
        listing_urls = [archive_url]
    else:
        year_list = years if years is not None else _default_years()
        listing_urls = [
            ARCHIVE_LISTING_URL_TEMPLATE.format(year=y) for y in year_list
        ]

    entries: list[SpeechListingEntry] = []
    seen_urls: set[str] = set()
    for listing_url in listing_urls:
        try:
            listing_html = _http_get_text(listing_url, timeout=timeout)
        except Exception as exc:
            warnings.warn(
                f"Speech listing fetch failed for {listing_url}: {exc}",
                stacklevel=2,
            )
            continue
        for entry in extract_speech_listing(listing_html):
            if entry.url in seen_urls:
                continue
            seen_urls.add(entry.url)
            entries.append(entry)

    if limit is not None:
        entries = entries[:limit]

    parsed: list[ParsedSpeech] = []
    for i, entry in enumerate(entries):
        try:
            page_html = _http_get_text(entry.url, timeout=timeout)
            single = parse_speech_page(page_html, source_url=entry.url)
            # The detail page may not always carry the speaker; fall
            # back to the listing-entry speaker when the parsed row is
            # empty, so the chair/governor classifier has something to
            # match on.
            if not single.speaker and entry.speaker:
                single = ParsedSpeech(
                    date=single.date or entry.date,
                    speaker=entry.speaker,
                    title=single.title or entry.title,
                    text=single.text,
                    url=single.url,
                )
            parsed.append(single)
        except Exception as exc:
            warnings.warn(
                f"Speech fetch failed for {entry.url}: {exc}",
                stacklevel=2,
            )
            continue
        if delay_seconds > 0 and i + 1 < len(entries):
            time.sleep(delay_seconds)

    chair_tmp = chair_target_path.with_suffix(chair_target_path.suffix + ".tmp")
    gov_tmp = governor_target_path.with_suffix(governor_target_path.suffix + ".tmp")
    try:
        chair_written = write_chair_speeches_json(parsed, chair_tmp)
        gov_written = write_governor_speeches_json(parsed, gov_tmp)
        if (chair_written + gov_written) == 0:
            raise RuntimeError(
                f"Speeches pull produced zero rows (listings tried: "
                f"{len(listing_urls)})"
            )
        chair_tmp.replace(chair_target_path)
        gov_tmp.replace(governor_target_path)
        return chair_written + gov_written
    except Exception:
        for tmp in (chair_tmp, gov_tmp):
            if tmp.exists():
                tmp.unlink()
        raise


if __name__ == "__main__":
    import argparse

    from app.config import DATA_DIR as _DEFAULT_DATA_DIR

    parser = argparse.ArgumentParser(
        description=(
            "Walk the federalreserve.gov speech archives and write both "
            f"{CHAIR_OUTPUT_FILENAME} and {GOVERNOR_OUTPUT_FILENAME} into "
            "the data directory. The resulting JSONs are picked up "
            "unchanged by `python -m app.data.ingest_sources --include-scraped`."
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
        help="Re-pull even if both cache files already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after fetching N speeches (smoke testing).",
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
    data_dir = Path(ns.data_dir)
    chair_target = data_dir / CHAIR_OUTPUT_FILENAME
    governor_target = data_dir / GOVERNOR_OUTPUT_FILENAME
    rows = pull_speeches_archive(
        chair_target,
        governor_target,
        force=ns.force,
        archive_url=ns.archive_url,
        years=ns.years,
        limit=ns.limit,
        delay_seconds=ns.delay_seconds,
    )
    print(
        f"Speeches cache at {chair_target} + {governor_target} "
        f"(rows: {rows})"
    )
