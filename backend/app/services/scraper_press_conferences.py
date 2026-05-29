"""FOMC press-conference scraper.

The /monetarypolicy/fomcpresconf{YYYYMMDD}.htm page is a video-only
landing page with no substantive transcript text. The actual transcript
is published as a PDF at /mediacenter/files/FOMCpresconf{date}.pdf.
parse_press_conference_page constructs the PDF URL from the HTML URL,
downloads it, extracts text via pypdf, and returns the transcript.
On download or extraction failure the text is empty — the caller
(write_press_conferences_json) skips empty-text rows.

#214 (Q&A separation): the press conference is the prepared remarks
followed by the journalist Q&A. The Q&A is the higher-information slice
— Powell's unscripted answers, dated `T (snapshot)` on the same FOMC
event as the statement. ``split_prepared_remarks_and_qa`` splits the
extracted PDF text on the moderator hand-off ("I look forward to your
questions" / "Michelle Smith, the moderator's name") so the two
sub-corpora can be carried separately on ``ParsedPressConference``.
``qa_text`` is empty when the boundary cannot be located, which lets
the caller treat the row as prepared-remarks-only without poisoning
downstream features.

A local PDF cache is written under ``data/raw/fomc_press_conferences/``
when ``cache_pdf_dir`` is passed; the bytes are re-used on subsequent
runs so the scraper does not re-pull the 250+ KB PDF per re-encode.
"""

from __future__ import annotations

import json
import re
import time
import warnings
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

OUTPUT_FILENAME = "press_conferences.json"
# Recent press-conference URLs live on the FOMC calendars page. The
# orchestrator can additionally walk per-year historical pages
# (/monetarypolicy/fomchistorical{year}.htm) to back-fill pre-2021
# events; the calendar by itself covers 2021 forward only.
ARCHIVE_LISTING_URL = ARCHIVE_BASE_URL + "/monetarypolicy/fomccalendars.htm"
HISTORICAL_YEAR_URL_TEMPLATE = (
    ARCHIVE_BASE_URL + "/monetarypolicy/fomchistorical{year}.htm"
)
# Press conferences started in April 2011. The default historical
# window walks calendar + 2011-2020 historical pages so the cache
# covers the full press-conf era.
_DEFAULT_HISTORICAL_YEARS_LOWER = 2011
_DEFAULT_HISTORICAL_YEARS_UPPER = 2020

# Page-header noise stamped on every page of the Federal Reserve press
# conference PDFs ("January 29, 2025   Chair Powell's Press Conference
# FINAL\nPage 4 of 27"). Stripped before Q&A splitting so the speaker-turn
# regex does not see the running header between two journalist lines.
# The apostrophe class covers ASCII ', U+2019, and U+2018 — pdf
# extraction surfaces any of the three depending on the font.
_PAGE_HEADER_RE = re.compile(
    r"\n\s*[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\s+Chair\s+\w+[‘’']s\s+Press\s+Conference\s+FINAL\s*"
    r"\n\s*\n\s*Page\s+\d+\s+of\s+\d+\s*\n\s*",
    re.IGNORECASE,
)

# Boundary phrases the chair has used over the years to hand off to the
# moderator. The first match anchors the start of the Q&A; the search is
# scoped to the first ~30% of the document so a stray "your questions"
# inside an answer does not move the split.
_QA_BOUNDARY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bI\s+look\s+forward\s+to\s+your\s+questions\b", re.IGNORECASE),
    re.compile(r"\bI'?ll\s+be\s+happy\s+to\s+take\s+your\s+questions\b", re.IGNORECASE),
    re.compile(r"\bhappy\s+to\s+take\s+your\s+questions\b", re.IGNORECASE),
    re.compile(r"\bwe(?:'ll|\s+will)\s+take\s+your\s+questions\b", re.IGNORECASE),
    re.compile(r"\bopen\s+(?:it|things)\s+up\s+(?:to|for)\s+questions\b", re.IGNORECASE),
)

# Fallback Q&A anchor: the moderator hand-off line ("MICHELLE SMITH.
# Steve."). The moderator turn label is one of a small set the Fed has
# rotated through over the years; the speaker-turn pattern below covers
# both ``Michelle`` and ``Michael`` variants while ignoring the chair's
# own all-caps label.
_MODERATOR_HANDOFF_RE = re.compile(
    r"\n[A-Z][A-Z'’\-]+\s+[A-Z][A-Z'’\-]+\.\s+[A-Z][a-z]+\.\s*\n",
)


@dataclass(frozen=True)
class PressConferenceListingEntry:
    date: str
    url: str


@dataclass(frozen=True)
class ParsedPressConference:
    """One press conference's parsed payload.

    ``text`` carries the full extracted PDF text (back-compat with the
    pre-#214 callers). ``prepared_remarks_text`` and ``qa_text`` are the
    split sub-corpora — ``qa_text`` is the high-information slice the
    #214 joint-corpus methodology targets. Both default to empty strings
    so a pre-#214 fixture round-trips through this dataclass without
    forcing the caller to compute the split.
    """

    date: str
    title: str
    text: str
    url: str
    prepared_remarks_text: str = ""
    qa_text: str = ""


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


def _strip_page_headers(text: str) -> str:
    """Remove the per-page running header from the joined PDF text.

    The Fed's PDF template stamps the running title and page index on
    every page; pypdf concatenates them verbatim, which leaves a noisy
    boilerplate band between any two speaker turns that wrap across a
    page boundary. Stripping the band before Q&A splitting keeps the
    speaker-turn regex from anchoring on the running header.
    """

    return _PAGE_HEADER_RE.sub("\n", text)


def split_prepared_remarks_and_qa(text: str) -> tuple[str, str]:
    """Split a Powell press-conference transcript into (remarks, Q&A).

    The split anchors on the chair's hand-off line ("I look forward to
    your questions" and variants); when the canonical phrase is absent
    the helper falls back to the first moderator hand-off ("MICHELLE
    SMITH. Steve."). When neither anchor is locatable the whole text is
    returned as prepared remarks and the Q&A is empty — the caller
    treats that case as a parse miss and downstream features collapse
    to the pre-2011 covariate-shift handling (``has_press_conf=0`` +
    zero-imputed Q&A vector). Both sides are stripped; empty input
    returns ``("", "")``.

    The search is anchored in the front half of the document so a stray
    "your questions" mid-Q&A does not move the boundary backwards.
    """

    if not text:
        return ("", "")
    stripped_pages = _strip_page_headers(text)
    head_limit = max(1, len(stripped_pages) // 2)
    boundary_end: int | None = None
    head = stripped_pages[:head_limit]
    for pattern in _QA_BOUNDARY_PATTERNS:
        match = pattern.search(head)
        if match is not None:
            boundary_end = match.end()
            break
    if boundary_end is None:
        moderator = _MODERATOR_HANDOFF_RE.search(stripped_pages, 0, head_limit)
        if moderator is not None:
            boundary_end = moderator.start()
    if boundary_end is None:
        return (stripped_pages.strip(), "")
    prepared = stripped_pages[:boundary_end].strip()
    qa = stripped_pages[boundary_end:].strip()
    return (prepared, qa)


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


def _date_yyyymmdd_from_url(source_url: str) -> str:
    match = DATE_FROM_URL_PATTERN.search(source_url)
    if not match:
        return ""
    return match.group(1)


def _load_or_fetch_pdf_bytes(
    pdf_url: str, *, cache_pdf_dir: Path | None, date_yyyymmdd: str
) -> bytes:
    """Return PDF bytes for the press conference, hitting a local cache when set.

    When ``cache_pdf_dir`` is provided and a prior run already saved
    ``{date_yyyymmdd}.pdf`` there, the cached bytes are returned without
    a network call. On cache miss the helper fetches the PDF and writes
    it under ``cache_pdf_dir`` for the next run. Network errors return
    empty bytes; the caller then leaves the parsed text empty so the
    row is skipped downstream.
    """

    if cache_pdf_dir is not None and date_yyyymmdd:
        cached = cache_pdf_dir / f"{date_yyyymmdd}.pdf"
        if cached.exists():
            try:
                return cached.read_bytes()
            except OSError:
                pass
    if not pdf_url:
        return b""
    try:
        response = requests.get(
            pdf_url,
            timeout=30,
            headers={"User-Agent": _USER_AGENT},
        )
        response.raise_for_status()
    except Exception:
        return b""
    content = response.content
    if cache_pdf_dir is not None and date_yyyymmdd and content:
        try:
            cache_pdf_dir.mkdir(parents=True, exist_ok=True)
            (cache_pdf_dir / f"{date_yyyymmdd}.pdf").write_bytes(content)
        except OSError:
            # Cache write failures are non-fatal; the bytes are still
            # returned for in-memory parsing.
            pass
    return content


def parse_press_conference_page(
    html: str,
    *,
    source_url: str,
    cache_pdf_dir: Path | None = None,
) -> ParsedPressConference:
    """Parse a press-conference page.

    The HTML page is a video-only landing page; the substantive
    transcript lives in a sibling PDF at /mediacenter/files/FOMCpresconf{date}.pdf.
    This function constructs the PDF URL, downloads it (or reads from
    ``cache_pdf_dir`` when the file is already on disk), extracts text
    via pypdf, splits the prepared remarks from the journalist Q&A
    (#214), and returns the four payload fields on
    ``ParsedPressConference``. On any download / extraction failure the
    text fields are empty — the caller can decide whether to keep the
    row.
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
    date_yyyymmdd = _date_yyyymmdd_from_url(source_url)

    pdf_url = _pdf_url_from_html_url(source_url)
    pdf_bytes = _load_or_fetch_pdf_bytes(
        pdf_url, cache_pdf_dir=cache_pdf_dir, date_yyyymmdd=date_yyyymmdd
    )
    text = _extract_pdf_text(pdf_bytes) if pdf_bytes else ""
    prepared_remarks, qa = split_prepared_remarks_and_qa(text) if text else ("", "")

    return ParsedPressConference(
        date=date_iso,
        title=title,
        text=text,
        url=source_url,
        prepared_remarks_text=prepared_remarks,
        qa_text=qa,
    )


def write_press_conferences_json(
    parsed: Iterable[ParsedPressConference], output_path: Path
) -> int:
    """Write parsed press conferences to output_path as a JSON list.

    Skips rows with empty text or missing date. Tags every row with
    document_type='press_conference'. Emits ``qa_text`` and
    ``prepared_remarks_text`` (#214) alongside the legacy ``text`` field
    so downstream lookups can address the high-information Q&A slice
    without re-running the splitter; both fields default to empty
    strings on rows where the boundary anchor was not located.
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
                "prepared_remarks_text": entry.prepared_remarks_text,
                "qa_text": entry.qa_text,
                "document_type": "press_conference",
                "url": entry.url,
                "scraped_at_utc": scraped_at,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return len(rows)


def build_qa_lookup(parsed: Iterable[ParsedPressConference]) -> dict[str, dict[str, str]]:
    """Build an ``event_date -> {qa_text, prepared_remarks_text, has_press_conf}`` lookup.

    The loader joins this lookup onto the supervised statement row at
    training time so the LoRA tokeniser sees the concatenated
    ``statement + Q&A`` text under route 1 of #214, and the static-cache
    path can flip the ``has_press_conf`` scalar feature without changing
    the per-bar feature size on pre-2011 rows. Rows with empty ``qa_text``
    are kept with ``has_press_conf="1"`` only when ``prepared_remarks_text``
    is populated — a press conference happened but the Q&A boundary was
    not locatable. Rows with empty ``text`` are skipped entirely.
    """

    lookup: dict[str, dict[str, str]] = {}
    for entry in parsed:
        if not entry.date or not entry.text:
            continue
        lookup[entry.date] = {
            "qa_text": entry.qa_text,
            "prepared_remarks_text": entry.prepared_remarks_text,
            "has_press_conf": "1",
        }
    return lookup


# federalreserve.gov 403s the stdlib default ``Python-urllib/x.y`` UA
# and the bare ``python-requests`` UA on some edge nodes. Identifying
# the project in the UA keeps the traffic auditable on the upstream's
# side and matches the convention the other federalreserve.gov
# scrapers in this package use.
_USER_AGENT = (
    "fed-pulse-data-ingester/1.0 "
    "(+https://github.com/yusufizzetmurat/fed-pulse)"
)


def _http_get_text(url: str, *, timeout: float) -> str:
    """Fetch ``url`` and return the response body as decoded text.

    Wraps ``requests.get`` so HTTP non-2xx surfaces as ``RuntimeError``
    (the upstream error is re-raised with the URL in the message so
    the caller logs see which page failed). The press-conference
    scraper uses ``requests`` rather than stdlib ``urllib`` for
    consistency with :func:`_load_or_fetch_pdf_bytes`, which already
    fetches PDFs via the same library.
    """

    try:
        response = requests.get(
            url, timeout=timeout, headers={"User-Agent": _USER_AGENT}
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "?"
        raise RuntimeError(f"HTTP {code} from {url}") from exc
    return response.text


def _default_historical_years() -> list[int]:
    """Years walked for the historical back-fill by default.

    Press conferences started in April 2011. The default window walks
    2011 through 2020 inclusive; 2021 onward is covered by the
    ``fomccalendars.htm`` listing already.
    """

    return list(
        range(_DEFAULT_HISTORICAL_YEARS_LOWER, _DEFAULT_HISTORICAL_YEARS_UPPER + 1)
    )


def pull_press_conferences_archive(  # noqa: PLR0913
    target_path: Path,
    *,
    force: bool = False,
    archive_url: str = ARCHIVE_LISTING_URL,
    historical_years: list[int] | None = None,
    limit: int | None = None,
    timeout: float = 30.0,
    delay_seconds: float = 0.5,
    cache_pdf_dir: Path | None = None,
) -> int:
    """Walk the federalreserve.gov FOMC calendars + historical pages and write parsed rows.

    Press-conference URLs live on two sources:

    1. ``/monetarypolicy/fomccalendars.htm`` -- the current calendar
       page, which surfaces 2021-onward press-conference links inline.
    2. ``/monetarypolicy/fomchistorical{year}.htm`` -- per-year
       historical pages, used to back-fill 2011-2020 press confs.

    The orchestrator fetches the calendar page (or whatever
    ``archive_url`` points at), then each historical year page,
    de-duplicates the discovered URLs, and parses every press
    conference. Each parse downloads the sibling PDF and runs the
    pypdf text extractor + Q&A splitter.

    Parsed rows are written to ``target_path`` via
    :func:`write_press_conferences_json` -- the same JSON shape
    ``ingest_sources._iter_scraped_records`` consumes. The Q&A
    lookup sidecar (#214) is intentionally NOT produced here; that
    follow-up writes it from the same parsed rows separately.

    Idempotent: when ``target_path`` already exists with a non-empty
    JSON list and ``force`` is False, the existing row count is
    returned without any HTTP traffic. Per-page failures are logged
    via :mod:`warnings` and the walk continues. A walk producing zero
    rows raises ``RuntimeError`` so a broken default surfaces loudly.
    """

    if target_path.exists() and not force:
        try:
            cached = json.loads(target_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cached = None
        if isinstance(cached, list) and len(cached) > 0:
            return len(cached)

    year_list = (
        historical_years
        if historical_years is not None
        else _default_historical_years()
    )
    listing_urls = [archive_url] + [
        HISTORICAL_YEAR_URL_TEMPLATE.format(year=y) for y in year_list
    ]

    entries: list[PressConferenceListingEntry] = []
    seen_urls: set[str] = set()
    for listing_url in listing_urls:
        try:
            listing_html = _http_get_text(listing_url, timeout=timeout)
        except Exception as exc:
            warnings.warn(
                f"Press conference listing fetch failed for {listing_url}: {exc}",
                stacklevel=2,
            )
            continue
        for entry in extract_press_conference_listing(listing_html):
            if entry.url in seen_urls:
                continue
            seen_urls.add(entry.url)
            entries.append(entry)

    # Sort by date so smoke-test ``--limit N`` returns the most recent
    # N press confs (most useful for end-to-end smoke tests).
    entries.sort(key=lambda e: e.date, reverse=True)
    if limit is not None:
        entries = entries[:limit]

    parsed: list[ParsedPressConference] = []
    for i, entry in enumerate(entries):
        try:
            page_html = _http_get_text(entry.url, timeout=timeout)
            parsed.append(
                parse_press_conference_page(
                    page_html,
                    source_url=entry.url,
                    cache_pdf_dir=cache_pdf_dir,
                )
            )
        except Exception as exc:
            warnings.warn(
                f"Press conference fetch failed for {entry.url}: {exc}",
                stacklevel=2,
            )
            continue
        if delay_seconds > 0 and i + 1 < len(entries):
            time.sleep(delay_seconds)

    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    try:
        written = write_press_conferences_json(parsed, tmp_path)
        if written == 0:
            raise RuntimeError(
                f"Press conferences pull produced zero rows (listings tried: "
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
            "Walk the federalreserve.gov FOMC calendar + historical pages "
            f"and write {OUTPUT_FILENAME} into the data directory. The "
            "resulting JSON is picked up unchanged by "
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
        help="Stop after fetching N press confs (smoke testing).",
    )
    parser.add_argument(
        "--archive-url",
        default=ARCHIVE_LISTING_URL,
        help="Override the FOMC calendar URL.",
    )
    parser.add_argument(
        "--historical-years",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Historical years to back-fill via "
            "/monetarypolicy/fomchistorical{year}.htm (default: 2011-2020)."
        ),
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.5,
        help="Polite delay between page fetches (default: 0.5s).",
    )
    parser.add_argument(
        "--cache-pdf-dir",
        default=None,
        help=(
            "Local directory under which press-conference PDFs are "
            "cached so a re-pull does not re-download every PDF "
            "(default: no PDF cache)."
        ),
    )
    ns = parser.parse_args()
    target = Path(ns.data_dir) / OUTPUT_FILENAME
    rows = pull_press_conferences_archive(
        target,
        force=ns.force,
        archive_url=ns.archive_url,
        historical_years=ns.historical_years,
        limit=ns.limit,
        delay_seconds=ns.delay_seconds,
        cache_pdf_dir=Path(ns.cache_pdf_dir) if ns.cache_pdf_dir else None,
    )
    print(f"Press conferences cache at {target} (rows: {rows})")
