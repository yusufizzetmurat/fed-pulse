from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from app.services.text_hygiene import clean_fomc_text

BASE_URL = "https://www.federalreserve.gov"
CALENDAR_URL = f"{BASE_URL}/monetarypolicy/fomccalendars.htm"
ARCHIVE_PATTERN = re.compile(r"^/monetarypolicy/fomchistorical\d{4}\.htm$")
DATE_PATTERN = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
    flags=re.IGNORECASE,
)
MINUTES_URL_DATE_PATTERN = re.compile(r"fomcminutes(\d{8})", flags=re.IGNORECASE)


@dataclass
class FomcDocument:
    date: str
    meeting_type: str
    title: str
    url: str
    source_page: str
    document_type: str
    text: str
    scraped_at_utc: str


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _href_str(anchor: Any) -> str:
    """Coerce a bs4 anchor href to a str. bs4 multi-value attrs come back as
    ``AttributeValueList``; the federal-reserve markup never uses that shape,
    so the join keeps the call sites monomorphic without changing behaviour.
    """

    href = anchor.get("href", "") or ""
    if isinstance(href, list):
        href = " ".join(str(part) for part in href)
    return str(href).strip()


def _fetch_soup(url: str, timeout: int = 20) -> BeautifulSoup:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def _extract_date(soup: BeautifulSoup) -> str:
    selectors = [
        "p.article__time",
        "p.article__meta",
        "time",
        "h5",
        "h3",
    ]
    for selector in selectors:
        for node in soup.select(selector):
            text = _clean_text(node.get_text(" ", strip=True))
            matched = DATE_PATTERN.search(text)
            if not matched:
                continue
            parsed = datetime.strptime(matched.group(0), "%B %d, %Y")
            return parsed.date().isoformat()
    return ""


def _extract_article_text(soup: BeautifulSoup) -> str:
    candidate_selectors = [
        "div.col-xs-12.col-sm-8.col-md-8 p",
        "div.col-xs-12.col-sm-8.col-md-8 li",
        "article p",
        "main p",
    ]
    for selector in candidate_selectors:
        nodes = soup.select(selector)
        if not nodes:
            continue
        text_chunks = [_clean_text(node.get_text(" ", strip=True)) for node in nodes]
        text = "\n".join(chunk for chunk in text_chunks if chunk)
        if text:
            return text
    return _clean_text(soup.get_text(" ", strip=True))


def _meeting_type_from_title(title: str, body: str) -> str:
    merged = f"{title} {body}".lower()
    if "unscheduled" in merged:
        return "Unscheduled"
    if "scheduled" in merged:
        return "Scheduled"
    return "Regular"


def _date_from_url(url: str) -> str:
    matched = MINUTES_URL_DATE_PATTERN.search(url)
    if not matched:
        return ""
    return datetime.strptime(matched.group(1), "%Y%m%d").date().isoformat()


def _normalized_title(title: str, document_type: str, document_url: str, date_value: str) -> str:
    cleaned = _clean_text(title)
    if cleaned.lower() in {"", "html", "pdf", "board of governors of the federal reserve system"}:
        if date_value:
            return f"FOMC {document_type} {date_value}"
        inferred_date = _date_from_url(document_url)
        if inferred_date:
            return f"FOMC {document_type} {inferred_date}"
        return f"FOMC {document_type}"
    return cleaned


def _calendar_pages() -> list[str]:
    # The current fomccalendars.htm only links to one ceremonial January
    # "longer-run goals reaffirmation" press release per recent year, so it
    # under-collects modern statements at ~1/year. The
    # /monetarypolicy/fomchistorical{year}.htm pages for 1994-2020 carry
    # direct press-release links for ~2011-2020 and prep-doc links for
    # earlier years. For 2021-current the Fed publishes statements at
    # /newsevents/pressreleases/monetary{YYYYMMDD}a.htm but never indexes
    # them on a discoverable archive page; the modern range is covered by
    # the direct-probe loop in ``_modern_statement_urls()`` instead.
    pages: set[str] = {CALENDAR_URL}
    for year in range(1994, 2021):
        pages.add(f"{BASE_URL}/monetarypolicy/fomchistorical{year}.htm")
    try:
        root = _fetch_soup(CALENDAR_URL)
        for anchor in root.select("a[href]"):
            href = _href_str(anchor)
            if ARCHIVE_PATTERN.match(href):
                pages.add(urljoin(BASE_URL, href))
    except Exception:  # pragma: no cover - the explicit enumeration above is
        # already sufficient; the dynamic pass is opportunistic.
        pass
    return sorted(pages, reverse=True)


# Known FOMC meeting dates 2021-2022, the gap between the historical-archive
# era and the dates carried by ``app.services.fomc_calendar``. Every entry is
# a regularly-scheduled FOMC meeting; statements are released on the listed
# date at 14:00 ET. Sourced from the Fed's published meeting calendar.
_FOMC_MEETING_DATES_GAP: tuple[str, ...] = (
    # 2021
    "20210127",
    "20210317",
    "20210428",
    "20210616",
    "20210728",
    "20210922",
    "20211103",
    "20211215",
    # 2022
    "20220126",
    "20220316",
    "20220504",
    "20220615",
    "20220727",
    "20220921",
    "20221102",
    "20221214",
)


def _modern_statement_urls() -> list[tuple[str, str]]:
    """Probe FOMC meeting dates directly for the modern statement-URL shape.

    The Fed's year-press archive does not surface statement URLs (only
    minutes / longer-run-goals / org announcements), so the catch-all
    page-walk in ``_calendar_pages`` cannot find them. Reach the modern
    range by enumerating known meeting dates and probing
    ``/newsevents/pressreleases/monetary{YYYYMMDD}a.htm`` for each.

    Returns the discoverable (url, label) pairs; 404s are skipped silently
    so the loop is forward-compatible with future meetings.
    """

    candidates: list[str] = list(_FOMC_MEETING_DATES_GAP)
    try:
        from app.services.fomc_calendar import list_all_meetings

        for meeting in list_all_meetings():
            # The Fed publishes statements on the meeting's CONCLUDING day
            # (day-2 for the standard two-day FOMC schedule). The calendar
            # dataclass stores ``meeting_date`` as day-1 and the release
            # date in ``statement_release_date``; the URL slug tracks the
            # release date, not the meeting start. Fall back to
            # ``meeting_date`` for any entry where the release date is
            # nullable / absent.
            release_date = meeting.statement_release_date or meeting.meeting_date
            candidates.append(release_date.strftime("%Y%m%d"))
    except Exception:  # pragma: no cover - calendar import is best-effort.
        pass

    seen: set[str] = set()
    discoverable: list[tuple[str, str]] = []
    for date_slug in candidates:
        if date_slug in seen:
            continue
        seen.add(date_slug)
        url = f"{BASE_URL}/newsevents/pressreleases/monetary{date_slug}a.htm"
        try:
            resp = requests.head(url, timeout=10, allow_redirects=True)
            if resp.status_code != 200:
                continue
        except Exception:  # pragma: no cover - network hiccup; skip.
            continue
        discoverable.append((url, "Statement"))
    return discoverable


def _statement_links_from_page(page_url: str) -> list[tuple[str, str]]:
    soup = _fetch_soup(page_url)
    links: list[tuple[str, str]] = []
    # The historical archive pages anchor statement links with text containing
    # "statement"; the year-press archive pages anchor them by date alone and
    # the "statement" filter would drop them. Recognise the modern URL pattern
    # ``monetary{YYYYMMDD}{suffix}.htm`` so both shapes land.
    statement_url_re = re.compile(
        r"/pressreleases/monetary(\d{8})([a-z])?\.htm$", flags=re.IGNORECASE
    )
    for anchor in soup.select("a[href]"):
        href = _href_str(anchor)
        text = _clean_text(anchor.get_text(" ", strip=True))
        if "pressreleases/monetary" not in href:
            continue
        url_match = statement_url_re.search(href)
        text_lower = text.lower()
        # Accept when the anchor text says "statement" (historical archives)
        # OR the URL ends with the canonical monetary{date}.htm shape AND the
        # anchor text does NOT explicitly identify a non-statement release
        # (minutes, longer-run-goals reaffirmation, meeting-schedule
        # announcements, discount-rate minutes). The negative filter keeps
        # the year-press archive's minutes / org announcements out without
        # losing the actual policy statements.
        if "statement" in text_lower:
            links.append((urljoin(BASE_URL, href), text))
            continue
        if url_match is None:
            continue
        if any(
            kw in text_lower
            for kw in (
                "minutes",
                "tentative meeting schedule",
                "longer-run goals",
                "longer run goals",
                "discount rate",
            )
        ):
            continue
        links.append((urljoin(BASE_URL, href), text))
    return links


def _minutes_links_from_page(page_url: str) -> list[tuple[str, str]]:
    soup = _fetch_soup(page_url)
    links: list[tuple[str, str]] = []
    for anchor in soup.select("a[href]"):
        href = _href_str(anchor)
        text = _clean_text(anchor.get_text(" ", strip=True))
        href_lower = href.lower()
        text_lower = text.lower()
        if "fomcminutes" not in href_lower:
            continue
        if not href_lower.endswith(".htm"):
            continue
        if "/monetarypolicy/" not in href_lower:
            continue
        links.append((urljoin(BASE_URL, href), text))
    return links


def _unique_links(links: Iterable[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    seen: set[str] = set()
    unique: list[tuple[str, str, str]] = []
    for link_url, label, source in links:
        if link_url in seen:
            continue
        seen.add(link_url)
        unique.append((link_url, label, source))
    return unique


def _scrape_documents_for_type(
    document_type: str,
    link_getter: Callable[[str], list[tuple[str, str]]],
    output_prefix: str,
    output_dir: str | Path = "/data",
) -> list[FomcDocument]:
    pages = _calendar_pages()
    collected_links: list[tuple[str, str, str]] = []
    for page in pages:
        for link_url, label in link_getter(page):
            collected_links.append((link_url, label, page))
    document_links = _unique_links(collected_links)

    records: list[FomcDocument] = []
    scraped_at = datetime.now(timezone.utc).isoformat()

    hygiene_kind: Literal["statement", "minutes", "press_conference"] = (
        "minutes" if document_type.lower().startswith("minutes") else "statement"
    )

    for document_url, fallback_title, source_page in document_links:
        soup = _fetch_soup(document_url)
        title_node = soup.select_one("h3.title") or soup.select_one("h1")
        title = _clean_text(title_node.get_text(" ", strip=True)) if title_node else fallback_title
        body = _extract_article_text(soup)
        cleaned_body = clean_fomc_text(body, kind=hygiene_kind)
        date_value = _extract_date(soup) or _date_from_url(document_url)
        document = FomcDocument(
            date=date_value,
            meeting_type=_meeting_type_from_title(title, body),
            title=_normalized_title(
                title or fallback_title, document_type, document_url, date_value
            ),
            url=document_url,
            source_page=source_page,
            document_type=document_type,
            text=cleaned_body,
            scraped_at_utc=scraped_at,
        )
        records.append(document)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / f"{output_prefix}.json"
    csv_path = output_path / f"{output_prefix}.csv"

    payload = [asdict(record) for record in records]
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    field_names = list(FomcDocument.__annotations__.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(payload)

    return records


def _statement_links_with_direct_probe(page_url: str) -> list[tuple[str, str]]:
    """Statement link discovery: page-walk first, then direct-probe."""

    links = _statement_links_from_page(page_url)
    # Bolt the modern-window direct-probe onto the FIRST page traversal so it
    # runs exactly once per scrape. Subsequent pages return their own
    # link-discovery results without re-probing. The _unique_links dedup
    # downstream collapses URL collisions between page-walked and probed
    # entries.
    if page_url == CALENDAR_URL:
        links.extend(_modern_statement_urls())
    return links


def scrape_fomc_statements(output_dir: str | Path = "/data") -> list[FomcDocument]:
    return _scrape_documents_for_type(
        document_type="Statement",
        link_getter=_statement_links_with_direct_probe,
        output_prefix="fomc_statements",
        output_dir=output_dir,
    )


def scrape_fomc_minutes(output_dir: str | Path = "/data") -> list[FomcDocument]:
    return _scrape_documents_for_type(
        document_type="Minutes",
        link_getter=_minutes_links_from_page,
        output_prefix="fomc_minutes",
        output_dir=output_dir,
    )


def _cli_main(argv: list[str] | None = None) -> int:
    """Command-line entry: walk the federalreserve.gov calendar archives
    and write ``fomc_statements.{json,csv}`` and ``fomc_minutes.{json,csv}``
    under ``--output-dir`` (defaults to ``/data`` to match the container
    volume mount). Used by operators to refresh the on-disk caches the
    ``/fomc/calendar`` availability badges and the ``/documents/{type}/
    {date}`` viewer route both read against.

    Returns 0 on success, non-zero on failure so docker-compose run exits
    cleanly.
    """

    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m app.services.scraper",
        description=(
            "Refresh the FOMC statement and minutes caches by walking the "
            "Federal Reserve's calendar / historical archive pages."
        ),
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="/data",
        help="Directory to write the JSON + CSV caches into (default: /data).",
    )
    args = parser.parse_args(argv)

    try:
        statements = scrape_fomc_statements(args.output_dir)
        minutes = scrape_fomc_minutes(args.output_dir)
    except Exception as exc:  # pragma: no cover - surfaced at the CLI.
        print(f"scrape failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Saved {len(statements)} statements to {args.output_dir}/fomc_statements.json"
        f" and {args.output_dir}/fomc_statements.csv, and "
        f"{len(minutes)} minutes to {args.output_dir}/fomc_minutes.json and "
        f"{args.output_dir}/fomc_minutes.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
