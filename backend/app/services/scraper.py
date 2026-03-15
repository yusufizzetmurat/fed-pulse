from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

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
    root = _fetch_soup(CALENDAR_URL)
    pages = {CALENDAR_URL}
    for anchor in root.select("a[href]"):
        href = anchor.get("href", "").strip()
        if ARCHIVE_PATTERN.match(href):
            pages.add(urljoin(BASE_URL, href))
    return sorted(pages, reverse=True)


def _statement_links_from_page(page_url: str) -> list[tuple[str, str]]:
    soup = _fetch_soup(page_url)
    links: list[tuple[str, str]] = []
    for anchor in soup.select("a[href]"):
        href = anchor.get("href", "").strip()
        text = _clean_text(anchor.get_text(" ", strip=True))
        if "pressreleases/monetary" not in href:
            continue
        if "statement" not in text.lower():
            continue
        links.append((urljoin(BASE_URL, href), text))
    return links


def _minutes_links_from_page(page_url: str) -> list[tuple[str, str]]:
    soup = _fetch_soup(page_url)
    links: list[tuple[str, str]] = []
    for anchor in soup.select("a[href]"):
        href = anchor.get("href", "").strip()
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

    for document_url, fallback_title, source_page in document_links:
        soup = _fetch_soup(document_url)
        title_node = soup.select_one("h3.title") or soup.select_one("h1")
        title = _clean_text(title_node.get_text(" ", strip=True)) if title_node else fallback_title
        body = _extract_article_text(soup)
        date_value = _extract_date(soup) or _date_from_url(document_url)
        document = FomcDocument(
            date=date_value,
            meeting_type=_meeting_type_from_title(title, body),
            title=_normalized_title(title or fallback_title, document_type, document_url, date_value),
            url=document_url,
            source_page=source_page,
            document_type=document_type,
            text=body,
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


def scrape_fomc_statements(output_dir: str | Path = "/data") -> list[FomcDocument]:
    return _scrape_documents_for_type(
        document_type="Statement",
        link_getter=_statement_links_from_page,
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


if __name__ == "__main__":
    statements = scrape_fomc_statements("/data")
    minutes = scrape_fomc_minutes("/data")
    print(
        "Saved "
        f"{len(statements)} statements to /data/fomc_statements.json and /data/fomc_statements.csv, "
        f"and {len(minutes)} minutes to /data/fomc_minutes.json and /data/fomc_minutes.csv"
    )
