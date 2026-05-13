from __future__ import annotations

from pathlib import Path
from typing import Iterable

from app.data.sources.base import BaseSourceScraper, Provenance, SourceMetadata
from app.data.sources.registry import register
from app.services.scraper_beige_book import (
    BeigeBookListingEntry,
    ParsedBeigeBook,
    extract_beige_book_listing,
    parse_beige_book_page,
    write_beige_book_json,
)


class BeigeBookScraper:
    metadata = SourceMetadata(
        name="Beige Book",
        source_type="beige_book",
        provenance=Provenance.SCRAPED,
        citation="federalreserve.gov/monetarypolicy/beigebook*.htm",
    )

    def fetch_listing(self, html: str) -> list[BeigeBookListingEntry]:
        return extract_beige_book_listing(html)

    def parse_entry(self, raw_html: str, *, source_url: str) -> ParsedBeigeBook:
        return parse_beige_book_page(raw_html, source_url=source_url)

    def write(self, parsed: Iterable[ParsedBeigeBook], output_path: Path) -> int:
        return write_beige_book_json(parsed, output_path)


register(BeigeBookScraper())
