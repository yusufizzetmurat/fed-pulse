from __future__ import annotations

from pathlib import Path
from typing import Iterable

from app.data.sources.base import BaseSourceScraper, Provenance, SourceMetadata
from app.data.sources.registry import register
from app.services.scraper_testimonies import (
    ParsedTestimony,
    TestimonyListingEntry,
    extract_testimony_listing,
    parse_testimony_page,
    write_testimonies_json,
)


class CongressionalTestimonyScraper:
    metadata = SourceMetadata(
        name="Federal Reserve Congressional testimony",
        source_type="congressional_testimony",
        provenance=Provenance.SCRAPED,
        citation="federalreserve.gov/newsevents/testimony/",
    )

    def fetch_listing(self, html: str) -> list[TestimonyListingEntry]:
        return extract_testimony_listing(html)

    def parse_entry(self, raw_html: str, *, source_url: str) -> ParsedTestimony:
        return parse_testimony_page(raw_html, source_url=source_url)

    def write(self, parsed: Iterable[ParsedTestimony], output_path: Path) -> int:
        return write_testimonies_json(parsed, output_path)


register(CongressionalTestimonyScraper())
