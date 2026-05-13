from __future__ import annotations

from pathlib import Path
from typing import Iterable

from app.data.sources.base import BaseSourceScraper, Provenance, SourceMetadata
from app.data.sources.registry import register
from app.services.scraper_regional_research import (
    ParsedRegionalResearch,
    RegionalResearchListingEntry,
    extract_lse_listing,
    parse_lse_post,
    write_regional_research_json,
)


class NyFedLibertyStreetScraper:
    metadata = SourceMetadata(
        name="NY Fed Liberty Street Economics",
        source_type="ny_fed_liberty_street",
        provenance=Provenance.SCRAPED,
        citation="libertystreeteconomics.newyorkfed.org",
    )

    def fetch_listing(self, html: str) -> list[RegionalResearchListingEntry]:
        return extract_lse_listing(html)

    def parse_entry(self, raw_html: str, *, source_url: str) -> ParsedRegionalResearch:
        return parse_lse_post(raw_html, source_url=source_url)

    def write(self, parsed: Iterable[ParsedRegionalResearch], output_path: Path) -> int:
        return write_regional_research_json(parsed, output_path)


register(NyFedLibertyStreetScraper())
