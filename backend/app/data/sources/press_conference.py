from __future__ import annotations

from pathlib import Path
from typing import Iterable

from app.data.sources.base import BaseSourceScraper, Provenance, SourceMetadata
from app.data.sources.registry import register
from app.services.scraper_press_conferences import (
    ParsedPressConference,
    PressConferenceListingEntry,
    extract_press_conference_listing,
    parse_press_conference_page,
    write_press_conferences_json,
)


class FomcPressConferenceScraper:
    metadata = SourceMetadata(
        name="FOMC press conference transcripts",
        source_type="fomc_press_conference",
        provenance=Provenance.SCRAPED,
        citation="federalreserve.gov/monetarypolicy/fomcpresconf*.htm",
    )

    def fetch_listing(self, html: str) -> list[PressConferenceListingEntry]:
        return extract_press_conference_listing(html)

    def parse_entry(self, raw_html: str, *, source_url: str) -> ParsedPressConference:
        return parse_press_conference_page(raw_html, source_url=source_url)

    def write(self, parsed: Iterable[ParsedPressConference], output_path: Path) -> int:
        return write_press_conferences_json(parsed, output_path)


register(FomcPressConferenceScraper())
