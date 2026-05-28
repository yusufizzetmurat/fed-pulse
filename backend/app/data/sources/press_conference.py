from __future__ import annotations

from pathlib import Path
from typing import Iterable

from app.config import DATA_DIR
from app.data.sources.base import Provenance, SourceMetadata
from app.data.sources.registry import register
from app.services.scraper_press_conferences import (
    ParsedPressConference,
    PressConferenceListingEntry,
    build_qa_lookup,
    extract_press_conference_listing,
    parse_press_conference_page,
    write_press_conferences_json,
)


# Local PDF cache for press conferences (#214). Subsequent re-encodes
# read from disk instead of re-pulling ~250 KB per event from the Fed
# CDN. Lives under ``data/raw/fomc_press_conferences/`` to match the
# audit doc row for the joint corpus.
DEFAULT_PDF_CACHE_DIR: Path = DATA_DIR / "raw" / "fomc_press_conferences"


class FomcPressConferenceScraper:
    metadata = SourceMetadata(
        name="FOMC press conference transcripts",
        source_type="fomc_press_conference",
        provenance=Provenance.SCRAPED,
        citation="federalreserve.gov/monetarypolicy/fomcpresconf*.htm",
    )

    def __init__(self, *, cache_pdf_dir: Path | None = None) -> None:
        self.cache_pdf_dir = (
            Path(cache_pdf_dir) if cache_pdf_dir is not None else DEFAULT_PDF_CACHE_DIR
        )

    def fetch_listing(self, html: str) -> list[PressConferenceListingEntry]:
        return extract_press_conference_listing(html)

    def parse_entry(self, raw_html: str, *, source_url: str) -> ParsedPressConference:
        return parse_press_conference_page(
            raw_html, source_url=source_url, cache_pdf_dir=self.cache_pdf_dir
        )

    def write(self, parsed: Iterable[ParsedPressConference], output_path: Path) -> int:
        return write_press_conferences_json(parsed, output_path)

    def build_qa_lookup(
        self, parsed: Iterable[ParsedPressConference]
    ) -> dict[str, dict[str, str]]:
        return build_qa_lookup(parsed)


register(FomcPressConferenceScraper())
