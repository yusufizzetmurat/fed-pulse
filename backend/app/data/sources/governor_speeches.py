from __future__ import annotations

from pathlib import Path
from typing import Iterable

from app.data.sources.base import BaseSourceScraper, Provenance, SourceMetadata
from app.data.sources.registry import register
from app.services.scraper_speeches import (
    ParsedSpeech,
    SpeechListingEntry,
    extract_speech_listing,
    parse_speech_page,
    write_speeches_json,
)


class GovernorSpeechesScraper:
    metadata = SourceMetadata(
        name="Fed governor speeches",
        source_type="governor_speech",
        provenance=Provenance.SCRAPED,
        citation="federalreserve.gov/newsevents/speech/",
    )

    def fetch_listing(self, html: str) -> list[SpeechListingEntry]:
        return extract_speech_listing(html)

    def parse_entry(self, raw_html: str, *, source_url: str) -> ParsedSpeech:
        return parse_speech_page(raw_html, source_url=source_url)

    def write(self, parsed: Iterable[ParsedSpeech], output_path: Path) -> int:
        return write_speeches_json(parsed, output_path)


_GOVERNOR_SPEECHES = GovernorSpeechesScraper()
_CHAIR_SPEECHES_METADATA = SourceMetadata(
    name="Fed chair speeches",
    source_type="chair_speech",
    provenance=Provenance.SCRAPED,
    citation="federalreserve.gov/newsevents/speech/",
)


class ChairSpeechesScraper:
    metadata = _CHAIR_SPEECHES_METADATA

    def fetch_listing(self, html: str) -> list[SpeechListingEntry]:
        return extract_speech_listing(html)

    def parse_entry(self, raw_html: str, *, source_url: str) -> ParsedSpeech:
        return parse_speech_page(raw_html, source_url=source_url)

    def write(self, parsed: Iterable[ParsedSpeech], output_path: Path) -> int:
        return write_speeches_json(parsed, output_path)


_CHAIR_SPEECHES = ChairSpeechesScraper()

register(_GOVERNOR_SPEECHES)
register(_CHAIR_SPEECHES)
