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
    write_chair_speeches_json,
    write_governor_speeches_json,
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
        return write_governor_speeches_json(parsed, output_path)


class ChairSpeechesScraper:
    metadata = SourceMetadata(
        name="Fed chair speeches",
        source_type="chair_speech",
        provenance=Provenance.SCRAPED,
        citation="federalreserve.gov/newsevents/speech/",
    )

    def fetch_listing(self, html: str) -> list[SpeechListingEntry]:
        return extract_speech_listing(html)

    def parse_entry(self, raw_html: str, *, source_url: str) -> ParsedSpeech:
        return parse_speech_page(raw_html, source_url=source_url)

    def write(self, parsed: Iterable[ParsedSpeech], output_path: Path) -> int:
        return write_chair_speeches_json(parsed, output_path)


register(GovernorSpeechesScraper())
register(ChairSpeechesScraper())
