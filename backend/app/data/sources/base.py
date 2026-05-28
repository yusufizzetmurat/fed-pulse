from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable


class Provenance(str, Enum):
    PEER_REVIEWED = "peer_reviewed"
    KAGGLE = "kaggle"
    SCRAPED = "scraped"


_VALID_SOURCE_TYPES = frozenset({
    "fomc_statement",
    "fomc_minutes",
    "fomc_meeting_transcript",
    "fomc_press_conference",
    "chair_speech",
    "governor_speech",
    "congressional_testimony",
    "beige_book",
    "regional_research",
    "ny_fed_liberty_street",
    "gss_factor_decomposition",
})


@dataclass(frozen=True)
class SourceMetadata:
    name: str
    source_type: str
    provenance: Provenance
    citation: str = ""

    def __post_init__(self) -> None:
        if self.source_type not in _VALID_SOURCE_TYPES:
            raise ValueError(
                f"source_type {self.source_type!r} not in approved vocabulary "
                f"(see ADR 0004); allowed: {sorted(_VALID_SOURCE_TYPES)}"
            )


@runtime_checkable
class BaseSourceScraper(Protocol):
    metadata: SourceMetadata

    def fetch_listing(self, html: str) -> Iterable[Any]:
        ...

    def parse_entry(self, raw_html: str, *, source_url: str) -> Any:
        ...

    def write(self, parsed: Iterable[Any], output_path: Path) -> int:
        ...
