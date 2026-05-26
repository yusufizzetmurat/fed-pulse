from __future__ import annotations

from typing import Iterable

from app.data.sources.base import BaseSourceScraper

SOURCES: dict[str, BaseSourceScraper] = {}


def register(scraper: BaseSourceScraper) -> BaseSourceScraper:
    source_type = scraper.metadata.source_type
    if source_type in SOURCES:
        raise ValueError(f"duplicate scraper registered for source_type={source_type!r}")
    SOURCES[source_type] = scraper
    return scraper


def source_for(source_type: str) -> BaseSourceScraper:
    if source_type not in SOURCES:
        raise KeyError(f"no scraper registered for source_type={source_type!r}")
    return SOURCES[source_type]


def source_types() -> Iterable[str]:
    return tuple(SOURCES.keys())
