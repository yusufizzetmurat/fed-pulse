from __future__ import annotations

import pytest

pytest.importorskip("bs4")

from app.data.sources import (  # noqa: E402
    SOURCES,
    BaseSourceScraper,
    Provenance,
    SourceMetadata,
    source_for,
    source_types,
)


def test_every_registered_scraper_has_metadata() -> None:
    assert SOURCES, "no scrapers were registered on package import"
    for source_type, scraper in SOURCES.items():
        assert scraper.metadata.source_type == source_type
        assert scraper.metadata.name
        assert isinstance(scraper.metadata.provenance, Provenance)


def test_registry_contains_expected_source_types() -> None:
    expected = {
        "governor_speech",
        "chair_speech",
        "congressional_testimony",
        "fomc_press_conference",
        "beige_book",
        "ny_fed_liberty_street",
    }
    assert expected.issubset(set(source_types()))


def test_source_for_resolves_registered_scrapers() -> None:
    for source_type in source_types():
        scraper = source_for(source_type)
        assert scraper.metadata.source_type == source_type


def test_source_for_unknown_raises() -> None:
    with pytest.raises(KeyError):
        source_for("not_a_real_source_type")


def test_source_metadata_rejects_unknown_source_type() -> None:
    with pytest.raises(ValueError, match="not in approved vocabulary"):
        SourceMetadata(
            name="bad",
            source_type="not_in_vocabulary",
            provenance=Provenance.SCRAPED,
        )


def test_registered_scrapers_satisfy_protocol() -> None:
    for scraper in SOURCES.values():
        assert isinstance(scraper, BaseSourceScraper)
