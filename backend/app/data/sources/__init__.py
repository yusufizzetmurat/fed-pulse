from app.data.sources.base import (
    BaseSourceScraper,
    Provenance,
    SourceMetadata,
)
from app.data.sources.registry import SOURCES, register, source_for, source_types

# Importing the adapter modules triggers their register() side-effects.
from app.data.sources import (  # noqa: F401  (side-effect imports)
    beige_book,
    governor_speeches,
    press_conference,
    regional_research,
    testimony,
)

__all__ = [
    "BaseSourceScraper",
    "Provenance",
    "SOURCES",
    "SourceMetadata",
    "register",
    "source_for",
    "source_types",
]
