"""Allowed source_type values and a small inference helper.

source_type is finer-grained than document_type and is what makes the broader
Fed-adjacent corpus ingestible with provenance tracking. The mapping is
deterministic: same (document_type, title) input yields the same value.
"""

from __future__ import annotations

SOURCE_TYPE_FOMC_MINUTES = "fomc_minutes"
SOURCE_TYPE_FOMC_STATEMENT = "fomc_statement"
SOURCE_TYPE_CHAIR_SPEECH = "chair_speech"
SOURCE_TYPE_GOVERNOR_SPEECH = "governor_speech"
SOURCE_TYPE_CONGRESSIONAL_TESTIMONY = "congressional_testimony"
SOURCE_TYPE_PRESS_CONFERENCE = "press_conference"
SOURCE_TYPE_REGIONAL_RESEARCH = "regional_research"
SOURCE_TYPE_BEIGE_BOOK = "beige_book"
SOURCE_TYPE_UNKNOWN = "unknown"

SOURCE_TYPE_VALUES: tuple[str, ...] = (
    SOURCE_TYPE_FOMC_MINUTES,
    SOURCE_TYPE_FOMC_STATEMENT,
    SOURCE_TYPE_CHAIR_SPEECH,
    SOURCE_TYPE_GOVERNOR_SPEECH,
    SOURCE_TYPE_CONGRESSIONAL_TESTIMONY,
    SOURCE_TYPE_PRESS_CONFERENCE,
    SOURCE_TYPE_REGIONAL_RESEARCH,
    SOURCE_TYPE_BEIGE_BOOK,
    SOURCE_TYPE_UNKNOWN,
)


def infer_source_type(*, document_type: str, title: str) -> str:
    """Map a record's (document_type, title) to a canonical source_type.

    document_type is the existing ingestion field. title is used to disambiguate
    where document_type alone is too coarse (Chair vs. Governor speeches).
    Anything we cannot classify falls back to SOURCE_TYPE_UNKNOWN.
    """

    doc = (document_type or "").strip().lower()
    ttl = (title or "").lower()

    if doc in {"minutes"}:
        return SOURCE_TYPE_FOMC_MINUTES
    if doc in {"statement"}:
        return SOURCE_TYPE_FOMC_STATEMENT
    if doc in {"beige_book", "beige-book"} or "beige book" in ttl:
        return SOURCE_TYPE_BEIGE_BOOK
    if doc in {"press_conference", "press-conference"} or "press conference" in ttl:
        return SOURCE_TYPE_PRESS_CONFERENCE
    if doc in {"testimony"} or "testimony" in ttl or "congress" in ttl:
        return SOURCE_TYPE_CONGRESSIONAL_TESTIMONY
    if doc in {"research"} or "liberty street" in ttl or "fed research" in ttl:
        return SOURCE_TYPE_REGIONAL_RESEARCH
    if doc in {"speech"} or "speech" in ttl or "remarks" in ttl:
        if "chair" in ttl and "vice chair" not in ttl:
            return SOURCE_TYPE_CHAIR_SPEECH
        return SOURCE_TYPE_GOVERNOR_SPEECH

    return SOURCE_TYPE_UNKNOWN
