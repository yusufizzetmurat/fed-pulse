from __future__ import annotations

import pytest

from app.data.source_type import (
    SOURCE_TYPE_BEIGE_BOOK,
    SOURCE_TYPE_CHAIR_SPEECH,
    SOURCE_TYPE_CONGRESSIONAL_TESTIMONY,
    SOURCE_TYPE_FOMC_MINUTES,
    SOURCE_TYPE_FOMC_STATEMENT,
    SOURCE_TYPE_GOVERNOR_SPEECH,
    SOURCE_TYPE_PRESS_CONFERENCE,
    SOURCE_TYPE_REGIONAL_RESEARCH,
    SOURCE_TYPE_UNKNOWN,
    SOURCE_TYPE_VALUES,
    infer_source_type,
)


@pytest.mark.parametrize(
    "document_type,title,expected",
    [
        ("minutes", "FOMC Meeting Minutes January 27, 2021", SOURCE_TYPE_FOMC_MINUTES),
        ("Minutes", "anything", SOURCE_TYPE_FOMC_MINUTES),
        ("statement", "FOMC statement", SOURCE_TYPE_FOMC_STATEMENT),
        ("Statement", "Press release", SOURCE_TYPE_FOMC_STATEMENT),
        ("speech", "Chair Powell speech on inflation", SOURCE_TYPE_CHAIR_SPEECH),
        ("speech", "Governor Waller on the labor market", SOURCE_TYPE_GOVERNOR_SPEECH),
        ("speech", "Vice Chair Brainard remarks", SOURCE_TYPE_GOVERNOR_SPEECH),
        ("testimony", "Semiannual Monetary Policy Report to Congress", SOURCE_TYPE_CONGRESSIONAL_TESTIMONY),
        ("press_conference", "FOMC press conference transcript", SOURCE_TYPE_PRESS_CONFERENCE),
        ("research", "NY Fed Liberty Street Economics", SOURCE_TYPE_REGIONAL_RESEARCH),
        ("beige_book", "Beige Book April 2024", SOURCE_TYPE_BEIGE_BOOK),
        ("unknown", "??", SOURCE_TYPE_UNKNOWN),
        ("", "", SOURCE_TYPE_UNKNOWN),
        # Precedence: document_type wins over title keywords
        ("statement", "Beige Book release", SOURCE_TYPE_FOMC_STATEMENT),
        # Precedence: testimony rule fires before speech rule when title contains "Congress"
        ("speech", "Speech to Congress about inflation", SOURCE_TYPE_CONGRESSIONAL_TESTIMONY),
        # Precedence: press_conference rule fires before speech rule
        ("press_conference", "Chair Powell press conference remarks", SOURCE_TYPE_PRESS_CONFERENCE),
        # Precedence: press_conference title keyword fires before research document_type
        ("research", "FOMC press conference research note", SOURCE_TYPE_PRESS_CONFERENCE),
    ],
)
def test_infer_source_type_returns_expected_value(
    document_type: str, title: str, expected: str
) -> None:
    assert infer_source_type(document_type=document_type, title=title) == expected


def test_all_returned_values_are_in_allowed_set() -> None:
    cases = [
        ("minutes", ""),
        ("statement", ""),
        ("speech", "Chair"),
        ("speech", "Governor"),
        ("testimony", ""),
        ("press_conference", ""),
        ("research", ""),
        ("beige_book", ""),
        ("", ""),
    ]
    for doc_type, title in cases:
        assert infer_source_type(document_type=doc_type, title=title) in SOURCE_TYPE_VALUES
