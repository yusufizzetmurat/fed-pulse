"""Loader contract tests for the Loughran-McDonald lexicon (#445)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.data.loughran_mcdonald import (
    DEFAULT_CACHE_ROOT,
    LM_CATEGORIES,
    LM_LEXICON_SHA,
    LoughranMcDonaldLexicon,
    _is_flagged,
    _normalise_token,
    _tokenise,
    load_loughran_mcdonald,
)


# ---------------------------------------------------------------------------
# Fixture CSV: 10 rows that cover every category at least once. ``year`` cells
# encode the year first added; ``0`` means the word is not flagged.
# Strong_Modal + Weak_Modal both fold into the ``modal`` bucket; the fixture
# carries one of each so the union is exercised.
# ---------------------------------------------------------------------------


_FIXTURE_HEADER = (
    "Word,Negative,Positive,Uncertainty,Litigious,Strong_Modal,Weak_Modal,Constraining\n"
)

_FIXTURE_ROWS = [
    # Word           Neg   Pos   Unc   Lit   Str   Wek   Con
    ("ABLE",         "0",  "2009","0",  "0",  "0",  "0",  "0"),
    ("LOSS",         "2009","0", "0",  "0",  "0",  "0",  "0"),
    ("UNCERTAIN",    "0",  "0",  "2009","0", "0",  "0",  "0"),
    ("LAWSUIT",      "0",  "0",  "0",  "2009","0", "0",  "0"),
    ("RESTRICTED",   "0",  "0",  "0",  "0",  "0",  "0",  "2009"),
    ("MUST",         "0",  "0",  "0",  "0",  "2009","0", "0"),
    ("MAY",          "0",  "0",  "0",  "0",  "0",  "2009","0"),
    # A word flagged on two categories (negative AND uncertainty) -- the
    # canonical lexicon has a handful of these cross-listed entries.
    ("DOUBTFUL",     "2009","0", "2009","0", "0",  "0",  "0"),
    # Unflagged filler. Should not appear in any category set.
    ("THE",          "0",  "0",  "0",  "0",  "0",  "0",  "0"),
    # Already-lowercase word + leading whitespace -- the loader must
    # normalise both.
    ("  gain ",      "0",  "2009","0",  "0",  "0",  "0",  "0"),
]


def _write_fixture(path: Path) -> Path:
    body = _FIXTURE_HEADER + "".join(
        ",".join(row) + "\n" for row in _FIXTURE_ROWS
    )
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Pure-Python contract: helpers + category list.
# ---------------------------------------------------------------------------


def test_lm_categories_pinned() -> None:
    assert LM_CATEGORIES == (
        "positive",
        "negative",
        "uncertainty",
        "litigious",
        "constraining",
        "modal",
    )


def test_normalise_token_lowercases_and_strips() -> None:
    assert _normalise_token("  HAWKISH  ") == "hawkish"
    assert _normalise_token("Loss") == "loss"


def test_tokenise_strips_punctuation_and_lowercases() -> None:
    tokens = _tokenise("Net sales rose 12%; profit fell sharply.")
    # No digits, no punctuation, every token lowercase.
    assert tokens == ["net", "sales", "rose", "profit", "fell", "sharply"]


def test_is_flagged_treats_nonzero_year_cells_as_true() -> None:
    assert _is_flagged("2009") is True
    assert _is_flagged(2009) is True
    assert _is_flagged("0") is False
    assert _is_flagged(0) is False
    assert _is_flagged("") is False
    assert _is_flagged(None) is False


# ---------------------------------------------------------------------------
# Loader: from local CSV fixture.
# ---------------------------------------------------------------------------


def test_load_loughran_mcdonald_from_fixture(tmp_path: Path) -> None:
    csv_path = _write_fixture(tmp_path / "fixture__master_dictionary.csv")
    lexicon = load_loughran_mcdonald(local_csv=csv_path)

    assert isinstance(lexicon, LoughranMcDonaldLexicon)
    assert lexicon.source_sha  # set off the filename stem when no override

    # Each category populated as expected.
    assert lexicon.words("positive") == frozenset({"able", "gain"})
    assert lexicon.words("negative") == frozenset({"loss", "doubtful"})
    assert lexicon.words("uncertainty") == frozenset({"uncertain", "doubtful"})
    assert lexicon.words("litigious") == frozenset({"lawsuit"})
    assert lexicon.words("constraining") == frozenset({"restricted"})
    # Modal is the union of Strong_Modal + Weak_Modal.
    assert lexicon.words("modal") == frozenset({"must", "may"})


def test_load_loughran_mcdonald_lowercases_word_column(tmp_path: Path) -> None:
    """Tokens flagged in CAPS on disk are normalised to lowercase keys."""

    csv_path = _write_fixture(tmp_path / "fixture__master_dictionary.csv")
    lexicon = load_loughran_mcdonald(local_csv=csv_path)
    # The fixture writes LAWSUIT in caps; the lookup is lowercase.
    assert "lawsuit" in lexicon.words("litigious")
    assert "LAWSUIT" not in lexicon.words("litigious")


def test_load_loughran_mcdonald_unknown_category_raises(tmp_path: Path) -> None:
    csv_path = _write_fixture(tmp_path / "fixture__master_dictionary.csv")
    lexicon = load_loughran_mcdonald(local_csv=csv_path)
    with pytest.raises(KeyError, match="Unknown L-M category"):
        lexicon.words("hawkish")


def test_load_loughran_mcdonald_missing_cache_raises(tmp_path: Path) -> None:
    """Loader does not reach for the network -- missing CSV => FileNotFoundError."""

    with pytest.raises(FileNotFoundError, match="cache miss"):
        load_loughran_mcdonald(cache_root=tmp_path)


def test_load_loughran_mcdonald_total_words_counts_distinct(tmp_path: Path) -> None:
    csv_path = _write_fixture(tmp_path / "fixture__master_dictionary.csv")
    lexicon = load_loughran_mcdonald(local_csv=csv_path)
    # 'doubtful' is in both negative + uncertainty; total_words counts
    # it once. Fixture distinct words: able, loss, uncertain, lawsuit,
    # restricted, must, may, doubtful, gain = 9.
    assert lexicon.total_words == 9


def test_default_cache_root_under_data_external(tmp_path: Path) -> None:
    """The default cache root lives where the wiki says it does."""

    assert DEFAULT_CACHE_ROOT.name == "loughran_mcdonald"
    assert DEFAULT_CACHE_ROOT.parent.name == "external"
    assert LM_LEXICON_SHA  # non-empty pin


def test_load_loughran_mcdonald_source_sha_override(tmp_path: Path) -> None:
    csv_path = _write_fixture(tmp_path / "fixture__master_dictionary.csv")
    lexicon = load_loughran_mcdonald(local_csv=csv_path, source_sha="custom_pin")
    assert lexicon.source_sha == "custom_pin"
