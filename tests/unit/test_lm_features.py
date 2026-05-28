"""compute_lm_features contract tests (#445)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.data.loughran_mcdonald import (
    LM_CATEGORIES,
    compute_lm_feature_vector,
    compute_lm_features,
    load_loughran_mcdonald,
)


# Reuse the fixture row writer from the loader test module.
from tests.unit.test_loughran_mcdonald_loader import _write_fixture


@pytest.fixture
def lexicon(tmp_path: Path):
    csv_path = _write_fixture(tmp_path / "fixture__master_dictionary.csv")
    return load_loughran_mcdonald(local_csv=csv_path)


def test_empty_document_returns_zeros(lexicon) -> None:
    features = compute_lm_features("", lexicon)
    assert features == {f"lm_{cat}_pct": 0.0 for cat in LM_CATEGORIES}


def test_all_positive_document(lexicon) -> None:
    # Three tokens, all in the positive set ('able', 'gain').
    features = compute_lm_features("Able gain able", lexicon)
    assert features["lm_positive_pct"] == pytest.approx(100.0)
    assert features["lm_negative_pct"] == pytest.approx(0.0)
    assert features["lm_uncertainty_pct"] == pytest.approx(0.0)


def test_all_uncertainty_document(lexicon) -> None:
    # 'uncertain' is uncertainty-only; 'doubtful' is cross-listed in
    # negative + uncertainty -- the counts increment both buckets.
    features = compute_lm_features("Uncertain doubtful", lexicon)
    assert features["lm_uncertainty_pct"] == pytest.approx(100.0)
    # doubtful is also negative, so 1/2 == 50%.
    assert features["lm_negative_pct"] == pytest.approx(50.0)
    assert features["lm_positive_pct"] == pytest.approx(0.0)


def test_mixed_document_percentages(lexicon) -> None:
    # 5 tokens: 'gain' (positive), 'loss' (negative), 'must' (modal),
    # 'restricted' (constraining), 'the' (not flagged).
    text = "Gain loss must restricted the"
    features = compute_lm_features(text, lexicon)
    assert features["lm_positive_pct"] == pytest.approx(20.0)
    assert features["lm_negative_pct"] == pytest.approx(20.0)
    assert features["lm_modal_pct"] == pytest.approx(20.0)
    assert features["lm_constraining_pct"] == pytest.approx(20.0)
    assert features["lm_uncertainty_pct"] == pytest.approx(0.0)
    assert features["lm_litigious_pct"] == pytest.approx(0.0)


def test_modal_union_strong_and_weak(lexicon) -> None:
    # 'must' is Strong_Modal, 'may' is Weak_Modal -- both count toward
    # lm_modal_pct because the loader unions the two columns.
    features = compute_lm_features("must may", lexicon)
    assert features["lm_modal_pct"] == pytest.approx(100.0)


def test_compute_lm_feature_vector_order(lexicon) -> None:
    """The list-shaped helper emits values in LM_CATEGORIES order."""

    text = "Gain loss uncertain lawsuit restricted must"
    vector = compute_lm_feature_vector(text, lexicon)
    assert len(vector) == len(LM_CATEGORIES)
    # Six tokens, one per category; each percentage should be ~16.67.
    for value in vector:
        assert value == pytest.approx(100.0 / 6.0)


def test_punctuation_and_digits_are_stripped(lexicon) -> None:
    # Digits, numerals, and punctuation must not inflate the denominator.
    text = "Gain, gain! 2025 25% the-loss."
    features = compute_lm_features(text, lexicon)
    # Tokens after tokenisation: gain, gain, the, loss => 4 tokens.
    # positive = 2 (gain x2) => 50%; negative = 1 (loss) => 25%.
    assert features["lm_positive_pct"] == pytest.approx(50.0)
    assert features["lm_negative_pct"] == pytest.approx(25.0)


def test_lowercase_input_matches_uppercase_lexicon(lexicon) -> None:
    # The lexicon stores words lowercase; compute_lm_features lowercases
    # the document, so mixed-case input still matches.
    features = compute_lm_features("LOSS Loss loss", lexicon)
    assert features["lm_negative_pct"] == pytest.approx(100.0)
