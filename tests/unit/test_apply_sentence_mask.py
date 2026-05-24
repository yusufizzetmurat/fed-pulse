from __future__ import annotations

from app.main import _apply_sentence_mask


def test_empty_mask_returns_input_unchanged():
    text = "First. Second. Third."
    assert _apply_sentence_mask(text, []) == text


def test_mask_drops_indexed_sentences():
    text = "Recent indicators expanded. Inflation remains elevated. The Committee is committed."
    result = _apply_sentence_mask(text, [1])
    assert "Inflation remains elevated" not in result
    assert "Recent indicators expanded" in result
    assert "The Committee is committed" in result


def test_out_of_range_indices_are_silently_ignored():
    text = "Only one sentence here."
    assert _apply_sentence_mask(text, [5, -1, 99]) == text


def test_masking_every_sentence_falls_back_to_original_text():
    # Defensive: if every sentence is struck the pipeline still needs text,
    # so the helper preserves the original input rather than handing the
    # classifier an empty string.
    text = "Alpha. Beta. Gamma."
    assert _apply_sentence_mask(text, [0, 1, 2]) == text


def test_mask_handles_duplicates():
    text = "Alpha. Beta. Gamma."
    result = _apply_sentence_mask(text, [1, 1, 2])
    assert "Alpha" in result
    assert "Beta" not in result
    assert "Gamma" not in result
