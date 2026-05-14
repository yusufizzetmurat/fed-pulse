"""Regression guard for the 2026-05-14 label-swap bug.

Trillion Dollar Words ships with raw integer labels:

    class 0 (515 train rows)  -> dovish
    class 1 (492 train rows)  -> hawkish
    class 2 (977 train rows)  -> neutral

Earlier this project mapped class 2 -> hawkish and class 1 -> neutral
(the inverse of TDW). The model learned the right classes but the
display labels were swapped for two out of three classes — live
/analyze returned "hawkish 0.997" on textbook-neutral input. This
test asserts the canonical mapping at the source so the bug can't
regress.

Cross-reference: 09_Risk_Register.md R-16, scripts/patch_checkpoint_id2label.py.
"""

from __future__ import annotations

import pytest

from app.data.normalize_labels import (
    DOVISH_TOKENS,
    HAWKISH_TOKENS,
    NEUTRAL_TOKENS,
    _map_label,
)


@pytest.mark.parametrize(
    "raw_label, expected",
    [
        # TDW canonical integer labels (verified by inspecting sample sentences
        # in test_data/tdw_sample_per_class.txt).
        ("0", "dovish"),
        ("LABEL_0", "dovish"),
        ("1", "hawkish"),
        ("LABEL_1", "hawkish"),
        ("2", "neutral"),
        ("LABEL_2", "neutral"),
        # Common string labels — case-insensitive
        ("hawkish", "hawkish"),
        ("HAWKISH", "hawkish"),
        ("dovish", "dovish"),
        ("DOVISH", "dovish"),
        ("neutral", "neutral"),
        ("NEUTRAL", "neutral"),
        # Soft-matching for compound labels
        ("tightening", "hawkish"),
        ("hawk", "hawkish"),
        ("easing", "dovish"),
        ("dove", "dovish"),
        ("mixed", "neutral"),
        ("balanced", "neutral"),
    ],
)
def test_map_label_matches_tdw_canonical(raw_label: str, expected: str) -> None:
    """Each raw label must map to its TDW canonical class name."""

    assert _map_label(raw_label) == expected


def test_integer_tokens_partition_correctly() -> None:
    """LABEL_0/0 -> dovish; LABEL_1/1 -> hawkish; LABEL_2/2 -> neutral.
    Critical: integer-token sets must not overlap across classes."""

    assert "0" in DOVISH_TOKENS and "label_0" in DOVISH_TOKENS
    assert "1" in HAWKISH_TOKENS and "label_1" in HAWKISH_TOKENS
    assert "2" in NEUTRAL_TOKENS and "label_2" in NEUTRAL_TOKENS

    overlaps = (
        (HAWKISH_TOKENS & DOVISH_TOKENS, "HAWKISH ∩ DOVISH"),
        (HAWKISH_TOKENS & NEUTRAL_TOKENS, "HAWKISH ∩ NEUTRAL"),
        (DOVISH_TOKENS & NEUTRAL_TOKENS, "DOVISH ∩ NEUTRAL"),
    )
    for shared, name in overlaps:
        assert not shared, f"{name} should be disjoint but shares {shared!r}"


def test_unmappable_label_returns_none() -> None:
    assert _map_label("") is None
    assert _map_label("garbage") is None
    assert _map_label("LABEL_99") is None
