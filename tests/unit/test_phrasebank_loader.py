"""Loader contract tests for the PhraseBank auxiliary-task source (#33)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.data.phrasebank import (
    ID2LABEL,
    LABEL2ID,
    N_CLASSES,
    PHRASEBANK_LABELS,
    PhraseBankRow,
    _coerce_label_idx,
    _iter_local_rows,
    class_counts,
    load_phrasebank_rows,
)


# ---------------------------------------------------------------------------
# Pure-Python contract tests (no torch / HF datasets required).
# ---------------------------------------------------------------------------


def test_label_ordering_is_pinned() -> None:
    """Canonical label order must be negative / neutral / positive."""

    assert PHRASEBANK_LABELS == ("negative", "neutral", "positive")
    assert N_CLASSES == 3
    assert LABEL2ID == {"negative": 0, "neutral": 1, "positive": 2}
    assert ID2LABEL == {0: "negative", 1: "neutral", 2: "positive"}


def test_coerce_label_idx_accepts_canonical_strings() -> None:
    assert _coerce_label_idx("negative") == 0
    assert _coerce_label_idx("Neutral") == 1
    assert _coerce_label_idx("POSITIVE") == 2


def test_coerce_label_idx_accepts_canonical_ints() -> None:
    assert _coerce_label_idx(0) == 0
    assert _coerce_label_idx(1) == 1
    assert _coerce_label_idx(2) == 2


def test_coerce_label_idx_rejects_out_of_range() -> None:
    assert _coerce_label_idx(-1) is None
    assert _coerce_label_idx(3) is None
    assert _coerce_label_idx("hawkish") is None
    assert _coerce_label_idx(None) is None


def test_iter_local_rows_drops_empty_and_invalid() -> None:
    rows = [
        {"sentence": "Net sales rose 12% on the year.", "label": "positive"},
        {"sentence": "", "label": "negative"},  # empty sentence -> dropped
        {"sentence": "Quarterly profit fell sharply.", "label": "bogus"},  # bad label -> dropped
        {"sentence": "Revenue was flat.", "label": 1},
    ]
    parsed = _iter_local_rows(rows)
    assert len(parsed) == 2
    assert parsed[0].label == "positive"
    assert parsed[1].label == "neutral"
    # Row ids are monotonic.
    assert parsed[0].row_id == "pb_00000"
    assert parsed[1].row_id == "pb_00003"


def test_class_counts_in_canonical_order() -> None:
    rows = [
        PhraseBankRow(row_id="r0", sentence="x", label_idx=0),
        PhraseBankRow(row_id="r1", sentence="x", label_idx=2),
        PhraseBankRow(row_id="r2", sentence="x", label_idx=2),
        PhraseBankRow(row_id="r3", sentence="x", label_idx=1),
    ]
    assert class_counts(rows) == [1, 1, 2]


def test_load_phrasebank_rows_from_local_jsonl(tmp_path: Path) -> None:
    """Local JSONL fixture round-trips through the loader."""

    fixture = tmp_path / "phrasebank_fixture.jsonl"
    rows = [
        {"sentence": "The company reported record earnings.", "label": "positive"},
        {"sentence": "Operating costs declined modestly.", "label": "neutral"},
        {"sentence": "Profit fell on weaker demand.", "label": "negative"},
    ]
    fixture.write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    loaded = load_phrasebank_rows(local_jsonl=fixture)
    assert [r.label for r in loaded] == ["positive", "neutral", "negative"]
    assert all(r.sentence for r in loaded)


def test_load_phrasebank_rows_rejects_unknown_subset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not in"):
        load_phrasebank_rows(subset="sentences_bogus", cache_root=tmp_path)


def test_load_phrasebank_rows_missing_fixture_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_phrasebank_rows(local_jsonl=tmp_path / "absent.jsonl")


def test_phrasebank_cache_round_trips(tmp_path: Path) -> None:
    """Cache write -> read returns the same rows without hitting HF."""

    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from app.data.phrasebank import _cache_path, _read_cache, _write_cache

    rows = [
        PhraseBankRow(row_id="pb_00000", sentence="Net sales rose.", label_idx=2),
        PhraseBankRow(row_id="pb_00001", sentence="Profit fell.", label_idx=0),
    ]
    cache_path = _cache_path(tmp_path, "sentences_allagree", None)
    _write_cache(cache_path, rows)
    assert cache_path.exists()
    cached = _read_cache(cache_path)
    assert cached is not None
    assert [(r.sentence, r.label_idx) for r in cached] == [
        ("Net sales rose.", 2),
        ("Profit fell.", 0),
    ]
