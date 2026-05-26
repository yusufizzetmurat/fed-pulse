"""Unit tests for ``app.retrieval.train.build_training_pairs`` (#294).

The training pair builder governs what (anchor, positive) examples
MNRL sees during the contrastive fine-tune. The walk-forward boundary
is enforced at this layer so the encoder's weights never observe
future text relative to the supplied train_end.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from app.retrieval import train as ret_train


def _make_row(
    *,
    event_date: str,
    event_kind: str,
    text: str,
    horizon: int = 1,
) -> dict:
    return {
        "event_date": event_date,
        "event_kind": event_kind,
        "text": text,
        "horizon": horizon,
        "text_hash": hashlib.sha256(f"{event_date}|{event_kind}|{text}".encode("utf-8")).hexdigest(),
    }


def _events_frame() -> pd.DataFrame:
    """Four meetings: two with statement+minutes, one statement-only, one minutes-only."""

    rows = [
        # 2010 — full meeting: statement + minutes + irrelevant macro_release on same date.
        _make_row(event_date="2010-03-16", event_kind="statement", text="2010 statement"),
        _make_row(event_date="2010-03-16", event_kind="minutes", text="2010 minutes"),
        _make_row(event_date="2010-03-16", event_kind="macro_release", text="2010 CPI"),
        # 2015 — statement + press_conference (also a valid POSITIVE_KIND).
        _make_row(event_date="2015-06-17", event_kind="statement", text="2015 statement"),
        _make_row(
            event_date="2015-06-17",
            event_kind="press_conference",
            text="2015 press conference",
        ),
        # 2018 — statement only; no sibling -> dropped post-review.
        _make_row(event_date="2018-09-26", event_kind="statement", text="2018 statement"),
        # 2022 — past train_end cutoff; dropped by walk-forward filter.
        _make_row(event_date="2022-09-21", event_kind="statement", text="2022 statement"),
        _make_row(event_date="2022-09-21", event_kind="minutes", text="2022 minutes"),
    ]
    return pd.DataFrame(rows)


def test_pairs_only_use_statement_anchors_with_minutes_or_press_conference_siblings() -> None:
    events = _events_frame()
    pairs = ret_train.build_training_pairs(events)
    sibling_kinds = {p.positive_kind for p in pairs}
    assert sibling_kinds <= set(ret_train.POSITIVE_KINDS)
    assert "macro_release" not in sibling_kinds


def test_pairs_drop_meetings_with_no_sibling_no_self_pair_fallback() -> None:
    """The 2018 statement has no sibling and must contribute zero pairs.

    Pre-review the builder shipped a degenerate ``(statement, statement)``
    self-pair so MNRL still saw the anchor; that signal teaches the
    encoder to maximise self-similarity and amplifies the self-match
    bias at retrieval time. Builder now silently drops the meeting.
    """

    events = _events_frame()
    pairs = ret_train.build_training_pairs(events)
    anchor_dates = {p.anchor_date for p in pairs}
    assert "2018-09-26" not in anchor_dates
    assert not any(p.positive_kind == "self" for p in pairs)
    assert not any(p.anchor == p.positive for p in pairs)


def test_pairs_respect_train_end_filter() -> None:
    events = _events_frame()
    pairs_unfiltered = ret_train.build_training_pairs(events)
    pairs_filtered = ret_train.build_training_pairs(events, train_end="2020-01-01")

    # The 2022 meeting is dropped under the filter, so the filtered
    # set is strictly smaller and contains no 2022 anchors.
    assert len(pairs_filtered) < len(pairs_unfiltered)
    assert all(p.anchor_date < "2020-01-01" for p in pairs_filtered)
    assert any(p.anchor_date == "2022-09-21" for p in pairs_unfiltered)


def test_pairs_filter_is_strictly_less_than_train_end() -> None:
    """``train_end`` cuts at strict ``<`` so the boundary day itself is excluded."""

    events = _events_frame()
    pairs = ret_train.build_training_pairs(events, train_end="2010-03-16")
    # 2010-03-16 must be excluded under strict <.
    assert all(p.anchor_date < "2010-03-16" for p in pairs)


def test_validate_train_end_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="ISO date"):
        ret_train._validate_train_end("not-a-date")


def test_resolve_train_end_from_fold_returns_manifest_value(tmp_path: Path) -> None:
    pkg = tmp_path / "tp_test"
    pkg.mkdir()
    events_parquet = pkg / "events.parquet"
    pd.DataFrame(_events_frame()).to_parquet(events_parquet, index=False)
    (pkg / ret_train.FOLD_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "folds": [
                    {"fold_id": "wf_fold_1", "train_end": "2016-09-21"},
                    {"fold_id": "wf_fold_3", "train_end": "2019-08-01"},
                ]
            }
        ),
        encoding="utf-8",
    )

    resolved = ret_train.resolve_train_end_from_fold(
        events_parquet=events_parquet, fold_id="wf_fold_3"
    )
    assert resolved == "2019-08-01"


def test_resolve_train_end_from_fold_raises_on_unknown_id(tmp_path: Path) -> None:
    pkg = tmp_path / "tp_test"
    pkg.mkdir()
    events_parquet = pkg / "events.parquet"
    pd.DataFrame(_events_frame()).to_parquet(events_parquet, index=False)
    (pkg / ret_train.FOLD_MANIFEST_FILENAME).write_text(
        json.dumps({"folds": [{"fold_id": "wf_fold_1", "train_end": "2016-09-21"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not found"):
        ret_train.resolve_train_end_from_fold(
            events_parquet=events_parquet, fold_id="wf_fold_99"
        )


def test_fine_tune_and_index_rejects_batch_size_below_two(tmp_path: Path) -> None:
    """MNRL requires at least one in-batch negative; batch_size < 2 must fail loudly."""

    events_parquet = tmp_path / "events.parquet"
    pd.DataFrame(_events_frame()).to_parquet(events_parquet, index=False)
    with pytest.raises(ValueError, match="batch_size >= 2"):
        ret_train.fine_tune_and_index(
            events_parquet=events_parquet,
            batch_size=1,
        )


def test_fine_tune_and_index_rejects_mutually_exclusive_flags(tmp_path: Path) -> None:
    events_parquet = tmp_path / "events.parquet"
    pd.DataFrame(_events_frame()).to_parquet(events_parquet, index=False)
    with pytest.raises(ValueError, match="mutually exclusive"):
        ret_train.fine_tune_and_index(
            events_parquet=events_parquet,
            train_end="2020-01-01",
            fold_id="wf_fold_3",
        )
