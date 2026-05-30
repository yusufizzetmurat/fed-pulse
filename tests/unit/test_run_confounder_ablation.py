"""Confounder-ablation runner tests (#495).

Covers the cell-spec resolution, the per-event control-vector math
(year FE, meeting-type FE, doc-length scalar), and the end-to-end
runner flow against a stubbed loader + trainer so the cell iteration
can be exercised without a real training package on disk.
"""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.run_confounder_ablation import (
    CANONICAL_CELLS,
    CANONICAL_MEETING_KINDS,
    ConfounderSpec,
    EventMetadata,
    _attach_confounder_block,
    _build_doc_length_scalar,
    _build_meeting_kind_one_hot,
    _build_year_one_hot,
    _group_metadata_by_date,
    _parse_args,
    _resolve_spec,
    _resolve_year_range,
    build_confounder_vector,
)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_parser_defaults_to_all_five_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_confounder_ablation",
            "--training-package-id",
            "tp_dummy",
        ],
    )
    args = _parse_args()
    assert tuple(args.cells) == CANONICAL_CELLS
    assert args.head_mode == "dual"
    assert args.regression_alpha == 0.5
    assert args.seeds == [11, 29, 47, 71, 97]


def test_parser_accepts_cell_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_confounder_ablation",
            "--training-package-id",
            "tp_dummy",
            "--cells",
            "baseline",
            "year_fe",
        ],
    )
    args = _parse_args()
    assert args.cells == ["baseline", "year_fe"]


def test_parser_rejects_unknown_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_confounder_ablation",
            "--training-package-id",
            "tp_dummy",
            "--cells",
            "definitely_not_a_cell",
        ],
    )
    with pytest.raises(SystemExit):
        _parse_args()


# ---------------------------------------------------------------------------
# Cell-builder math
# ---------------------------------------------------------------------------


def test_resolve_year_range_min_max_from_events() -> None:
    events = [
        EventMetadata(text_hash="a", event_date="2018-03-15", event_kind="statement", token_count=100),
        EventMetadata(text_hash="b", event_date="2022-12-14", event_kind="minutes", token_count=200),
        EventMetadata(text_hash="c", event_date="2020-06-10", event_kind="statement", token_count=150),
    ]
    assert _resolve_year_range(events) == (2018, 2022)


def test_resolve_year_range_rejects_empty() -> None:
    with pytest.raises(ValueError):
        _resolve_year_range([])


def test_year_one_hot_positions_year_relative_to_range() -> None:
    vector = _build_year_one_hot("2020-06-10", (2018, 2022))
    assert vector == [0.0, 0.0, 1.0, 0.0, 0.0]


def test_year_one_hot_zeros_when_year_out_of_range() -> None:
    vector = _build_year_one_hot("2030-01-01", (2018, 2022))
    assert vector == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_meeting_kind_one_hot_uses_canonical_position() -> None:
    for idx, kind in enumerate(CANONICAL_MEETING_KINDS):
        vector = _build_meeting_kind_one_hot(kind)
        assert sum(vector) == 1.0
        assert vector[idx] == 1.0


def test_meeting_kind_one_hot_zeros_unknown_kind() -> None:
    vector = _build_meeting_kind_one_hot("random_kind")
    assert vector == [0.0] * len(CANONICAL_MEETING_KINDS)


def test_doc_length_scalar_is_log1p_of_token_count() -> None:
    assert _build_doc_length_scalar(0) == [0.0]
    assert _build_doc_length_scalar(99) == [math.log1p(99)]
    assert _build_doc_length_scalar(2500) == [math.log1p(2500)]


def test_build_confounder_vector_concatenates_in_documented_order() -> None:
    """all_three spec: ``[year_fe || meeting_type_fe || doc_length]``."""

    spec = _resolve_spec("all_three", year_range=(2018, 2020))
    assert spec.width == 3 + len(CANONICAL_MEETING_KINDS) + 1
    meta = EventMetadata(
        text_hash="a",
        event_date="2019-06-15",
        event_kind="statement",
        token_count=512,
    )
    vector = build_confounder_vector(meta, spec)
    assert len(vector) == spec.width
    # First block: year FE for 2019 inside (2018, 2020).
    assert vector[:3] == [0.0, 1.0, 0.0]
    # Second block: meeting-kind FE for ``statement`` at index 0.
    meeting_block = vector[3 : 3 + len(CANONICAL_MEETING_KINDS)]
    assert meeting_block[0] == 1.0 and sum(meeting_block) == 1.0
    # Third block: scalar log(1 + 512).
    assert vector[-1] == math.log1p(512)


# ---------------------------------------------------------------------------
# Spec dispatch
# ---------------------------------------------------------------------------


def test_baseline_spec_width_is_zero() -> None:
    spec = _resolve_spec("baseline", year_range=(2018, 2020))
    assert spec.width == 0
    assert (
        spec.use_year_fe is False
        and spec.use_meeting_fe is False
        and spec.use_doc_length is False
    )


def test_year_fe_spec_width_matches_observed_year_count() -> None:
    spec = _resolve_spec("year_fe", year_range=(2018, 2023))
    assert spec.width == 6  # 2018..2023 inclusive


def test_meeting_type_fe_spec_width_matches_canonical_vocabulary() -> None:
    spec = _resolve_spec("meeting_type_fe", year_range=(2018, 2020))
    assert spec.width == len(CANONICAL_MEETING_KINDS)


def test_doc_length_spec_is_single_scalar() -> None:
    spec = _resolve_spec("doc_length", year_range=(2018, 2020))
    assert spec.width == 1


def test_unknown_cell_raises() -> None:
    with pytest.raises(ValueError):
        _resolve_spec("ghost_cell", year_range=(2018, 2020))


# ---------------------------------------------------------------------------
# Attachment adapter
# ---------------------------------------------------------------------------


class _Bar:
    """Minimal FeatureVector stand-in carrying just ``confounder_features``."""

    def __init__(self) -> None:
        self.confounder_features: list[float] | None = None


def _make_sequence(n_bars: int) -> list[_Bar]:
    return [_Bar() for _ in range(n_bars)]


def test_attach_confounder_block_writes_year_one_hot_on_every_bar() -> None:
    spec = _resolve_spec("year_fe", year_range=(2019, 2020))
    metadata_by_date = {
        "2019-03-20": [
            EventMetadata(
                text_hash="h1",
                event_date="2019-03-20",
                event_kind="statement",
                token_count=100,
            ),
        ],
        "2020-12-16": [
            EventMetadata(
                text_hash="h2",
                event_date="2020-12-16",
                event_kind="minutes",
                token_count=300,
            ),
        ],
    }
    sequences = [_make_sequence(5), _make_sequence(5)]
    event_dates = ["2019-03-20", "2020-12-16"]
    _attach_confounder_block(
        sequences,
        event_dates,
        metadata_by_date=metadata_by_date,
        spec=spec,
    )
    for bar in sequences[0]:
        assert bar.confounder_features == [1.0, 0.0]
    for bar in sequences[1]:
        assert bar.confounder_features == [0.0, 1.0]


def test_attach_confounder_block_baseline_is_no_op() -> None:
    spec = _resolve_spec("baseline", year_range=(2019, 2020))
    sequences = [_make_sequence(3)]
    _attach_confounder_block(
        sequences,
        ["2019-06-20"],
        metadata_by_date={},
        spec=spec,
    )
    for bar in sequences[0]:
        assert bar.confounder_features is None


def test_attach_confounder_block_advances_cursor_per_text_hash() -> None:
    """Two sequences sharing one event_date must read consecutive metadata
    rows in text_hash order — the same positional contract the loader
    follows when emitting multiple events per date."""

    spec = _resolve_spec("meeting_type_fe", year_range=(2019, 2019))
    metadata_by_date = {
        "2019-06-19": [
            EventMetadata(
                text_hash="hash_a",
                event_date="2019-06-19",
                event_kind="statement",
                token_count=300,
            ),
            EventMetadata(
                text_hash="hash_b",
                event_date="2019-06-19",
                event_kind="press_conference",
                token_count=900,
            ),
        ],
    }
    sequences = [_make_sequence(2), _make_sequence(2)]
    event_dates = ["2019-06-19", "2019-06-19"]
    _attach_confounder_block(
        sequences,
        event_dates,
        metadata_by_date=metadata_by_date,
        spec=spec,
    )
    statement_index = CANONICAL_MEETING_KINDS.index("statement")
    press_conf_index = CANONICAL_MEETING_KINDS.index("press_conference")
    assert sequences[0][0].confounder_features is not None
    assert sequences[0][0].confounder_features[statement_index] == 1.0
    assert sequences[1][0].confounder_features is not None
    assert sequences[1][0].confounder_features[press_conf_index] == 1.0


def test_attach_confounder_block_unknown_date_fills_zeros() -> None:
    spec = _resolve_spec("year_fe", year_range=(2019, 2020))
    sequences = [_make_sequence(3)]
    _attach_confounder_block(
        sequences,
        ["2024-01-01"],
        metadata_by_date={},
        spec=spec,
    )
    for bar in sequences[0]:
        assert bar.confounder_features == [0.0, 0.0]


def test_group_metadata_by_date_sorts_buckets_by_text_hash() -> None:
    events = [
        EventMetadata(text_hash="z", event_date="2019-06-19", event_kind="statement", token_count=100),
        EventMetadata(text_hash="a", event_date="2019-06-19", event_kind="press_conference", token_count=500),
        EventMetadata(text_hash="m", event_date="2020-01-29", event_kind="statement", token_count=200),
    ]
    bucketed = _group_metadata_by_date(events)
    assert [m.text_hash for m in bucketed["2019-06-19"]] == ["a", "z"]
    assert [m.text_hash for m in bucketed["2020-01-29"]] == ["m"]


# ---------------------------------------------------------------------------
# End-to-end smoke: runner walks the (cell, seed, fold) cube with a stubbed
# loader + trainer.
# ---------------------------------------------------------------------------


class _StubSplit:
    fold_id = "fold_001"
    protocol = "walk-forward"

    def __init__(self) -> None:
        self.train: list[list[Any]] = [[_Bar(), _Bar(), _Bar()]]
        self.val: list[list[Any]] = [[_Bar(), _Bar(), _Bar()]]
        self.test: list[list[Any]] = [[_Bar(), _Bar(), _Bar()]]
        self.train_event_dates: list[str] = ["2019-06-19"]
        self.val_event_dates: list[str] = ["2020-03-15"]
        self.test_event_dates: list[str] = ["2021-09-22"]


def _make_metadata_lookup() -> dict[str, list[EventMetadata]]:
    rows = [
        EventMetadata(text_hash="h1", event_date="2019-06-19", event_kind="statement", token_count=512),
        EventMetadata(text_hash="h2", event_date="2020-03-15", event_kind="press_conference", token_count=800),
        EventMetadata(text_hash="h3", event_date="2021-09-22", event_kind="minutes", token_count=1024),
    ]
    return _group_metadata_by_date(rows)


def test_run_one_cell_threads_input_size_through_to_model_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch", reason="train_model import path needs torch")
    from app.models.config import RICH_FEATURE_SIZE
    from app.training import loaders as loaders_module
    from app.training import loop as loop_module
    from scripts import run_confounder_ablation as runner

    captured: dict[str, list[Any]] = {"train_calls": [], "splits": []}

    def _fake_load_walk_forward_split(**_kwargs: Any) -> _StubSplit:
        split = _StubSplit()
        captured["splits"].append(split)
        return split

    def _fake_train_model(**kwargs: Any) -> SimpleNamespace:
        captured["train_calls"].append(kwargs)
        return SimpleNamespace(
            summary=SimpleNamespace(
                test_metrics=SimpleNamespace(
                    regime_f1_macro=0.42,
                    regime_accuracy=0.42,
                    regime_loss=1.0,
                    regression_rmse_log_rv=0.5,
                    regression_mae_log_rv=0.4,
                    regression_loss=0.25,
                )
            )
        )

    monkeypatch.setattr(
        loaders_module, "load_walk_forward_split", _fake_load_walk_forward_split
    )
    monkeypatch.setattr(loop_module, "train_model", _fake_train_model)

    args = SimpleNamespace(
        training_package_id="tp_dummy",
        head_mode="dual",
        regression_alpha=0.5,
        hidden_size=64,
        epochs=1,
    )
    spec = runner._resolve_spec("all_three", year_range=(2018, 2022))
    runner._run_one_cell(
        spec,
        seed=11,
        args=args,
        fold_ids=["fold_001"],
        metadata_by_date=_make_metadata_lookup(),
    )
    assert len(captured["train_calls"]) == 1
    model_config = captured["train_calls"][0]["model_config"]
    expected_width = (2022 - 2018 + 1) + len(CANONICAL_MEETING_KINDS) + 1
    assert model_config.input_size == RICH_FEATURE_SIZE + expected_width

    # The stub split carries one sequence per partition; the attachment
    # adapter must have written a ``spec.width``-length vector onto each
    # bar of the train sequence.
    split = captured["splits"][0]
    for bar in split.train[0]:
        assert bar.confounder_features is not None
        assert len(bar.confounder_features) == expected_width
    # Year FE block should fire on the 2019 train event.
    year_block = split.train[0][0].confounder_features[: 2022 - 2018 + 1]
    assert year_block[2019 - 2018] == 1.0


def test_run_one_cell_baseline_writes_no_confounder_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch", reason="train_model import path needs torch")
    from app.models.config import RICH_FEATURE_SIZE
    from app.training import loaders as loaders_module
    from app.training import loop as loop_module
    from scripts import run_confounder_ablation as runner

    captured: dict[str, list[Any]] = {"train_calls": []}

    def _fake_load_walk_forward_split(**_kwargs: Any) -> _StubSplit:
        return _StubSplit()

    def _fake_train_model(**kwargs: Any) -> SimpleNamespace:
        captured["train_calls"].append(kwargs)
        return SimpleNamespace(
            summary=SimpleNamespace(
                test_metrics=SimpleNamespace(
                    regime_f1_macro=0.4,
                    regime_accuracy=0.4,
                    regime_loss=1.0,
                    regression_rmse_log_rv=0.5,
                    regression_mae_log_rv=0.4,
                    regression_loss=0.25,
                )
            )
        )

    monkeypatch.setattr(
        loaders_module, "load_walk_forward_split", _fake_load_walk_forward_split
    )
    monkeypatch.setattr(loop_module, "train_model", _fake_train_model)

    args = SimpleNamespace(
        training_package_id="tp_dummy",
        head_mode="dual",
        regression_alpha=0.5,
        hidden_size=64,
        epochs=1,
    )
    spec = runner._resolve_spec("baseline", year_range=(2018, 2022))
    runner._run_one_cell(
        spec,
        seed=11,
        args=args,
        fold_ids=["fold_001"],
        metadata_by_date=_make_metadata_lookup(),
    )
    model_config = captured["train_calls"][0]["model_config"]
    assert model_config.input_size == RICH_FEATURE_SIZE


def test_confounder_spec_width_matches_active_blocks() -> None:
    """``ConfounderSpec.width`` is the load-bearing width contract every
    consumer relies on (model input_size + attachment vector length)."""

    spec = ConfounderSpec(
        cell="all_three",
        use_year_fe=True,
        use_meeting_fe=True,
        use_doc_length=True,
        year_range=(2018, 2020),
    )
    assert spec.width == 3 + len(CANONICAL_MEETING_KINDS) + 1

    just_doc = ConfounderSpec(
        cell="doc_length",
        use_year_fe=False,
        use_meeting_fe=False,
        use_doc_length=True,
        year_range=(0, -1),
    )
    assert just_doc.width == 1
