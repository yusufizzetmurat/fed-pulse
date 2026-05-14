"""Smoke tests for the CPU-side of continued_pretraining.

The actual MLM training run is GPU-bound and exercised via `make`-driven
smoke runs, not pytest. These tests cover the pair-collection paths that
shape data before it hits the model.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from app.data import continued_pretraining as cpt


def test_iter_local_pairs_skips_missing_and_empty(tmp_path: Path) -> None:
    (tmp_path / "speeches.json").write_text(
        json.dumps(
            [
                {"text": "Inflation pressures remain elevated."},
                {"body": "Activity has expanded at a moderate pace."},
                {"text": "   "},  # empty after strip → dropped
                {"unrelated": "no text"},
            ]
        ),
        encoding="utf-8",
    )
    pairs = cpt._iter_local_pairs(tmp_path, ["speeches.json", "missing.json"])
    assert len(pairs) == 2
    assert {p["sequenceA"] for p in pairs} == {
        "Inflation pressures remain elevated.",
        "Activity has expanded at a moderate pace.",
    }
    assert all(p["sequenceB"] == "" for p in pairs)
    assert all(p["next_sentence_label"] == 0 for p in pairs)


def test_iter_local_pairs_skips_non_list_payload(tmp_path: Path) -> None:
    (tmp_path / "wrong.json").write_text(json.dumps({"text": "not a list"}), encoding="utf-8")
    pairs = cpt._iter_local_pairs(tmp_path, ["wrong.json"])
    assert pairs == []


def _install_fake_datasets(monkeypatch, rows: list[dict]) -> None:
    fake = types.SimpleNamespace()
    fake.load_dataset = lambda dataset_id, **kw: iter(rows)
    monkeypatch.setitem(sys.modules, "datasets", fake)


def test_bis_pair_stream_filters_empty_and_respects_max_rows(monkeypatch) -> None:
    _install_fake_datasets(
        monkeypatch,
        [
            {"sequenceA": "A1", "sequenceB": "B1", "next_sentence_label": 0},
            {"sequenceA": "", "sequenceB": "B2", "next_sentence_label": 0},  # empty A → dropped
            {"sequenceA": "A3", "sequenceB": "B3", "next_sentence_label": 1},
            {"sequenceA": "A4", "sequenceB": None, "next_sentence_label": 0},
            {"sequenceA": "A5", "sequenceB": "B5", "next_sentence_label": 1},
        ],
    )
    rows = list(
        cpt._bis_pair_stream(
            "samchain/BIS_speeches_97_23_MLM",
            None,
            streaming=False,
            max_rows=3,
        )
    )
    assert len(rows) == 3
    assert [r["sequenceA"] for r in rows] == ["A1", "A3", "A4"]
    assert rows[1]["next_sentence_label"] == 1
    assert rows[2]["sequenceB"] == ""


def test_collect_pairs_substrate_local(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "chair_speeches.json").write_text(
        json.dumps([{"text": "Local speech text."}]),
        encoding="utf-8",
    )
    args = cpt._parse_args(
        [
            "--substrate",
            "local",
            "--data-dir",
            str(tmp_path),
            "--corpus-files",
            "chair_speeches.json",
        ]
    )
    pairs = cpt._collect_pairs(args)
    assert pairs == [
        {"sequenceA": "Local speech text.", "sequenceB": "", "next_sentence_label": 0}
    ]


def test_collect_pairs_substrate_bis_uses_streaming(monkeypatch) -> None:
    _install_fake_datasets(
        monkeypatch,
        [
            {"sequenceA": "BIS A1", "sequenceB": "BIS B1", "next_sentence_label": 0},
            {"sequenceA": "BIS A2", "sequenceB": "BIS B2", "next_sentence_label": 1},
        ],
    )
    args = cpt._parse_args(
        [
            "--substrate",
            "bis",
            "--streaming",
            "--max-rows",
            "10",
        ]
    )
    pairs = cpt._collect_pairs(args)
    assert len(pairs) == 2
    assert pairs[0]["sequenceA"] == "BIS A1"


def test_collect_pairs_substrate_both_respects_max_rows(monkeypatch, tmp_path: Path) -> None:
    _install_fake_datasets(
        monkeypatch,
        [
            {"sequenceA": "BIS A1", "sequenceB": "BIS B1", "next_sentence_label": 0},
            {"sequenceA": "BIS A2", "sequenceB": "BIS B2", "next_sentence_label": 0},
        ],
    )
    (tmp_path / "chair_speeches.json").write_text(
        json.dumps([{"text": "Local 1."}, {"text": "Local 2."}]),
        encoding="utf-8",
    )
    args = cpt._parse_args(
        [
            "--substrate",
            "both",
            "--data-dir",
            str(tmp_path),
            "--corpus-files",
            "chair_speeches.json",
            "--max-rows",
            "3",
        ]
    )
    pairs = cpt._collect_pairs(args)
    assert len(pairs) == 3
    assert pairs[0]["sequenceA"].startswith("BIS")
    assert pairs[2]["sequenceA"].startswith("Local")


def test_parse_args_objective_validates_choices() -> None:
    args = cpt._parse_args(["--objective", "mlm"])
    assert args.objective == "mlm"
    with pytest.raises(SystemExit):
        cpt._parse_args(["--objective", "nonsense"])
