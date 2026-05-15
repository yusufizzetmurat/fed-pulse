"""Unit tests for app.evaluation.bakeoff_aggregator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation import bakeoff_aggregator as ba


def _write_aggregate(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def two_encoders(tmp_path: Path) -> Path:
    """A typical finetune_batch output with two encoders × five seeds each."""

    _write_aggregate(
        tmp_path / "run_a" / "aggregate.json",
        {
            "by_encoder": {
                "finbert": {
                    "checkpoint": "ProsusAI/finbert",
                    "per_seed": {
                        "11": {"macro_f1": 0.68, "weighted_f1": 0.71, "accuracy": 0.74},
                        "29": {"macro_f1": 0.70, "weighted_f1": 0.72, "accuracy": 0.75},
                        "47": {"macro_f1": 0.69, "weighted_f1": 0.71, "accuracy": 0.74},
                        "71": {"macro_f1": 0.71, "weighted_f1": 0.73, "accuracy": 0.76},
                        "97": {"macro_f1": 0.67, "weighted_f1": 0.70, "accuracy": 0.73},
                    },
                },
                "bge_large_en_v15": {
                    "checkpoint": "BAAI/bge-large-en-v1.5",
                    "per_seed": {
                        "11": {"macro_f1": 0.62, "weighted_f1": 0.65, "accuracy": 0.69},
                        "29": {"macro_f1": 0.63, "weighted_f1": 0.66, "accuracy": 0.70},
                        "47": {"macro_f1": 0.61, "weighted_f1": 0.64, "accuracy": 0.68},
                        "71": {"macro_f1": 0.64, "weighted_f1": 0.66, "accuracy": 0.70},
                        "97": {"macro_f1": 0.60, "weighted_f1": 0.63, "accuracy": 0.67},
                    },
                },
            }
        },
    )
    return tmp_path


def test_aggregate_orders_by_macro_f1_descending(two_encoders: Path) -> None:
    rows, _markdown, _payload = ba.aggregate(two_encoders, n_resamples=100, seed=11)
    assert [r.encoder_key for r in rows] == ["finbert", "bge_large_en_v15"]
    assert rows[0].macro_f1_ci.point > rows[1].macro_f1_ci.point


def test_aggregate_attaches_ci_to_every_metric(two_encoders: Path) -> None:
    rows, _markdown, _payload = ba.aggregate(two_encoders, n_resamples=100, seed=11)
    for row in rows:
        for ci in (row.macro_f1_ci, row.weighted_f1_ci, row.accuracy_ci):
            assert ci.lo <= ci.point <= ci.hi
            assert 0.0 <= ci.lo and ci.hi <= 1.0


def test_markdown_renders_all_rows(two_encoders: Path) -> None:
    rows, markdown, _payload = ba.aggregate(two_encoders, n_resamples=100, seed=11)
    assert "finbert" in markdown
    assert "bge_large_en_v15" in markdown
    assert "macro-F1" in markdown
    assert markdown.count("\n| ") >= len(rows)


def test_aggregate_merges_multiple_aggregate_files(tmp_path: Path) -> None:
    _write_aggregate(
        tmp_path / "first" / "aggregate.json",
        {
            "by_encoder": {
                "finbert": {
                    "checkpoint": "ProsusAI/finbert",
                    "per_seed": {
                        "11": {"macro_f1": 0.7, "weighted_f1": 0.7, "accuracy": 0.7},
                    },
                },
            }
        },
    )
    _write_aggregate(
        tmp_path / "second" / "aggregate.json",
        {
            "by_encoder": {
                "finbert": {
                    "checkpoint": "ProsusAI/finbert",
                    "per_seed": {
                        "29": {"macro_f1": 0.72, "weighted_f1": 0.72, "accuracy": 0.72},
                    },
                },
            }
        },
    )
    rows, _markdown, _payload = ba.aggregate(tmp_path, n_resamples=50, seed=11)
    assert len(rows) == 1
    assert sorted(rows[0].seeds) == [11, 29]
    assert len(rows[0].macro_f1_values) == 2


def test_aggregate_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ba.aggregate(tmp_path / "missing", n_resamples=10, seed=11)
