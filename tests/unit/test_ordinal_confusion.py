"""Unit tests for app.eval.ordinal_confusion (#496)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from app.eval.ordinal_confusion import (
    compute_ordinal_confusion,
    decompose_ordinal,
    extract_cells,
)


def test_perfect_matrix_zero_errors() -> None:
    cm = [[5, 0, 0], [0, 4, 0], [0, 0, 3]]
    d = decompose_ordinal(cm)
    assert d["total_errors"] == 0
    assert d["adjacent_error_rate"] == 0.0
    assert d["ordinal_accuracy"] == 1.0


def test_all_adjacent_errors() -> None:
    cm = [[0, 2, 0], [0, 0, 3], [0, 0, 5]]
    d = decompose_ordinal(cm)
    assert d["adjacent_errors"] == 5
    assert d["non_adjacent_errors"] == 0
    assert d["adjacent_error_rate"] == pytest.approx(1.0)
    assert d["ordinal_accuracy"] == pytest.approx(1.0)


def test_all_non_adjacent_errors() -> None:
    cm = [[0, 0, 4], [0, 5, 0], [0, 0, 0]]
    d = decompose_ordinal(cm)
    assert d["non_adjacent_errors"] == 4
    assert d["non_adjacent_error_rate"] == pytest.approx(1.0)
    assert d["ordinal_accuracy"] == pytest.approx(0.0)


def test_mixed_errors_rates() -> None:
    cm = [[0, 2, 1], [0, 3, 0], [0, 0, 2]]
    d = decompose_ordinal(cm)
    assert d["total_errors"] == 3
    assert d["adjacent_error_rate"] == pytest.approx(2 / 3)
    assert d["non_adjacent_error_rate"] == pytest.approx(1 / 3)
    assert d["ordinal_accuracy"] == pytest.approx(2 / 3)


def _write_no_breakdown(tmp_path: Path) -> Path:
    p = tmp_path / "no_bd.json"
    p.write_text(json.dumps({
        "trials": {
            "classification": [{
                "seed": 11,
                "folds": [{"fold_id": "wf_fold_1", "metrics": {"regime_f1_macro": 0.42}}],
            }]
        }
    }), encoding="utf-8")
    return p


def test_skip_missing_breakdown_with_warning(tmp_path: Path) -> None:
    path = _write_no_breakdown(tmp_path)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cells = extract_cells(path)
    assert cells == []
    assert any("classification_breakdown" in str(warning.message) for warning in w)


def _write_with_breakdown(tmp_path: Path) -> Path:
    adj_cm = [[0, 2, 0], [0, 4, 1], [0, 0, 3]]
    nonadj_cm = [[0, 0, 3], [0, 5, 0], [0, 0, 2]]
    p = tmp_path / "with_bd.json"
    p.write_text(json.dumps({
        "trials": {
            "classification": [
                {
                    "seed": 11,
                    "folds": [{"fold_id": "wf_fold_1", "metrics": {
                        "regime_f1_macro": 0.42,
                        "classification_breakdown": {"confusion_matrix": adj_cm},
                    }}],
                },
                {
                    "seed": 29,
                    "folds": [{"fold_id": "wf_fold_1", "metrics": {
                        "regime_f1_macro": 0.40,
                        "classification_breakdown": {"confusion_matrix": nonadj_cm},
                    }}],
                },
            ]
        }
    }), encoding="utf-8")
    return p


def test_end_to_end_aggregation(tmp_path: Path) -> None:
    path = _write_with_breakdown(tmp_path)
    result = compute_ordinal_confusion([path])
    assert result["n_cells_total"] == 2
    assert len(result["arm_summaries"]) == 1
    s = result["arm_summaries"][0]
    assert s["head_mode"] == "classification"
    assert s["ordinal_accuracy"]["mean"] == pytest.approx(0.5, abs=0.01)
