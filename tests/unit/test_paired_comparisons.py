"""Unit tests for app.eval.paired_comparisons (#497)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from app.eval.paired_comparisons import (
    compute_paired_comparisons,
    effect_size,
    extract_paired_deltas,
    holm_bonferroni,
    wilcoxon_signed_rank,
)


def test_wilcoxon_positive_shift_significant() -> None:
    deltas = [0.05, 0.04, 0.06, 0.03, 0.05, 0.04, 0.06, 0.03, 0.05, 0.04]
    _, pval = wilcoxon_signed_rank(deltas)
    assert not math.isnan(pval)
    assert pval < 0.05


def test_wilcoxon_all_zero_returns_nan() -> None:
    _, pval = wilcoxon_signed_rank([0.0, 0.0, 0.0])
    assert math.isnan(pval)


def test_wilcoxon_one_nonzero_returns_nan() -> None:
    """scipy.stats.wilcoxon is undefined for n<2; the wrapper must early-return."""
    stat, pval = wilcoxon_signed_rank([0.0, 0.05])
    assert math.isnan(stat)
    assert math.isnan(pval)


def test_wilcoxon_symmetric_large_p() -> None:
    deltas = [0.1, -0.1, 0.2, -0.2, 0.05, -0.05]
    _, pval = wilcoxon_signed_rank(deltas)
    assert pval > 0.1


def test_effect_size_positive_deltas() -> None:
    d = effect_size([0.05, 0.04, 0.06, 0.05, 0.03])
    assert d > 0


def test_effect_size_nan_for_single() -> None:
    assert math.isnan(effect_size([0.1]))


def test_effect_size_nan_for_constant() -> None:
    assert math.isnan(effect_size([0.1, 0.1, 0.1]))


def test_holm_inflates_p_values() -> None:
    p_values = [0.01, 0.04, 0.02, 0.30]
    corrected = holm_bonferroni(p_values)
    for orig, corr in zip(p_values, corrected, strict=False):
        assert corr >= orig - 1e-12


def test_holm_single_unchanged() -> None:
    assert holm_bonferroni([0.03])[0] == pytest.approx(0.03)


def test_holm_most_significant_multiplied_by_n() -> None:
    p = [0.05, 0.01, 0.02]
    corrected = holm_bonferroni(p)
    assert corrected[1] == pytest.approx(0.03)


def _write_sweep(tmp_path: Path) -> Path:
    sweep = {
        "trials": {
            "classification": [
                {
                    "seed": 11,
                    "folds": [
                        {"fold_id": "wf_fold_1", "metrics": {"regime_f1_macro": 0.40}},
                        {"fold_id": "wf_fold_2", "metrics": {"regime_f1_macro": 0.45}},
                    ],
                },
                {
                    "seed": 29,
                    "folds": [
                        {"fold_id": "wf_fold_1", "metrics": {"regime_f1_macro": 0.42}},
                        {"fold_id": "wf_fold_2", "metrics": {"regime_f1_macro": 0.47}},
                    ],
                },
            ],
            "dual": [
                {
                    "seed": 11,
                    "folds": [
                        {"fold_id": "wf_fold_1", "metrics": {"regime_f1_macro": 0.43}},
                        {"fold_id": "wf_fold_2", "metrics": {"regime_f1_macro": 0.48}},
                    ],
                },
                {
                    "seed": 29,
                    "folds": [
                        {"fold_id": "wf_fold_1", "metrics": {"regime_f1_macro": 0.44}},
                        {"fold_id": "wf_fold_2", "metrics": {"regime_f1_macro": 0.50}},
                    ],
                },
            ],
        }
    }
    p = tmp_path / "sweep.json"
    p.write_text(json.dumps(sweep), encoding="utf-8")
    return p


def test_extract_deltas_count(tmp_path: Path) -> None:
    sweep = _write_sweep(tmp_path)
    deltas, _ = extract_paired_deltas(sweep, "classification", "dual", "regime_f1_macro")
    assert len(deltas) == 4


def test_extract_deltas_sign(tmp_path: Path) -> None:
    sweep = _write_sweep(tmp_path)
    deltas, _ = extract_paired_deltas(sweep, "classification", "dual", "regime_f1_macro")
    assert all(d < 0 for d in deltas)


def test_compute_paired_holm_applied(tmp_path: Path) -> None:
    sweep = _write_sweep(tmp_path)
    results = compute_paired_comparisons(
        [sweep], [("classification", "dual")], "regime_f1_macro", n_resamples=100
    )
    assert len(results) == 1
    r = results[0]
    assert not math.isnan(r.mean_delta)
    assert r.p_value_holm >= r.p_value - 1e-12
