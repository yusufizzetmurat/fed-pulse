"""Unit tests for the ensemble aggregator (#226 + Phase 5 multi-run).

Locks the three aggregation strategies against synthetic inputs where
the analytical answer is known, asserts the alignment + missing-arch
edge cases the production runner must handle, and exercises the Phase
5 multi-run logit-average path (calibrated logit-averaging across
distinct training runs with per-fold conformal calibration).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from app.evaluation.ensemble_aggregator import (
    DEFAULT_REDUNDANCY_KAPPA_THRESHOLD,
    MultiRunEnsembleResult,
    RunSpec,
    _cohen_kappa,
    _mean_logit,
    _mean_softmax,
    _plurality_vote,
    aggregate,
    aggregate_multi_run_ensemble,
)


def _trial(
    *,
    architecture: str,
    fold_id: str,
    seed: int,
    predictions: list[int],
    targets: list[int],
    class_scores: list[list[float]] | None = None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "model_config": {"hidden_size": 256, "num_layers": 2, "dropout": 0.2},
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "test_metrics": {
            "predictions": predictions,
            "targets": targets,
            "classification_breakdown": {"macro_f1": 0.5},
        },
    }
    if class_scores is not None:
        summary["test_metrics"]["class_scores"] = class_scores  # type: ignore[index]
    return {
        "architecture": architecture,
        "fold_id": fold_id,
        "seed": seed,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Per-row strategy helpers
# ---------------------------------------------------------------------------


def test_mean_logit_picks_argmax_of_logsum() -> None:
    """Two architectures: scores [0.7, 0.2, 0.1] and [0.6, 0.3, 0.1]. The
    log-sum across class 0 dominates; class 0 wins."""
    assert _mean_logit([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]]) == 0


def test_mean_softmax_picks_argmax_of_sum() -> None:
    """Three architectures: each model's argmax is 0, 1, 2 respectively.
    Class 1's averaged softmax probability is highest -> mean_softmax
    picks 1, plurality picks the lowest tied class."""
    scores = [
        [0.5, 0.3, 0.2],
        [0.2, 0.7, 0.1],
        [0.2, 0.3, 0.5],
    ]
    assert _mean_softmax(scores) == 1


def test_plurality_breaks_ties_to_lowest_class() -> None:
    """Two architectures vote class 1 and class 2 — tied at 1 vote
    each. Tie-breaker is the lowest class index."""
    assert _plurality_vote([1, 2]) == 1


def test_plurality_consensus() -> None:
    assert _plurality_vote([2, 2, 0]) == 2


# ---------------------------------------------------------------------------
# Aggregator end-to-end
# ---------------------------------------------------------------------------


def test_ensemble_requires_two_architectures() -> None:
    """The aggregator raises ValueError on a single architecture."""
    trial = _trial(
        architecture="lstm",
        fold_id="wf_fold_1",
        seed=11,
        predictions=[0, 1, 2] * 5,
        targets=[0, 1, 2] * 5,
        class_scores=[[1.0, 0.0, 0.0]] * 15,
    )
    with pytest.raises(ValueError, match="at least 2 architectures"):
        aggregate({"lstm": [{"trials": [trial], "selection_metric": "macro_f1"}]})


def test_ensemble_aligns_per_fold_seed_cells() -> None:
    """Two architectures' (fold, seed)-aligned predictions are averaged
    into the ensemble; the cell breakdown matches the rule."""

    lstm_trial = _trial(
        architecture="lstm",
        fold_id="wf_fold_1",
        seed=11,
        predictions=[0, 0, 1],
        targets=[0, 1, 1],
        class_scores=[[0.7, 0.2, 0.1], [0.5, 0.3, 0.2], [0.2, 0.7, 0.1]],
    )
    tft_trial = _trial(
        architecture="tft",
        fold_id="wf_fold_1",
        seed=11,
        predictions=[0, 1, 1],
        targets=[0, 1, 1],
        class_scores=[[0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [0.3, 0.5, 0.2]],
    )
    payload = aggregate(
        {
            "lstm": [{"trials": [lstm_trial], "selection_metric": "macro_f1"}],
            "tft": [{"trials": [tft_trial], "selection_metric": "macro_f1"}],
        },
        strategies=("mean_softmax",),
        n_classes=3,
        n_resamples=20,
    )
    per_cell = payload["per_cell"]
    pooled = payload["pooled"]
    # One cell, one strategy -> one per_cell row + one pooled row.
    assert len(per_cell) == 1
    assert len(pooled) == 1
    cell = per_cell[0]
    assert cell.fold_id == "wf_fold_1"
    assert cell.seed == 11
    # Row 1: lstm scores [0.7,0.2,0.1], tft scores [0.6,0.3,0.1] -> mean
    # [0.65,0.25,0.10] -> argmax 0. Row 2: lstm [0.5,0.3,0.2] + tft
    # [0.3,0.6,0.1] -> mean [0.4,0.45,0.15] -> 1. Row 3: lstm
    # [0.2,0.7,0.1] + tft [0.3,0.5,0.2] -> mean [0.25,0.6,0.15] -> 1.
    # Targets are [0, 1, 1] -> all three correct -> macro-F1 = 1.0 over
    # classes 0 and 1 (class 2 absent in this minimal cell).
    assert cell.breakdown.macro_f1 == pytest.approx(1.0, rel=0, abs=1e-6)


def test_ensemble_skips_cells_with_missing_architectures() -> None:
    """If a cell has only one of the two architectures' predictions, the
    aggregator drops the cell rather than computing a biased ensemble."""

    lstm_a = _trial(
        architecture="lstm",
        fold_id="wf_fold_1",
        seed=11,
        predictions=[0, 1, 2] * 3,
        targets=[0, 1, 2] * 3,
        class_scores=[[1.0, 0.0, 0.0]] * 9,
    )
    lstm_b = _trial(
        architecture="lstm",
        fold_id="wf_fold_2",
        seed=11,
        predictions=[0, 1, 2] * 3,
        targets=[0, 1, 2] * 3,
        class_scores=[[1.0, 0.0, 0.0]] * 9,
    )
    tft_a = _trial(
        architecture="tft",
        fold_id="wf_fold_1",
        seed=11,
        predictions=[0, 1, 2] * 3,
        targets=[0, 1, 2] * 3,
        class_scores=[[1.0, 0.0, 0.0]] * 9,
    )
    payload = aggregate(
        {
            "lstm": [
                {"trials": [lstm_a, lstm_b], "selection_metric": "macro_f1"}
            ],
            "tft": [{"trials": [tft_a], "selection_metric": "macro_f1"}],
        },
        strategies=("plurality_vote",),
        n_classes=3,
        n_resamples=20,
    )
    per_cell = payload["per_cell"]
    # Only the (wf_fold_1, 11) cell is shared; wf_fold_2 has lstm only.
    assert len(per_cell) == 1
    assert per_cell[0].fold_id == "wf_fold_1"


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    from app.evaluation.ensemble_aggregator import main

    for arch in ("lstm", "tft"):
        sub = tmp_path / arch
        sub.mkdir()
        trial = _trial(
            architecture=arch,
            fold_id="wf_fold_1",
            seed=11,
            predictions=[0, 1, 2] * 5,
            targets=[0, 1, 2] * 5,
            class_scores=[[0.8, 0.1, 0.1]] * 15,
        )
        (sub / "forecaster_sweep_results.json").write_text(
            json.dumps({"trials": [trial], "selection_metric": "macro_f1"})
        )
    rc = main(
        [
            "--arch-sweep-dir",
            str(tmp_path),
            "--n-resamples",
            "10",
        ]
    )
    assert rc == 0
    assert (tmp_path / "ensemble_results.json").exists()
    assert (tmp_path / "ensemble_results.md").exists()


# ---------------------------------------------------------------------------
# Phase 5 multi-run logit-average + per-fold conformal
# ---------------------------------------------------------------------------


def _run_trials(
    *,
    fold_seed_layout: list[tuple[str, int]],
    per_trial_preds: list[list[int]],
    per_trial_targets: list[list[int]],
    per_trial_scores: list[list[list[float]]],
) -> list[dict[str, object]]:
    """Build a list of trial dicts matching the existing sweep-JSON shape."""

    assert len(fold_seed_layout) == len(per_trial_preds)
    assert len(fold_seed_layout) == len(per_trial_targets)
    assert len(fold_seed_layout) == len(per_trial_scores)
    out: list[dict[str, object]] = []
    for (fold, seed), preds, targets, scores in zip(
        fold_seed_layout, per_trial_preds, per_trial_targets, per_trial_scores
    ):
        out.append(
            _trial(
                architecture="dummy",
                fold_id=fold,
                seed=seed,
                predictions=preds,
                targets=targets,
                class_scores=scores,
            )
        )
    return out


def _write_run_blob(path: Path, trials: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"trials": trials, "selection_metric": "macro_f1"})
    )
    return path


def test_multi_run_logit_average_picks_argmax_of_log_mean(tmp_path: Path) -> None:
    """Two runs x two folds x three trials x three classes.

    Per-trial logits are picked so the analytical ensemble argmax is
    known by hand: averaging log-probabilities is equivalent to a
    geometric mean of softmax probabilities, and the row-wise mean
    here picks class 0 on every row (each run is confident on the
    same row in agreement, and the geometric mean preserves the
    consensus).
    """

    layout = [("wf_fold_1", 11), ("wf_fold_2", 11)]
    # Each trial has 3 rows; targets are [0, 0, 0] so a perfect
    # ensemble = macro-F1 of 1.0 on the only-class-with-support 0.
    targets = [[0, 0, 0], [0, 0, 0]]
    preds_run_a = [[0, 0, 0], [0, 0, 0]]
    preds_run_b = [[0, 0, 0], [0, 0, 0]]
    scores_run_a = [
        [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.55, 0.3, 0.15]],
        [[0.65, 0.2, 0.15], [0.5, 0.3, 0.2], [0.7, 0.15, 0.15]],
    ]
    scores_run_b = [
        [[0.65, 0.25, 0.1], [0.55, 0.35, 0.1], [0.6, 0.25, 0.15]],
        [[0.7, 0.2, 0.1], [0.55, 0.3, 0.15], [0.6, 0.25, 0.15]],
    ]
    run_a = _run_trials(
        fold_seed_layout=layout,
        per_trial_preds=preds_run_a,
        per_trial_targets=targets,
        per_trial_scores=scores_run_a,
    )
    run_b = _run_trials(
        fold_seed_layout=layout,
        per_trial_preds=preds_run_b,
        per_trial_targets=targets,
        per_trial_scores=scores_run_b,
    )
    path_a = _write_run_blob(tmp_path / "run_a" / "forecaster_sweep_results.json", run_a)
    path_b = _write_run_blob(tmp_path / "run_b" / "forecaster_sweep_results.json", run_b)
    specs = [
        RunSpec(
            run_id="run_a",
            architecture="lstm",
            encoder_alias="none",
            seed=11,
            results_path=str(path_a),
        ),
        RunSpec(
            run_id="run_b",
            architecture="tft",
            encoder_alias="none",
            seed=11,
            results_path=str(path_b),
        ),
    ]
    # Override the redundancy guard so both identical-prediction runs
    # survive — this test exercises the averaging math, not the
    # dedup pass.
    result = aggregate_multi_run_ensemble(
        specs, conformal_alpha=0.2, redundancy_kappa_threshold=1.5
    )
    assert isinstance(result, MultiRunEnsembleResult)
    # Both runs agree on every row -> kept = both.
    assert set(result.kept_run_ids) == {"run_a", "run_b"}
    # Pooled macro-F1 = 1.0 on the only class with support.
    assert result.pooled_breakdown.macro_f1 == pytest.approx(1.0)
    # Per-fold breakdown: 2 folds, each macro-F1 = 1.0.
    assert len(result.per_fold) == 2
    for fold in result.per_fold:
        assert fold.breakdown.macro_f1 == pytest.approx(1.0)
        # Coverage must lie in [0, 1] when softmax_quantile is finite.
        assert math.isfinite(fold.softmax_quantile)
        assert 0.0 <= fold.coverage <= 1.0
        assert 1.0 <= fold.avg_set_size <= 3.0


def test_multi_run_logit_average_known_analytical_result(tmp_path: Path) -> None:
    """Two runs whose class-1 row argmaxes only emerge after averaging.

    Run A predicts [0, 0, 0] with scores ``[[0.45,0.4,0.15], ...]`` —
    class 0 narrowly wins for every row. Run B predicts [1, 1, 1]
    with scores ``[[0.3,0.65,0.05], ...]`` — class 1 wins for every
    row. The geometric mean of class-1 probabilities beats class-0's
    on every row (``sqrt(0.4*0.65) ≈ 0.51`` vs ``sqrt(0.45*0.3) ≈
    0.367``), so the ensemble argmax should be 1 on every row.

    Targets are [1, 1, 1] so the ensemble macro-F1 = 1.0 even though
    neither component scores above 0.0.
    """

    layout = [("wf_fold_1", 11)]
    targets = [[1, 1, 1]]
    preds_run_a = [[0, 0, 0]]
    preds_run_b = [[1, 1, 1]]
    scores_run_a = [[[0.45, 0.4, 0.15], [0.46, 0.42, 0.12], [0.5, 0.4, 0.1]]]
    scores_run_b = [[[0.3, 0.65, 0.05], [0.25, 0.7, 0.05], [0.2, 0.7, 0.1]]]
    run_a = _run_trials(
        fold_seed_layout=layout,
        per_trial_preds=preds_run_a,
        per_trial_targets=targets,
        per_trial_scores=scores_run_a,
    )
    run_b = _run_trials(
        fold_seed_layout=layout,
        per_trial_preds=preds_run_b,
        per_trial_targets=targets,
        per_trial_scores=scores_run_b,
    )
    path_a = _write_run_blob(tmp_path / "run_a" / "forecaster_sweep_results.json", run_a)
    path_b = _write_run_blob(tmp_path / "run_b" / "forecaster_sweep_results.json", run_b)
    specs = [
        RunSpec(
            run_id="run_a",
            architecture="lstm",
            encoder_alias="none",
            seed=11,
            results_path=str(path_a),
        ),
        RunSpec(
            run_id="run_b",
            architecture="tft",
            encoder_alias="none",
            seed=11,
            results_path=str(path_b),
        ),
    ]
    result = aggregate_multi_run_ensemble(specs, conformal_alpha=0.2)
    assert result.pooled_breakdown.macro_f1 == pytest.approx(1.0)
    # The two runs disagree on every row, so they must NOT be dropped
    # by the redundancy guard.
    assert set(result.kept_run_ids) == {"run_a", "run_b"}


def test_cohen_kappa_is_one_for_identical_predictions() -> None:
    """Two perfectly-agreeing components score kappa = 1.0."""

    preds = [0, 1, 2, 1, 0, 2, 1, 0, 2]
    assert _cohen_kappa(preds, preds) == pytest.approx(1.0)


def test_cohen_kappa_lt_one_for_disagreeing_pair() -> None:
    """A disagreeing pair scores kappa < 1.0."""

    preds_a = [0, 1, 2, 0, 1, 2]
    preds_b = [0, 0, 2, 1, 1, 2]
    kappa = _cohen_kappa(preds_a, preds_b)
    assert kappa < 1.0
    assert kappa > 0.0  # still positive — some agreement remains


def test_agreement_matrix_keys_carry_kappa_for_perfectly_agreeing_pair(
    tmp_path: Path,
) -> None:
    """Two runs whose predictions match row-for-row report kappa=1.0
    in the agreement matrix; the redundancy guard then drops one of
    them."""

    layout = [("wf_fold_1", 11)]
    targets = [[0, 1, 2]]
    same_preds = [[0, 1, 2]]
    same_scores = [[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]]
    run_a = _run_trials(
        fold_seed_layout=layout,
        per_trial_preds=same_preds,
        per_trial_targets=targets,
        per_trial_scores=same_scores,
    )
    run_b = _run_trials(
        fold_seed_layout=layout,
        per_trial_preds=same_preds,
        per_trial_targets=targets,
        per_trial_scores=same_scores,
    )
    path_a = _write_run_blob(tmp_path / "a" / "forecaster_sweep_results.json", run_a)
    path_b = _write_run_blob(tmp_path / "b" / "forecaster_sweep_results.json", run_b)
    specs = [
        RunSpec(
            run_id="run_a",
            architecture="lstm",
            encoder_alias="none",
            seed=11,
            results_path=str(path_a),
        ),
        RunSpec(
            run_id="run_b",
            architecture="lstm",
            encoder_alias="none",
            seed=29,
            results_path=str(path_b),
        ),
    ]
    result = aggregate_multi_run_ensemble(
        specs,
        conformal_alpha=0.2,
        redundancy_kappa_threshold=DEFAULT_REDUNDANCY_KAPPA_THRESHOLD,
    )
    assert result.agreement[("run_a", "run_b")] == pytest.approx(1.0)
    # Redundancy guard: one of the two identical runs must be dropped.
    assert len(result.kept_run_ids) == 1
    assert len(result.dropped_run_ids) == 1
    dropped_run, redundant_with, kappa = result.dropped_run_ids[0]
    assert dropped_run == "run_b"
    assert redundant_with == "run_a"
    assert kappa == pytest.approx(1.0)


def test_redundancy_guard_drops_one_of_three_identical_components(
    tmp_path: Path,
) -> None:
    """Three components where runs 1 and 2 are identical; run 3 disagrees.

    The redundancy guard must keep run_a (first), drop run_b (identical
    to run_a), and keep run_c (disagrees). 2 kept, 1 dropped."""

    layout = [("wf_fold_1", 11)]
    targets = [[0, 1, 2]]
    same_preds = [[0, 1, 2]]
    same_scores = [[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]]
    diff_preds = [[2, 0, 1]]
    diff_scores = [[[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]]
    run_a = _run_trials(
        fold_seed_layout=layout,
        per_trial_preds=same_preds,
        per_trial_targets=targets,
        per_trial_scores=same_scores,
    )
    run_b = _run_trials(
        fold_seed_layout=layout,
        per_trial_preds=same_preds,
        per_trial_targets=targets,
        per_trial_scores=same_scores,
    )
    run_c = _run_trials(
        fold_seed_layout=layout,
        per_trial_preds=diff_preds,
        per_trial_targets=targets,
        per_trial_scores=diff_scores,
    )
    path_a = _write_run_blob(tmp_path / "a" / "forecaster_sweep_results.json", run_a)
    path_b = _write_run_blob(tmp_path / "b" / "forecaster_sweep_results.json", run_b)
    path_c = _write_run_blob(tmp_path / "c" / "forecaster_sweep_results.json", run_c)
    specs = [
        RunSpec(
            run_id="run_a",
            architecture="lstm",
            encoder_alias="none",
            seed=11,
            results_path=str(path_a),
        ),
        RunSpec(
            run_id="run_b",
            architecture="lstm",
            encoder_alias="none",
            seed=29,
            results_path=str(path_b),
        ),
        RunSpec(
            run_id="run_c",
            architecture="tft",
            encoder_alias="none",
            seed=11,
            results_path=str(path_c),
        ),
    ]
    result = aggregate_multi_run_ensemble(specs, conformal_alpha=0.2)
    assert set(result.kept_run_ids) == {"run_a", "run_c"}
    assert len(result.dropped_run_ids) == 1
    assert result.dropped_run_ids[0][0] == "run_b"


def test_redundancy_guard_threshold_is_configurable(tmp_path: Path) -> None:
    """A pair with kappa around 0.5 should survive the default guard
    (threshold = 0.85) but drop under a stricter threshold of 0.4."""

    layout = [("wf_fold_1", 11)]
    targets = [[0, 0, 0, 1, 1, 1, 2, 2, 2]]
    preds_a = [[0, 0, 0, 1, 1, 1, 2, 2, 2]]
    preds_b = [[0, 0, 1, 1, 2, 1, 2, 0, 2]]  # ~ 0.5 agreement
    scores = [[[0.8, 0.1, 0.1]] * 9]
    run_a = _run_trials(
        fold_seed_layout=layout,
        per_trial_preds=preds_a,
        per_trial_targets=targets,
        per_trial_scores=scores,
    )
    run_b = _run_trials(
        fold_seed_layout=layout,
        per_trial_preds=preds_b,
        per_trial_targets=targets,
        per_trial_scores=scores,
    )
    path_a = _write_run_blob(tmp_path / "a" / "forecaster_sweep_results.json", run_a)
    path_b = _write_run_blob(tmp_path / "b" / "forecaster_sweep_results.json", run_b)
    specs = [
        RunSpec(
            run_id="run_a",
            architecture="lstm",
            encoder_alias="none",
            seed=11,
            results_path=str(path_a),
        ),
        RunSpec(
            run_id="run_b",
            architecture="tft",
            encoder_alias="none",
            seed=11,
            results_path=str(path_b),
        ),
    ]
    # Default 0.85 threshold should keep both.
    keep_both = aggregate_multi_run_ensemble(specs, conformal_alpha=0.2)
    assert set(keep_both.kept_run_ids) == {"run_a", "run_b"}
    # Stricter 0.4 threshold should drop one (kappa ~0.5 > 0.4).
    drop_one = aggregate_multi_run_ensemble(
        specs, conformal_alpha=0.2, redundancy_kappa_threshold=0.4
    )
    assert len(drop_one.kept_run_ids) == 1


def test_fold_layout_mismatch_raises(tmp_path: Path) -> None:
    """Two runs with different (fold, seed) trial sets must raise."""

    run_a_layout = [("wf_fold_1", 11), ("wf_fold_2", 11)]
    run_b_layout = [("wf_fold_1", 11), ("wf_fold_3", 11)]
    targets = [[0, 1, 2], [0, 1, 2]]
    preds = [[0, 1, 2], [0, 1, 2]]
    scores = [
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
    ]
    run_a = _run_trials(
        fold_seed_layout=run_a_layout,
        per_trial_preds=preds,
        per_trial_targets=targets,
        per_trial_scores=scores,
    )
    run_b = _run_trials(
        fold_seed_layout=run_b_layout,
        per_trial_preds=preds,
        per_trial_targets=targets,
        per_trial_scores=scores,
    )
    path_a = _write_run_blob(tmp_path / "a" / "forecaster_sweep_results.json", run_a)
    path_b = _write_run_blob(tmp_path / "b" / "forecaster_sweep_results.json", run_b)
    specs = [
        RunSpec(
            run_id="run_a",
            architecture="lstm",
            encoder_alias="none",
            seed=11,
            results_path=str(path_a),
        ),
        RunSpec(
            run_id="run_b",
            architecture="tft",
            encoder_alias="none",
            seed=11,
            results_path=str(path_b),
        ),
    ]
    with pytest.raises(ValueError, match="layout mismatch"):
        aggregate_multi_run_ensemble(specs, conformal_alpha=0.2)


def test_multi_run_per_fold_conformal_coverage_in_expected_range(
    tmp_path: Path,
) -> None:
    """Per-fold conformal coverage at alpha=0.2 should hit ~0.8 + finite-
    sample correction. The split-conformal helper bumps the rank to
    ``ceil(0.8 * (n+1))`` so empirical coverage on the calibration
    fold is ``rank/n``."""

    layout = [("wf_fold_1", 11), ("wf_fold_2", 11)]
    # 10 trials per fold; pick obvious-class confidence and targets that
    # match argmax so the conformal threshold settles around 1 - max_p.
    n_rows = 10
    targets = [[i % 3 for i in range(n_rows)]] * 2
    preds = [[i % 3 for i in range(n_rows)]] * 2
    scores = [
        [
            [0.7, 0.2, 0.1] if (i % 3) == 0
            else ([0.2, 0.7, 0.1] if (i % 3) == 1 else [0.1, 0.2, 0.7])
            for i in range(n_rows)
        ]
    ] * 2
    run_a = _run_trials(
        fold_seed_layout=layout,
        per_trial_preds=preds,
        per_trial_targets=targets,
        per_trial_scores=scores,
    )
    run_b = _run_trials(
        fold_seed_layout=layout,
        per_trial_preds=preds,
        per_trial_targets=targets,
        per_trial_scores=scores,
    )
    path_a = _write_run_blob(tmp_path / "a" / "forecaster_sweep_results.json", run_a)
    path_b = _write_run_blob(tmp_path / "b" / "forecaster_sweep_results.json", run_b)
    specs = [
        RunSpec(
            run_id="run_a",
            architecture="lstm",
            encoder_alias="none",
            seed=11,
            results_path=str(path_a),
        ),
        RunSpec(
            run_id="run_b",
            architecture="tft",
            encoder_alias="none",
            seed=11,
            results_path=str(path_b),
        ),
    ]
    # These two runs are identical -> redundancy guard drops one. To
    # exercise both runs survived, raise the threshold above 1.0.
    result = aggregate_multi_run_ensemble(
        specs,
        conformal_alpha=0.2,
        redundancy_kappa_threshold=1.5,
    )
    # Both runs kept (redundancy threshold > 1).
    assert set(result.kept_run_ids) == {"run_a", "run_b"}
    # Coverage on the calibration fold should be >= 0.8 (the nominal
    # coverage). The empirical coverage on the calibration set is
    # ``rank/n = ceil(0.8 * (n+1)) / n`` which is >= 0.8 for any
    # finite n.
    for fold in result.per_fold:
        assert fold.coverage >= 0.8 - 1e-9
        assert fold.coverage <= 1.0 + 1e-9
