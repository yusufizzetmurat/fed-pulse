"""Cover ``app.evaluation.forecaster_sweep_aggregator``.

Three slices:

- CI bands at synthetic inputs match the block-bootstrap protocol (lo <= point <= hi).
- Rank ordering follows ascending combined-RMSE.
- The aggregator is deterministic at a fixed bootstrap seed -- the same
  input deck produces a byte-stable markdown table on repeat calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation.forecaster_sweep_aggregator import (
    ArchitectureRow,
    aggregate,
    render_markdown,
)


def _trial(
    *,
    architecture: str,
    seed: int,
    combined_rmse: float,
    close_rmse: float | None = None,
    volatility_rmse: float | None = None,
    credibility_features: bool = False,
) -> dict:
    if close_rmse is None:
        close_rmse = combined_rmse * 0.8
    if volatility_rmse is None:
        volatility_rmse = combined_rmse * 0.2
    return {
        "trial_index": 0,
        "architecture": architecture,
        "seed": seed,
        "summary": {
            "model_config": {
                "architecture": architecture,
                "credibility_features": credibility_features,
            },
            "metrics": {
                "combined_rmse": combined_rmse,
                "close_rmse": close_rmse,
                "volatility_rmse": volatility_rmse,
                "loss": combined_rmse,
            },
        },
    }


def _write_report(path: Path, trials: list[dict]) -> Path:
    payload = {"mode": "sweep", "trials": trials}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_ci_bands_bracket_the_point_estimate(tmp_path: Path) -> None:
    trials = [
        _trial(architecture="lstm", seed=11, combined_rmse=0.10),
        _trial(architecture="lstm", seed=29, combined_rmse=0.12),
        _trial(architecture="lstm", seed=47, combined_rmse=0.11),
        _trial(architecture="lstm", seed=71, combined_rmse=0.13),
        _trial(architecture="lstm", seed=97, combined_rmse=0.115),
        _trial(architecture="gru", seed=11, combined_rmse=0.20),
        _trial(architecture="gru", seed=29, combined_rmse=0.21),
        _trial(architecture="gru", seed=47, combined_rmse=0.19),
        _trial(architecture="gru", seed=71, combined_rmse=0.22),
        _trial(architecture="gru", seed=97, combined_rmse=0.205),
    ]
    _write_report(tmp_path / "sweep_results.json", trials)

    rows, _, payload = aggregate(tmp_path, seed=11)
    # Two architectures, five seeds each
    assert {row.architecture for row in rows} == {"lstm", "gru"}
    for row in rows:
        assert len(row.combined_rmse_values) == 5
        assert row.combined_rmse_ci.lo <= row.combined_rmse_ci.point <= row.combined_rmse_ci.hi
        assert row.combined_rmse_ci.coverage == 0.95
    # LSTM should be ranked first (lower combined-RMSE).
    assert rows[0].architecture == "lstm"
    assert rows[1].architecture == "gru"
    # JSON payload mirrors the rows in order.
    assert [arch["architecture"] for arch in payload["architectures"]] == [
        "lstm",
        "gru",
    ]


def test_aggregator_handles_multiple_reports(tmp_path: Path) -> None:
    """Two separate sweep files in the same directory must be merged per architecture."""

    _write_report(
        tmp_path / "a_sweep_results.json",
        [_trial(architecture="lstm", seed=11, combined_rmse=0.10)],
    )
    _write_report(
        tmp_path / "b_sweep_results.json",
        [
            _trial(architecture="lstm", seed=29, combined_rmse=0.12),
            _trial(architecture="tcn", seed=11, combined_rmse=0.18),
        ],
    )

    rows, _, _ = aggregate(tmp_path, seed=11)
    by_arch = {row.architecture: row for row in rows}
    assert sorted(by_arch["lstm"].seeds) == [11, 29]
    assert by_arch["tcn"].seeds == [11]


def test_markdown_table_is_deterministic(tmp_path: Path) -> None:
    """Same inputs + same bootstrap seed must yield the same markdown."""

    trials = [
        _trial(architecture="lstm", seed=11, combined_rmse=0.10),
        _trial(architecture="lstm", seed=29, combined_rmse=0.11),
        _trial(architecture="gru", seed=11, combined_rmse=0.20),
        _trial(architecture="gru", seed=29, combined_rmse=0.205),
    ]
    _write_report(tmp_path / "sweep_results.json", trials)

    _, md_first, _ = aggregate(tmp_path, seed=11)
    _, md_second, _ = aggregate(tmp_path, seed=11)
    assert md_first == md_second


def test_render_markdown_handles_empty_rows() -> None:
    assert "no forecaster sweep rows" in render_markdown([], coverage=0.95)


def test_credibility_flag_surfaces_in_markdown(tmp_path: Path) -> None:
    """When the trials are credibility-on the headline must reflect that label."""

    trials = [
        _trial(
            architecture="lstm",
            seed=11,
            combined_rmse=0.10,
            credibility_features=True,
        ),
        _trial(
            architecture="lstm",
            seed=29,
            combined_rmse=0.11,
            credibility_features=True,
        ),
    ]
    _write_report(tmp_path / "sweep_results.json", trials)

    rows, markdown, _ = aggregate(tmp_path, seed=11)
    assert rows[0].credibility_features is True
    assert "| on |" in markdown


def test_missing_artifact_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        aggregate(tmp_path / "does_not_exist", seed=11)


def test_directory_without_reports_raises(tmp_path: Path) -> None:
    (tmp_path / "unrelated.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        aggregate(tmp_path, seed=11)


def test_row_dataclass_is_frozen() -> None:
    row = ArchitectureRow(
        architecture="lstm",
        seeds=[11],
        credibility_features=False,
        combined_rmse_values=[0.1],
        close_rmse_values=[0.08],
        volatility_rmse_values=[0.02],
        combined_rmse_ci=None,  # type: ignore[arg-type]
        close_rmse_ci=None,  # type: ignore[arg-type]
        volatility_rmse_ci=None,  # type: ignore[arg-type]
    )
    with pytest.raises(Exception):
        row.architecture = "gru"  # type: ignore[misc]


def _fold_trial(
    *,
    architecture: str,
    seed: int,
    fold_id: str,
    test_rmse: float,
    train_rmse: float,
    val_rmse: float | None = None,
) -> dict:
    if val_rmse is None:
        val_rmse = test_rmse
    return {
        "trial_index": 0,
        "architecture": architecture,
        "seed": seed,
        "fold_id": fold_id,
        "summary": {
            "model_config": {
                "architecture": architecture,
                "credibility_features": False,
            },
            "metrics": {
                "combined_rmse": test_rmse,
                "close_rmse": test_rmse * 0.8,
                "volatility_rmse": test_rmse * 0.2,
                "loss": test_rmse,
            },
            "train_metrics": {
                "combined_rmse": train_rmse,
                "close_rmse": train_rmse * 0.8,
                "volatility_rmse": train_rmse * 0.2,
                "loss": train_rmse,
            },
            "val_metrics": {
                "combined_rmse": val_rmse,
                "close_rmse": val_rmse * 0.8,
                "volatility_rmse": val_rmse * 0.2,
                "loss": val_rmse,
            },
            "test_metrics": {
                "combined_rmse": test_rmse,
                "close_rmse": test_rmse * 0.8,
                "volatility_rmse": test_rmse * 0.2,
                "loss": test_rmse,
            },
            "fold_id": fold_id,
            "protocol": "walk-forward",
        },
    }


def test_aggregator_emits_test_rmse_column_in_markdown(tmp_path: Path) -> None:
    """The markdown headline reflects the new test-RMSE label."""

    trials = [
        _trial(architecture="lstm", seed=11, combined_rmse=0.10),
        _trial(architecture="lstm", seed=29, combined_rmse=0.11),
    ]
    _write_report(tmp_path / "sweep_results.json", trials)
    _, markdown, _ = aggregate(tmp_path, seed=11)
    assert "test-RMSE" in markdown
    # Legacy combined-RMSE headline must no longer appear on the
    # markdown table; the column rename was an explicit contract change.
    assert "combined-RMSE (mean" not in markdown


def test_aggregator_per_fold_plus_all_folds_rows(tmp_path: Path) -> None:
    """Walk-forward trials emit one row per (arch, fold) plus an aggregate row."""

    from app.evaluation.forecaster_sweep_aggregator import aggregate

    trials = []
    for fold_id, base_rmse in (
        ("wf_fold_1", 0.10),
        ("wf_fold_2", 0.11),
        ("wf_fold_3", 0.12),
        ("wf_fold_4", 0.13),
    ):
        for seed in (11, 29):
            trials.append(
                _fold_trial(
                    architecture="lstm",
                    seed=seed,
                    fold_id=fold_id,
                    test_rmse=base_rmse,
                    train_rmse=base_rmse * 0.8,
                )
            )
    _write_report(tmp_path / "wf_sweep_results.json", trials)

    rows, markdown, _ = aggregate(tmp_path, seed=11)
    fold_rows = [r for r in rows if r.fold_id and r.fold_id != "all-folds"]
    all_folds_rows = [r for r in rows if r.fold_id == "all-folds"]
    assert {r.fold_id for r in fold_rows} == {
        "wf_fold_1",
        "wf_fold_2",
        "wf_fold_3",
        "wf_fold_4",
    }
    # One all-folds aggregate row per architecture.
    assert len(all_folds_rows) == 1
    all_folds = all_folds_rows[0]
    assert all_folds.architecture == "lstm"
    # All-folds row collects every per-fold trial -> 4 folds x 2 seeds
    # = 8 cells in its test_rmse_values list.
    assert len(all_folds.test_rmse_values) == 8
    # All-folds row carries the bootstrap CI computed across the 8 cells.
    assert all_folds.test_rmse_ci is not None
    assert all_folds.test_rmse_ci.lo <= all_folds.test_rmse_ci.point <= all_folds.test_rmse_ci.hi
    # Markdown contains each fold id and the all-folds tag.
    for fold_id in ("wf_fold_1", "wf_fold_2", "wf_fold_3", "wf_fold_4", "all-folds"):
        assert fold_id in markdown
    # Protocol column reflects walk-forward.
    assert "walk-forward" in markdown


def test_aggregator_test_train_gap_uses_held_out_test_metric(tmp_path: Path) -> None:
    """``test_train_gap`` is (test_rmse - train_rmse) / train_rmse."""

    trials = [
        _fold_trial(
            architecture="lstm",
            seed=11,
            fold_id="wf_fold_1",
            test_rmse=0.40,
            train_rmse=0.10,
        ),
    ]
    _write_report(tmp_path / "wf_sweep_results.json", trials)
    rows, _, _ = aggregate(tmp_path, seed=11)
    # Per-fold row + all-folds row.
    fold_row = next(r for r in rows if r.fold_id == "wf_fold_1")
    assert fold_row.test_train_gap == pytest.approx((0.40 - 0.10) / 0.10)
    assert fold_row.gap_flag == "high"
