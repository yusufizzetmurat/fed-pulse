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
    fold_id: str | None = None,
) -> dict:
    if close_rmse is None:
        close_rmse = combined_rmse * 0.8
    if volatility_rmse is None:
        volatility_rmse = combined_rmse * 0.2
    record: dict = {
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
    if fold_id is not None:
        record["fold_id"] = fold_id
    return record


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


def test_aggregator_emits_per_fold_and_all_folds_rows(tmp_path: Path) -> None:
    """Per-architecture-per-encoder rows are emitted once per fold plus
    one aggregated row across all folds. The per-fold rows precede
    the ``all-folds`` aggregate for each architecture in the markdown."""

    trials: list[dict] = []
    for fold_idx in range(1, 5):
        # Two seeds per fold per architecture so the bootstrap has a
        # non-degenerate sample size.
        for seed_value in (11, 29):
            trials.append(
                _trial(
                    architecture="lstm",
                    seed=seed_value,
                    combined_rmse=0.10 + 0.01 * fold_idx,
                    fold_id=f"wf_fold_{fold_idx}",
                )
            )
            trials.append(
                _trial(
                    architecture="gru",
                    seed=seed_value,
                    combined_rmse=0.20 + 0.01 * fold_idx,
                    fold_id=f"wf_fold_{fold_idx}",
                )
            )
    _write_report(tmp_path / "sweep_results.json", trials)

    rows, markdown, _ = aggregate(tmp_path, seed=11)

    # Two architectures x (4 per-fold rows + 1 all-folds row) = 10 rows.
    assert len(rows) == 10

    # Within each architecture, four per-fold rows precede the
    # all-folds aggregate. Group rows by (architecture, target_mode)
    # in emission order.
    grouped: dict[str, list[ArchitectureRow]] = {}
    for row in rows:
        grouped.setdefault(row.architecture, []).append(row)
    assert set(grouped.keys()) == {"lstm", "gru"}
    for arch, rows_for_arch in grouped.items():
        folds_seen = [r.fold for r in rows_for_arch]
        assert folds_seen == [
            "wf_fold_1",
            "wf_fold_2",
            "wf_fold_3",
            "wf_fold_4",
            "all-folds",
        ], f"architecture={arch} fold order broke: {folds_seen}"

    # The all-folds aggregate spans every cell -- 2 seeds x 4 folds = 8 values.
    all_folds_rows = [row for row in rows if row.fold == "all-folds"]
    assert all_folds_rows, "all-folds aggregate row missing"
    for row in all_folds_rows:
        assert len(row.combined_rmse_values) == 8

    # Per-fold rows aggregate over the per-fold seed pool (n=2 here).
    per_fold_rows = [row for row in rows if row.fold != "all-folds"]
    for row in per_fold_rows:
        assert len(row.combined_rmse_values) == 2

    # Markdown carries the fold column and renders the per-fold rows
    # under each architecture before the all-folds line.
    assert "| fold |" in markdown
    assert "| wf_fold_1 |" in markdown
    assert "| all-folds |" in markdown


def test_aggregator_single_fold_path_preserves_pre_pr_contract(tmp_path: Path) -> None:
    """Trials without a fold_id collapse into one ``all-folds`` row per architecture.

    The pre-PR sweep contract emits exactly one row per architecture;
    pinning the row count + fold label here keeps that contract intact
    for callers running the legacy single-fold path.
    """

    trials = [
        _trial(architecture="lstm", seed=11, combined_rmse=0.10),
        _trial(architecture="lstm", seed=29, combined_rmse=0.11),
        _trial(architecture="gru", seed=11, combined_rmse=0.20),
        _trial(architecture="gru", seed=29, combined_rmse=0.21),
    ]
    _write_report(tmp_path / "sweep_results.json", trials)

    rows, _, _ = aggregate(tmp_path, seed=11)

    # One row per architecture, tagged all-folds.
    assert {row.architecture for row in rows} == {"lstm", "gru"}
    assert len(rows) == 2
    for row in rows:
        assert row.fold == "all-folds"
