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
