from __future__ import annotations

import argparse

import pytest

pytest.importorskip("torch")

from app.services.forecaster import EvaluationMetrics, ModelConfig, TrainingRunSummary
from app.train_forecaster import build_sweep_candidates, select_best_summary


def _summary(
    *,
    hidden_size: int,
    learning_rate: float,
    combined_rmse: float,
    loss: float,
) -> TrainingRunSummary:
    return TrainingRunSummary(
        model_config=ModelConfig(hidden_size=hidden_size, num_layers=2, dropout=0.15, head_hidden_size=32),
        device="cpu",
        epochs_requested=20,
        epochs_completed=12,
        batch_size=16,
        learning_rate=learning_rate,
        validation_split=0.2,
        early_stopping_patience=4,
        sequence_groups=1,
        total_windows=24,
        train_windows=19,
        validation_windows=5,
        checkpoint_path="backend/models/forecaster_best.pt",
        checkpoint_saved=False,
        best_epoch=9,
        metrics=EvaluationMetrics(
            loss=loss,
            close_rmse=combined_rmse,
            volatility_rmse=combined_rmse / 2,
            combined_rmse=combined_rmse,
        ),
    )


def test_select_best_summary_prefers_lowest_combined_rmse():
    summaries = [
        _summary(hidden_size=32, learning_rate=1e-3, combined_rmse=0.22, loss=0.10),
        _summary(hidden_size=64, learning_rate=5e-4, combined_rmse=0.11, loss=0.12),
        _summary(hidden_size=96, learning_rate=2e-4, combined_rmse=0.14, loss=0.08),
    ]

    best = select_best_summary(summaries)

    assert best is not None
    assert best.model_config.hidden_size == 64
    assert best.metrics.combined_rmse == pytest.approx(0.11)


def test_build_sweep_candidates_creates_cartesian_product():
    args = argparse.Namespace(
        hidden_size=64,
        num_layers=2,
        dropout=0.15,
        learning_rate=1e-3,
        epochs=40,
        head_hidden_size=32,
        hidden_sizes=[32, 64],
        num_layers_grid=[1, 2],
        dropouts=[0.10],
        learning_rates=[1e-3, 5e-4],
        epochs_grid=[20],
    )

    candidates = build_sweep_candidates(args)

    assert len(candidates) == 8
    assert {candidate["model_config"].hidden_size for candidate in candidates} == {32, 64}
    assert {candidate["model_config"].num_layers for candidate in candidates} == {1, 2}
    assert {candidate["learning_rate"] for candidate in candidates} == {1e-3, 5e-4}
