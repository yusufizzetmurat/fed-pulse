"""Tests for the shuffled-targets memorisation control on the forecaster.

The shuffled-targets control runs the model against a deterministic
permutation of the target column. macro-RMSE on the shuffled-targets
run should sit near the constant-mean predictor; a real-targets run
whose RMSE is close to its shuffled counterpart is memorising rather
than learning the input-target mapping.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from app.models.config import FeatureVector, ModelConfig
from app.training.loop import train_model


def _synthetic_vectors(n: int) -> list[FeatureVector]:
    """Build a single small sequence the train loop can window over."""

    out: list[FeatureVector] = []
    for i in range(n):
        sentiment = math.sin(i * 0.41) * 0.6 + 0.1
        close = 4000.0 + 12.0 * i + 5.0 * math.sin(i * 0.3)
        vol = 0.012 + 0.003 * math.sin(i * 0.7 + 1.1)
        prev_close = (
            4000.0 + 12.0 * (i - 1) + 5.0 * math.sin((i - 1) * 0.3) if i else close
        )
        prev_vol = (
            0.012 + 0.003 * math.sin((i - 1) * 0.7 + 1.1) if i else vol
        )
        out.append(
            FeatureVector.from_market_state(
                date=f"2024-01-{i + 1:02d}",
                sentiment_score=sentiment,
                market_close=close,
                market_volatility=vol,
                previous_close=prev_close,
                previous_volatility=prev_vol,
                elapsed_time=float(i % 30),
            )
        )
    return out


_MODEL_CONFIG = ModelConfig(
    input_size=6,
    hidden_size=8,
    num_layers=1,
    dropout=0.0,
    head_hidden_size=8,
)


def _train_summary(
    *,
    vectors: list[FeatureVector],
    seed: int,
    shuffle_targets_control: bool,
    checkpoint: Path,
):
    result = train_model(
        vectors=vectors,
        epochs=4,
        batch_size=8,
        learning_rate=1e-3,
        validation_fraction=0.25,
        early_stopping_patience=2,
        save_checkpoint=False,
        checkpoint_path=checkpoint,
        device="cpu",
        seed=seed,
        model_config=_MODEL_CONFIG,
        shuffle_targets_control=shuffle_targets_control,
    )
    return result.summary


def test_shuffled_targets_seed_determinism(tmp_path: Path) -> None:
    """Same seed produces the same permutation -> same metrics."""

    vectors = _synthetic_vectors(28)
    first = _train_summary(
        vectors=vectors,
        seed=11,
        shuffle_targets_control=True,
        checkpoint=tmp_path / "first.pt",
    )
    second = _train_summary(
        vectors=vectors,
        seed=11,
        shuffle_targets_control=True,
        checkpoint=tmp_path / "second.pt",
    )
    assert first.metrics is not None and second.metrics is not None
    assert first.metrics.combined_rmse == pytest.approx(
        second.metrics.combined_rmse, abs=1e-8
    )
    assert first.target_mode == "shuffled"
    assert second.target_mode == "shuffled"


def test_shuffled_targets_independent_per_fold(tmp_path: Path) -> None:
    """Different seeds produce different shuffles -> different metrics.

    The shuffle generator is seeded by the run seed, so two folds with
    different seeds see different target permutations and the trained
    model converges to different validation RMSEs.
    """

    vectors = _synthetic_vectors(28)
    seed_11 = _train_summary(
        vectors=vectors,
        seed=11,
        shuffle_targets_control=True,
        checkpoint=tmp_path / "s11.pt",
    )
    seed_29 = _train_summary(
        vectors=vectors,
        seed=29,
        shuffle_targets_control=True,
        checkpoint=tmp_path / "s29.pt",
    )
    assert seed_11.metrics is not None and seed_29.metrics is not None
    assert seed_11.metrics.combined_rmse != pytest.approx(
        seed_29.metrics.combined_rmse, abs=1e-6
    )


def test_shuffled_targets_marks_summary(tmp_path: Path) -> None:
    """The summary's ``target_mode`` flips to ``shuffled`` on the control."""

    vectors = _synthetic_vectors(28)
    real_summary = _train_summary(
        vectors=vectors,
        seed=11,
        shuffle_targets_control=False,
        checkpoint=tmp_path / "real.pt",
    )
    shuffled_summary = _train_summary(
        vectors=vectors,
        seed=11,
        shuffle_targets_control=True,
        checkpoint=tmp_path / "shuffled.pt",
    )
    assert real_summary.target_mode == "real"
    assert shuffled_summary.target_mode == "shuffled"


def test_summary_carries_train_metrics(tmp_path: Path) -> None:
    """``TrainingRunSummary`` exposes train_metrics for the gap derivative."""

    vectors = _synthetic_vectors(28)
    summary = _train_summary(
        vectors=vectors,
        seed=11,
        shuffle_targets_control=False,
        checkpoint=tmp_path / "real.pt",
    )
    assert summary.train_metrics is not None
    assert math.isfinite(summary.train_metrics.combined_rmse)
    assert summary.metrics is not None
    assert math.isfinite(summary.metrics.combined_rmse)
