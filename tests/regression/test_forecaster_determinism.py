from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from app.services.forecaster import FeatureVector, ModelConfig, train_model  # noqa: E402


def _synthetic_vectors(n: int) -> list[FeatureVector]:
    out: list[FeatureVector] = []
    for i in range(n):
        sentiment = math.sin(i * 0.41) * 0.6 + 0.1
        close = 4000.0 + 12.0 * i + 5.0 * math.sin(i * 0.3)
        vol = 0.012 + 0.003 * math.sin(i * 0.7 + 1.1)
        prev_close = 4000.0 + 12.0 * (i - 1) + 5.0 * math.sin((i - 1) * 0.3) if i else close
        prev_vol = 0.012 + 0.003 * math.sin((i - 1) * 0.7 + 1.1) if i else vol
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


def _train_loss(seed: int, vectors: list[FeatureVector], checkpoint: Path) -> float:
    result = train_model(
        vectors=vectors,
        epochs=4,
        batch_size=8,
        learning_rate=1e-3,
        validation_split=0.25,
        early_stopping_patience=2,
        save_checkpoint=False,
        checkpoint_path=checkpoint,
        device="cpu",
        seed=seed,
        model_config=ModelConfig(input_size=6, hidden_size=8, num_layers=1, dropout=0.0, head_hidden_size=8),
    )
    metrics = result.summary.metrics
    assert metrics is not None
    return float(metrics.combined_rmse)


def test_train_model_is_bit_identical_under_same_seed(tmp_path: Path) -> None:
    vectors = _synthetic_vectors(28)
    first = _train_loss(seed=11, vectors=vectors, checkpoint=tmp_path / "first.pt")
    second = _train_loss(seed=11, vectors=vectors, checkpoint=tmp_path / "second.pt")
    assert first == pytest.approx(second, abs=1e-8), (
        f"determinism regression: seed=11 produced two different combined RMSE values "
        f"({first} vs {second}); the seed protocol leaks somewhere"
    )


def test_train_model_diverges_under_different_seeds(tmp_path: Path) -> None:
    vectors = _synthetic_vectors(28)
    seed_11 = _train_loss(seed=11, vectors=vectors, checkpoint=tmp_path / "s11.pt")
    seed_29 = _train_loss(seed=29, vectors=vectors, checkpoint=tmp_path / "s29.pt")
    assert seed_11 != pytest.approx(seed_29, abs=1e-6), (
        "different seeds produced identical RMSE; randomness is collapsed somewhere"
    )
