"""Issue #181 — ``validation_split`` is the deprecated alias for the
canonical ``validation_fraction`` kwarg on ``train_model`` and
``bootstrap_checkpoint``. The CLI side already canonicalised on
``--validation-fraction``; this PR mirrors the rename at the function
boundary while keeping the old kwarg accepted for a deprecation window.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _synthetic_vectors(n: int):
    from app.services.forecaster import FeatureVector

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


def _train_with(fraction_kwarg: str, value: float, *, tmp_path: Path) -> float:
    from app.services.forecaster import ModelConfig, train_model

    vectors = _synthetic_vectors(28)
    kwargs: dict[str, float] = {fraction_kwarg: value}
    result = train_model(
        vectors=vectors,
        epochs=4,
        batch_size=8,
        learning_rate=1e-3,
        early_stopping_patience=2,
        save_checkpoint=False,
        checkpoint_path=tmp_path / "ck.pt",
        device="cpu",
        seed=11,
        model_config=ModelConfig(
            input_size=6, hidden_size=8, num_layers=1, dropout=0.0, head_hidden_size=8
        ),
        **kwargs,
    )
    metrics = result.summary.metrics
    assert metrics is not None
    return float(metrics.combined_rmse)


def test_canonical_kwarg_runs_without_warnings(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        rmse = _train_with("validation_fraction", 0.25, tmp_path=tmp_path)
    assert rmse > 0


def test_legacy_kwarg_still_runs_but_warns(tmp_path: Path) -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        rmse = _train_with("validation_split", 0.25, tmp_path=tmp_path)
    assert rmse > 0
    matched = [
        w for w in captured if "validation_split" in str(w.message) and issubclass(w.category, DeprecationWarning)
    ]
    assert matched, "expected DeprecationWarning when passing validation_split"


def test_canonical_and_legacy_produce_identical_rmse(tmp_path: Path) -> None:
    """The alias must be a pure rename: same numeric input through
    either kwarg produces a byte-identical training result."""

    canonical = _train_with("validation_fraction", 0.25, tmp_path=tmp_path / "a")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = _train_with("validation_split", 0.25, tmp_path=tmp_path / "b")
    assert canonical == pytest.approx(legacy, abs=1e-8)


def test_passing_both_kwargs_raises_typeerror(tmp_path: Path) -> None:
    """``train_model`` refuses ambiguous calls. Pass one or the other,
    not both."""

    from app.services.forecaster import ModelConfig, train_model

    vectors = _synthetic_vectors(28)
    with pytest.raises(TypeError, match="validation_fraction"):
        train_model(
            vectors=vectors,
            epochs=2,
            batch_size=8,
            learning_rate=1e-3,
            validation_fraction=0.2,
            validation_split=0.25,
            save_checkpoint=False,
            device="cpu",
            seed=11,
            model_config=ModelConfig(
                input_size=6, hidden_size=8, num_layers=1, dropout=0.0, head_hidden_size=8
            ),
        )


def test_bootstrap_checkpoint_accepts_canonical_kwarg(tmp_path: Path) -> None:
    from app.services.forecaster import ModelConfig
    from app.training.loop import bootstrap_checkpoint

    vectors = _synthetic_vectors(28)
    result = bootstrap_checkpoint(
        vectors=vectors,
        epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        validation_fraction=0.2,
        early_stopping_patience=2,
        checkpoint_path=tmp_path / "boot.pt",
    )
    assert result.summary.metrics is not None
