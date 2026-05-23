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
        architecture="lstm",
        architectures=None,
        seed=None,
        seeds=None,
        credibility_features=False,
    )

    candidates = build_sweep_candidates(args)

    assert len(candidates) == 8
    assert {candidate["model_config"].hidden_size for candidate in candidates} == {32, 64}
    assert {candidate["model_config"].num_layers for candidate in candidates} == {1, 2}
    assert {candidate["learning_rate"] for candidate in candidates} == {1e-3, 5e-4}
    # When the caller doesn't pass `--architectures`, the sweep stays on the
    # single architecture so default behaviour (LSTM-only) is preserved.
    assert {candidate["model_config"].architecture for candidate in candidates} == {"lstm"}
    assert {candidate["seed"] for candidate in candidates} == {None}


# ---------------------------------------------------------------------------
# Walk-forward CLI tests
#
# The new --folds / --protocol flags drive the candidate enumeration
# to expand per-fold trials and route the trainer through the
# walk-forward partition path.
# ---------------------------------------------------------------------------


def _walk_forward_args(
    *,
    folds: list[str] | None,
    architectures: list[str] | None = None,
    seeds: list[int] | None = None,
) -> argparse.Namespace:
    """Minimal argparse namespace for the per-fold candidate tests."""

    return argparse.Namespace(
        hidden_size=32,
        num_layers=1,
        dropout=0.1,
        learning_rate=1e-3,
        epochs=4,
        head_hidden_size=16,
        hidden_sizes=None,
        num_layers_grid=None,
        dropouts=None,
        learning_rates=None,
        epochs_grid=None,
        weight_decay=1e-4,
        weight_decays=None,
        text_adapter_dim=64,
        text_adapter_dims=None,
        text_encoder="none",
        use_text_embeddings=True,
        training_package_id=None,
        rich_features=False,
        architecture="lstm",
        architectures=architectures or ["lstm"],
        seed=None,
        seeds=seeds or [11],
        credibility_features=False,
        random_search=False,
        random_search_samples=50,
        random_search_seed=42,
        folds=folds,
    )


def test_build_sweep_candidates_expands_per_fold():
    """When --folds is set the candidate count multiplies by the fold count."""

    no_folds = build_sweep_candidates(_walk_forward_args(folds=None))
    four_folds = build_sweep_candidates(
        _walk_forward_args(folds=["wf_fold_1", "wf_fold_2", "wf_fold_3", "wf_fold_4"])
    )
    assert len(four_folds) == len(no_folds) * 4
    fold_ids = sorted({c.get("fold_id") for c in four_folds if c.get("fold_id")})
    assert fold_ids == ["wf_fold_1", "wf_fold_2", "wf_fold_3", "wf_fold_4"]


def test_build_sweep_candidates_no_fold_id_when_unset():
    """The exhaustive single-fold path never tags candidates with fold_id."""

    candidates = build_sweep_candidates(_walk_forward_args(folds=None))
    for record in candidates:
        assert "fold_id" not in record


def test_train_model_with_pre_split_skips_internal_validation_split():
    """Walk-forward path consumes pre-split train/val/test lists directly."""

    import math

    from app.services.forecaster import FeatureVector
    from app.training.loop import train_model
    from app.training import loop as loop_module

    def _vectors(n: int, *, base: float) -> list[FeatureVector]:
        out: list[FeatureVector] = []
        for i in range(n):
            close = base + 12.0 * i + 5.0 * math.sin(i * 0.3)
            vol = 0.012 + 0.003 * math.sin(i * 0.7 + 1.1)
            prev_close = base + 12.0 * (i - 1) + 5.0 * math.sin((i - 1) * 0.3) if i else close
            prev_vol = 0.012 + 0.003 * math.sin((i - 1) * 0.7 + 1.1) if i else vol
            out.append(
                FeatureVector.from_market_state(
                    date=f"2024-01-{i + 1:02d}",
                    sentiment_score=math.sin(i * 0.41) * 0.6 + 0.1,
                    market_close=close,
                    market_volatility=vol,
                    previous_close=prev_close,
                    previous_volatility=prev_vol,
                    elapsed_time=float(i % 30),
                )
            )
        return out

    # The legacy ``_split_train_validation`` MUST NOT run on the
    # walk-forward path; intercept it and fail the test if it does.
    original_split = loop_module._split_train_validation

    def _exploding_split(*args, **kwargs):
        raise AssertionError(
            "walk-forward path must not call the legacy 80/20 internal split"
        )

    loop_module._split_train_validation = _exploding_split
    try:
        result = train_model(
            train_sequence_groups=[_vectors(22, base=4400.0)],
            val_sequence_groups=[_vectors(22, base=4500.0)],
            test_sequence_groups=[_vectors(22, base=4600.0)],
            fold_id="wf_fold_1",
            protocol="walk-forward",
            epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            validation_fraction=0.25,
            early_stopping_patience=2,
            save_checkpoint=False,
            device="cpu",
            seed=11,
        )
    finally:
        loop_module._split_train_validation = original_split

    summary = result.summary
    # Walk-forward summary carries explicit train/val/test metrics.
    assert summary.fold_id == "wf_fold_1"
    assert summary.protocol == "walk-forward"
    assert summary.train_metrics is not None
    assert summary.val_metrics is not None
    assert summary.test_metrics is not None
    # The headline ``metrics`` slot maps to the held-out test_metrics
    # on the walk-forward path so downstream selection ranks on the
    # real test number.
    assert summary.metrics is summary.test_metrics


def test_test_train_gap_flag_threshold_uses_held_out():
    """``gap_flag = high`` when test_train_gap exceeds 0.5."""

    from app.evaluation.forecaster_sweep_aggregator import _build_rows

    by_arch = {
        "lstm": {
            "architecture": "lstm",
            "target_mode": "real",
            "fold_id": None,
            "protocol": "single-fold",
            "seeds": [11],
            "combined_rmse": [0.40],
            "close_rmse": [0.30],
            "volatility_rmse": [0.10],
            "train_combined_rmse": [0.10],
            "val_combined_rmse": [0.35],
            "test_combined_rmse": [0.40],
            "credibility_features": False,
        }
    }
    rows = _build_rows(by_arch, block_size=1, n_resamples=100, coverage=0.95, seed=11)
    assert len(rows) == 1
    row = rows[0]
    # test_train_gap = (0.40 - 0.10) / 0.10 = 3.0; gap_flag should fire.
    assert row.test_train_gap == pytest.approx(3.0)
    assert row.gap_flag == "high"
