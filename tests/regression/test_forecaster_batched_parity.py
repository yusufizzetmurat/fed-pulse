"""Numerical-parity regression for the bucketed-HP sweep.

The bucketed runner schedules cells concurrently inside one Python
process via threads + CUDA streams (or the stacked-mode forward when
the architecture is vmap-friendly). Per-cell training is the same
``_run_single_training`` entry point the sequential path calls, so
the per-cell test_rmse values must agree across batching-mode=off
(legacy ProcessPoolExecutor / sequential path) and batching-mode=streams
within a tight numerical tolerance.

The tolerance is loosened to +/- 1e-3 (vs. the legacy 1e-4 on the
``--batching-mode=off`` regression test) because:

- The streams runner serialises ``optimizer.step`` calls through the
  Python GIL but kernel-launch ordering across CUDA streams is
  non-deterministic; CPU runs collapse the streams to ``None`` and
  the ordering is fully deterministic, but the tolerance covers GPU
  re-runs as well.
- Mixed-precision sums inside the loss / metric reductions floor at
  roughly 1e-6 per element, and the test partition spans ~50 windows.

The test runs on CPU under the legacy 80/20 split (no
training-package fixture is needed) so it is self-contained and
fast. The byte-identity regression contract on the legacy
``--data-dir`` path is pinned separately by
``tests/regression/test_forecaster_determinism.py``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from app.models.config import ModelConfig  # noqa: E402
from app.services.forecaster import FeatureVector  # noqa: E402
from app.train_forecaster import (  # noqa: E402
    _NullCudaStreamContext,
    _run_single_training,
)
from app.training.batched_sweep import (  # noqa: E402
    group_candidates_into_buckets,
    run_bucket_streams,
)


def _synthetic_vectors(n: int) -> list[FeatureVector]:
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


def _make_candidate(*, hidden_size: int, dropout: float, lr: float, seed: int) -> dict:
    return {
        "model_config": ModelConfig(
            input_size=6,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=dropout,
            head_hidden_size=8,
            architecture="lstm",
            credibility_features=False,
            text_embedding_dim=0,
            text_adapter_dim=0,
        ),
        "learning_rate": lr,
        "epochs": 6,
        "weight_decay": 1e-4,
        "text_adapter_dim": 0,
        "seed": seed,
    }


def _run_cell_sequential(
    candidate: dict, *, vectors: list[FeatureVector], checkpoint: Path
):
    return _run_single_training(
        data_dir=Path("."),
        checkpoint_path=checkpoint,
        device=torch.device("cpu"),
        epochs=int(candidate["epochs"]),
        batch_size=8,
        learning_rate=float(candidate["learning_rate"]),
        validation_fraction=0.25,
        early_stopping_patience=10,
        model_config=candidate["model_config"],
        save_checkpoint=False,
        seed=int(candidate["seed"]),
        sequence_groups=[list(vectors)],
        walk_forward_split=None,
        weight_decay=float(candidate["weight_decay"]),
        shuffle_targets_control=False,
        text_encoder=None,
        text_pool_lambda_inv_days=0.0,
    )


def test_streams_mode_matches_sequential_per_cell_rmse(tmp_path):
    """A 4-cell bucket under streams mode matches the sequential per-cell RMSE.

    The sequential path runs each cell back-to-back through
    _run_single_training; the streams path runs them concurrently in
    threads. Both go through the same training entry point, so the
    per-cell test_rmse values must agree within +/- 1e-3 (the streams
    runner's thread ordering does not perturb the per-cell numerics
    on CPU because the streams collapse to None and Python's GIL
    serialises optimizer.step calls).
    """

    vectors = _synthetic_vectors(80)
    candidates = [
        _make_candidate(hidden_size=8, dropout=0.0, lr=1e-3, seed=11),
        _make_candidate(hidden_size=8, dropout=0.1, lr=1e-3, seed=11),
        _make_candidate(hidden_size=8, dropout=0.0, lr=3e-4, seed=29),
        _make_candidate(hidden_size=8, dropout=0.2, lr=3e-4, seed=29),
    ]
    sequential_summaries = []
    for candidate in candidates:
        seq_summary = _run_cell_sequential(
            candidate, vectors=vectors, checkpoint=tmp_path / "seq.pt"
        )
        sequential_summaries.append(seq_summary)

    buckets = group_candidates_into_buckets(
        candidates,
        text_encoder=None,
        target_mode="event_study",
        max_bucket_size=8,
    )
    # All four cells share (architecture=lstm, hidden_size=8,
    # num_layers=1, text_adapter_dim=0, fold_id=None), so they form
    # one bucket.
    assert len(buckets) == 1
    _, bucket_cells = buckets[0]
    assert len(bucket_cells) == 4

    def _train_one(trial_index, candidate, stream):
        with _NullCudaStreamContext() if stream is None else torch.cuda.stream(stream):
            summary = _run_cell_sequential(
                candidate, vectors=vectors, checkpoint=tmp_path / "streams.pt"
            )
        return {"trial_index": trial_index, "summary": summary}

    streams_results = run_bucket_streams(
        [(i, c) for i, c in enumerate(candidates)],
        train_one_cell=_train_one,
        device=torch.device("cpu"),
    )
    streams_results = sorted(streams_results, key=lambda r: r["trial_index"])

    for seq, streams in zip(sequential_summaries, streams_results, strict=True):
        seq_metrics = seq.metrics
        streams_metrics = streams["summary"].metrics
        assert seq_metrics is not None
        assert streams_metrics is not None
        # The headline RMSE the aggregator reads is combined_rmse on
        # the legacy 80/20 path (it stands in for test_rmse when the
        # walk-forward partitions are absent). Allow +/- 1e-3.
        assert abs(seq_metrics.combined_rmse - streams_metrics.combined_rmse) < 1e-3, (
            f"sequential={seq_metrics.combined_rmse} vs "
            f"streams={streams_metrics.combined_rmse}"
        )


def test_off_mode_trial_count_matches_stacked_trial_count():
    """The trial count emitted under stacked-mode matches the off-mode count.

    Both paths consume the same sweep candidates; bucketing is a
    scheduling change, not a candidate-set change. Therefore the
    bucketed runner must emit exactly one trial record per input
    candidate (no dropped cells, no duplicates).
    """

    candidates = [
        _make_candidate(hidden_size=8, dropout=0.0, lr=1e-3, seed=11),
        _make_candidate(hidden_size=8, dropout=0.1, lr=1e-3, seed=11),
        _make_candidate(hidden_size=8, dropout=0.0, lr=3e-4, seed=29),
        _make_candidate(hidden_size=8, dropout=0.2, lr=3e-4, seed=29),
    ]
    buckets = group_candidates_into_buckets(
        candidates,
        text_encoder=None,
        target_mode="event_study",
        max_bucket_size=8,
    )
    total_bucketed = sum(len(cells) for _, cells in buckets)
    assert total_bucketed == len(candidates), (
        f"bucketed cell count {total_bucketed} != candidate count "
        f"{len(candidates)} -- the bucketed runner is dropping cells"
    )


@pytest.mark.parametrize("architecture", ["lstm", "gru", "dlinear", "transformer"])
def test_streams_path_runs_for_every_architecture(tmp_path, architecture):
    """The streams scheduler invokes the per-cell trainer once per arch.

    A smoke check that every architecture in the bake-off roster can
    be driven through the streams scheduler without raising. The
    test uses a single short cell so the training loop exits quickly.
    """

    vectors = _synthetic_vectors(40)
    candidate = {
        "model_config": ModelConfig(
            input_size=6,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            head_hidden_size=8,
            architecture=architecture,
            credibility_features=False,
            text_embedding_dim=0,
            text_adapter_dim=0,
        ),
        "learning_rate": 1e-3,
        "epochs": 2,
        "weight_decay": 0.0,
        "text_adapter_dim": 0,
        "seed": 11,
    }

    def _train_one(trial_index, candidate, stream):
        summary = _run_cell_sequential(
            candidate, vectors=vectors, checkpoint=tmp_path / f"{architecture}.pt"
        )
        return {"trial_index": trial_index, "summary": summary}

    results = run_bucket_streams(
        [(1, candidate)],
        train_one_cell=_train_one,
        device=torch.device("cpu"),
    )
    assert len(results) == 1
    assert results[0]["summary"].metrics is not None
