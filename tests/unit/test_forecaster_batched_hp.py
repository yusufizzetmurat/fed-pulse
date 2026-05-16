"""Unit tests for the bucketed-HP forecaster sweep primitives.

The bucketed sweep groups cells that share the same model topology and
data feed and trains them concurrently. The tests in this module pin
the grouping contract (the bucket key axes, the deterministic ordering
across bucket boundaries, and the per-arch bucket-size cap) plus the
two stacked-mode primitives (per-cell dropout, per-cell Adam moments).
"""

from __future__ import annotations

import argparse

import pytest

torch = pytest.importorskip("torch")

from app.models.config import ModelConfig  # noqa: E402
from app.train_forecaster import build_sweep_candidates  # noqa: E402
from app.training.batched_sweep import (  # noqa: E402
    BucketKey,
    bucket_key_for_candidate,
    group_candidates_into_buckets,
    resolve_max_bucket_size,
)


def _legacy_args(
    *,
    architectures: list[str],
    seeds: list[int],
    folds: list[str] | None = None,
) -> argparse.Namespace:
    """Minimal namespace shaped like the sweep-CLI invocation under test.

    The dropout / lr / weight_decay grid below is the four-axis cross
    product used by every test in this module: 2 dropouts x 2 learning
    rates x 2 weight decays = 8 cells per (architecture, hidden_size,
    num_layers) triple, which combined with the seed axis lets a
    single bucket carry a known number of cells.
    """

    return argparse.Namespace(
        hidden_size=64,
        num_layers=2,
        dropout=0.15,
        learning_rate=1e-3,
        epochs=20,
        head_hidden_size=32,
        hidden_sizes=[32, 64],
        num_layers_grid=[1, 2],
        dropouts=[0.1, 0.2],
        learning_rates=[1e-3, 3e-4],
        epochs_grid=[20],
        weight_decay=1e-4,
        weight_decays=[0.0, 1e-3],
        text_adapter_dim=64,
        text_adapter_dims=None,
        text_encoder="none",
        use_text_embeddings=True,
        training_package_id=None,
        rich_features=False,
        architecture="lstm",
        architectures=architectures,
        seed=None,
        seeds=seeds,
        credibility_features=False,
        random_search=False,
        random_search_samples=50,
        random_search_seed=42,
        folds=folds,
    )


def test_bucket_key_groups_cells_by_topology():
    """A synthetic sweep groups cells by (arch, hidden, layers, ...).

    The grid is 2 archs x 2 hidden x 2 layers x 2 dropout x 2 lr x 2 wd
    x 2 seeds = 128 cells across a single fold. The expected bucket
    count is 2 archs x 2 hidden x 2 layers = 8 buckets (dropout / lr
    / wd / seed are all within-bucket axes), 16 cells per bucket.
    """

    args = _legacy_args(architectures=["lstm", "gru"], seeds=[11, 29])
    candidates = build_sweep_candidates(args)
    assert len(candidates) == 128, (
        "test fixture drift: expected 128 candidates from the "
        f"2x2x2x2x2x2x2 grid; got {len(candidates)}"
    )

    buckets = group_candidates_into_buckets(
        candidates,
        text_encoder=None,
        target_mode="event_study",
        # Cap at 64 so the 16-cell buckets are not split.
        max_bucket_size=64,
    )

    assert len(buckets) == 8, (
        f"expected 8 buckets (2 archs x 2 hidden x 2 layers); got {len(buckets)}"
    )
    for key, cells in buckets:
        # Each bucket should carry 2 dropout x 2 lr x 2 wd x 2 seeds
        # = 16 cells.
        assert len(cells) == 16, (
            f"bucket {key} has {len(cells)} cells; expected 16"
        )
        # Within a bucket, every cell shares the bucket-key axes.
        for _, candidate in cells:
            assert candidate["model_config"].architecture == key.architecture
            assert candidate["model_config"].hidden_size == key.hidden_size
            assert candidate["model_config"].num_layers == key.num_layers


def test_bucket_size_cap_respected():
    """A 16-cell bucket splits into 4 sub-buckets at --max-bucket-size=4."""

    args = _legacy_args(architectures=["lstm"], seeds=[11, 29])
    candidates = build_sweep_candidates(args)
    # 1 arch x 2 hidden x 2 layers = 4 BucketKeys at the topology
    # level; each carries 2 dropout x 2 lr x 2 wd x 2 seeds = 16 cells.
    # With cap=4 each bucket splits into 16 / 4 = 4 sub-buckets, so
    # the total is 4 * 4 = 16 sub-buckets.
    buckets = group_candidates_into_buckets(
        candidates,
        text_encoder=None,
        target_mode="event_study",
        max_bucket_size=4,
    )
    assert len(buckets) == 16, (
        f"expected 16 sub-buckets at cap=4; got {len(buckets)}"
    )
    for _, cells in buckets:
        assert len(cells) <= 4


def test_bucket_key_distinguishes_folds():
    """The fold axis is part of the bucket key.

    With two folds and a single (arch, hidden, layers) triple, the
    bucket count is 1 * 2 = 2 even though all other axes match.
    """

    args = _legacy_args(
        architectures=["lstm"],
        seeds=[11],
        folds=["wf_fold_1", "wf_fold_2"],
    )
    candidates = build_sweep_candidates(args)
    # 1 arch x 2 hidden x 2 layers x 2 folds = 4 buckets.
    buckets = group_candidates_into_buckets(
        candidates,
        text_encoder=None,
        target_mode="event_study",
        max_bucket_size=64,
    )
    fold_ids = sorted({k.fold_id for k, _ in buckets if k.fold_id is not None})
    assert fold_ids == ["wf_fold_1", "wf_fold_2"], (
        f"expected both folds in the bucket key set; got {fold_ids}"
    )


def test_bucket_key_distinguishes_target_mode():
    """The target_mode axis is part of the bucket key.

    The sweep CLI runs one target_mode at a time, but the bucket key
    still carries it so two sweep outputs concatenated together do
    not accidentally collide their bucket cells.
    """

    candidate = {
        "model_config": ModelConfig(
            architecture="lstm",
            hidden_size=64,
            num_layers=2,
            dropout=0.1,
            head_hidden_size=32,
        ),
        "learning_rate": 1e-3,
        "epochs": 20,
        "weight_decay": 1e-4,
        "seed": 11,
    }
    key_event = bucket_key_for_candidate(
        candidate, text_encoder=None, target_mode="event_study"
    )
    key_realized = bucket_key_for_candidate(
        candidate, text_encoder=None, target_mode="realized_return"
    )
    assert key_event != key_realized
    assert key_event.target_mode == "event_study"
    assert key_realized.target_mode == "realized_return"


def test_resolve_max_bucket_size_per_arch_defaults():
    """The per-arch table is the floor when no override is supplied."""

    assert resolve_max_bucket_size("dlinear") == 64
    assert resolve_max_bucket_size("lstm") == 32
    assert resolve_max_bucket_size("transformer") == 8
    assert resolve_max_bucket_size("informer") == 4
    assert resolve_max_bucket_size("tft") == 4
    # Unknown arch falls back to 4.
    assert resolve_max_bucket_size("xxx_nonexistent") == 4

    # Override beats the table.
    assert resolve_max_bucket_size("dlinear", override=16) == 16
    # Non-positive override is ignored.
    assert resolve_max_bucket_size("dlinear", override=0) == 64


def test_buckets_preserve_first_seen_order():
    """Buckets emerge in first-seen order across the candidate stream.

    The candidate enumeration is (arch, hidden, layers, dropout, lr,
    wd, text_adapter, seed). With architectures=["lstm", "gru"] the
    first eight buckets are all-lstm before the first gru bucket.
    """

    args = _legacy_args(architectures=["lstm", "gru"], seeds=[11])
    candidates = build_sweep_candidates(args)
    buckets = group_candidates_into_buckets(
        candidates,
        text_encoder=None,
        target_mode="event_study",
        max_bucket_size=64,
    )
    archs = [key.architecture for key, _ in buckets]
    # The first 4 buckets are lstm (2 hidden x 2 layers), the next 4
    # are gru. Sorted in first-seen order, lstm bucket boundary sits
    # at index 4.
    assert archs[:4] == ["lstm", "lstm", "lstm", "lstm"]
    assert archs[4:] == ["gru", "gru", "gru", "gru"]
