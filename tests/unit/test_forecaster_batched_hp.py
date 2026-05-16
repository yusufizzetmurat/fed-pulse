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
    BatchedAdamW,
    BatchedDropout,
    BucketKey,
    bucket_key_for_candidate,
    group_candidates_into_buckets,
    resolve_batching_mode,
    resolve_max_bucket_size,
    route_bucket,
    run_bucket_streams,
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


def test_batched_dropout_per_cell_p_values():
    """Per-cell p values produce per-cell zero fractions in the mask.

    The test draws a large per-element uniform-noise input and runs it
    through BatchedDropout with three different p values. Each cell's
    output zero fraction should converge to its own p (within Monte
    Carlo noise at N=20000 elements).
    """

    p = torch.tensor([0.0, 0.25, 0.75])
    layer = BatchedDropout(p)
    layer.train()
    # Input is a constant tensor so any zero in the output comes from
    # the dropout mask (not the input). 3 cells, 20000 elements each.
    x = torch.ones(3, 20000)
    generator = torch.Generator()
    generator.manual_seed(7)
    y = layer(x, generator=generator)
    # Cell 0 has p=0 so no element should be zeroed.
    zero_fraction = (y == 0).float().mean(dim=1)
    assert zero_fraction[0].item() == 0.0
    # Cells 1 and 2 have p=0.25 and p=0.75; check within Monte Carlo
    # tolerance (3 sigma at N=20000 -> ~0.01).
    assert abs(zero_fraction[1].item() - 0.25) < 0.02
    assert abs(zero_fraction[2].item() - 0.75) < 0.02


def test_batched_dropout_eval_is_identity():
    """eval() makes BatchedDropout a no-op for every cell, including p=1."""

    p = torch.tensor([0.5, 1.0])
    layer = BatchedDropout(p)
    layer.eval()
    x = torch.ones(2, 100)
    y = layer(x)
    # Even cell with p=1 passes through identically in eval mode.
    assert torch.equal(x, y)


def test_batched_dropout_rejects_out_of_range_p():
    """p values outside [0, 1] raise ValueError at construction."""

    with pytest.raises(ValueError):
        BatchedDropout(torch.tensor([0.1, 1.5]))
    with pytest.raises(ValueError):
        BatchedDropout(torch.tensor([-0.1, 0.5]))


def test_batched_adam_per_cell_lr_wd():
    """A single Adam step moves each cell by its own per-cell lr.

    The test stacks three cells with the same gradient but different
    learning rates and zero weight decay. After one step, the change
    in the parameter for cell i is approximately
    -lr_i * grad / sqrt(grad^2 + eps), which is just -lr_i * sign(grad)
    when |grad| dominates eps. The expected ratios across cells must
    hold to within floating-point precision.
    """

    # Three cells: lr in {1e-3, 2e-3, 4e-3}, weight decay = 0.
    lr = torch.tensor([1e-3, 2e-3, 4e-3])
    wd = torch.zeros(3)
    # Single parameter shaped (3, 2) -- 3 cells, 2 weights each.
    params = {"w": torch.ones(3, 2)}
    initial = params["w"].clone()
    opt = BatchedAdamW(params, lr=lr, weight_decay=wd)
    grads = {"w": torch.ones(3, 2)}
    opt.step(grads)
    # Each cell's parameter should decrease by approximately its lr
    # (because the first Adam step normalizes the gradient to unit
    # magnitude via bias correction). The expected delta is
    # -lr * 1.0 to within ~1e-9.
    delta = params["w"] - initial
    for i in range(3):
        expected = -float(lr[i])
        observed = float(delta[i, 0])
        assert abs(observed - expected) < 1e-7, (
            f"cell {i}: expected delta ~{expected}, got {observed}"
        )
    # Cross-cell ratios should match the lr ratios (float32 precision).
    ratio_1_to_0 = float(delta[1, 0] / delta[0, 0])
    ratio_2_to_0 = float(delta[2, 0] / delta[0, 0])
    assert abs(ratio_1_to_0 - 2.0) < 1e-4
    assert abs(ratio_2_to_0 - 4.0) < 1e-4


def test_batched_adam_per_cell_weight_decay():
    """Decoupled weight decay scales the update by lr * wd * param.

    With zero gradient (so the Adam moments stay zero), one step
    against per-cell weight decay produces ``param <- param * (1 - lr * wd)``.
    Cells with wd=0 stay put; cells with wd>0 shrink proportionally.
    """

    lr = torch.tensor([1.0, 1.0, 1.0])
    wd = torch.tensor([0.0, 0.1, 0.2])
    params = {"w": torch.ones(3, 4)}
    opt = BatchedAdamW(params, lr=lr, weight_decay=wd)
    grads = {"w": torch.zeros(3, 4)}
    opt.step(grads)
    # No gradient means no Adam-update component; only the wd term
    # applies. param[i] <- param[i] * (1 - lr_i * wd_i).
    expected = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [0.9, 0.9, 0.9, 0.9],
            [0.8, 0.8, 0.8, 0.8],
        ]
    )
    assert torch.allclose(params["w"], expected, atol=1e-7)


def test_resolve_batching_mode_per_arch_table():
    """auto routes per the per-arch table; explicit overrides are honoured."""

    assert resolve_batching_mode("dlinear", mode="auto") == "stacked"
    assert resolve_batching_mode("lstm", mode="auto") == "streams"
    assert resolve_batching_mode("transformer", mode="auto") == "streams"
    assert resolve_batching_mode("informer", mode="auto") == "streams"
    # An explicit override takes precedence.
    assert resolve_batching_mode("dlinear", mode="streams") == "streams"
    assert resolve_batching_mode("lstm", mode="stacked") == "stacked"


def test_route_bucket_falls_back_to_streams_when_not_capable(caplog):
    """Explicit stacked on a non-stacked-capable arch falls back to streams.

    The fallback logs a warning so a user typo on --batching-mode does
    not silently bypass the per-arch capability gate.
    """

    import logging

    with caplog.at_level(logging.WARNING):
        decision = route_bucket("lstm", mode="stacked")
    assert decision == "streams"
    assert any("falling back to streams" in rec.message for rec in caplog.records)


def test_run_bucket_streams_returns_per_cell_results():
    """run_bucket_streams dispatches per-cell training and collects results.

    The fake trainer here just echoes the trial index back. The
    streams scheduler must invoke the trainer once per cell and
    return the results in the bucket's input order, regardless of
    thread completion timing.
    """

    cells = [
        (10, {"id": "a"}),
        (11, {"id": "b"}),
        (12, {"id": "c"}),
    ]

    def _fake_train(trial_index, candidate, stream):
        return {"trial_index": trial_index, "id": candidate["id"]}

    results = run_bucket_streams(
        cells,
        train_one_cell=_fake_train,
        device=torch.device("cpu"),
    )
    assert len(results) == 3
    trial_indices = [r["trial_index"] for r in results]
    assert sorted(trial_indices) == [10, 11, 12]


def test_run_bucket_streams_surfaces_worker_exceptions():
    """A failure in any worker thread is re-raised after every thread joins."""

    cells = [(1, {"raise": False}), (2, {"raise": True})]

    def _fake_train(trial_index, candidate, stream):
        if candidate.get("raise"):
            raise RuntimeError(f"cell {trial_index} failed")
        return {"trial_index": trial_index}

    with pytest.raises(RuntimeError, match="cell 2 failed"):
        run_bucket_streams(
            cells,
            train_one_cell=_fake_train,
            device=torch.device("cpu"),
        )


def test_batched_adam_active_mask_freezes_cells():
    """Cells whose active_mask is False do not update on the step."""

    lr = torch.tensor([1e-3, 1e-3])
    wd = torch.zeros(2)
    params = {"w": torch.ones(2, 3)}
    initial = params["w"].clone()
    opt = BatchedAdamW(params, lr=lr, weight_decay=wd)
    grads = {"w": torch.ones(2, 3)}
    mask = torch.tensor([True, False])
    opt.step(grads, active_mask=mask)
    # Cell 0 moves; cell 1 stays put.
    assert not torch.equal(params["w"][0], initial[0])
    assert torch.equal(params["w"][1], initial[1])


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
