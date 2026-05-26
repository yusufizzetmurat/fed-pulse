"""Unit tests for the batched-sweep dispatch helpers.

These pin the per-architecture routing table + the bucket key
contract + the run_bucket_streams error path. The vmap-friendly
``StackedDLinear`` + ``BatchedAdamW`` machinery is already covered
in ``test_forecaster_batched_hp.py`` and the parity test under
``tests/regression/test_forecaster_batched_parity.py``; the tests
here focus on the routing primitives and the streams-mode worker
error contract.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.models.config import ModelConfig
from app.training.batched_sweep import (
    BATCHING_MODES,
    BucketKey,
    DEFAULT_BATCHING_MODE_BY_ARCH,
    DEFAULT_MAX_BUCKET_SIZE_BY_ARCH,
    bucket_key_for_candidate,
    format_bucket_log_line,
    group_candidates_into_buckets,
    resolve_batching_mode,
    resolve_max_bucket_size,
    route_bucket,
    run_bucket_streams,
)


def _candidate(
    *,
    architecture: str = "lstm",
    hidden_size: int = 64,
    num_layers: int = 2,
    text_adapter_dim: int = 0,
    fold_id: str | None = None,
    extras: dict[str, object] | None = None,
) -> dict[str, object]:
    cfg = ModelConfig(
        architecture=architecture,
        hidden_size=hidden_size,
        num_layers=num_layers,
        text_adapter_dim=text_adapter_dim,
    )
    out: dict[str, object] = {"model_config": cfg, "learning_rate": 1e-3}
    if fold_id is not None:
        out["fold_id"] = fold_id
    if extras:
        out.update(extras)
    return out


def test_resolve_max_bucket_size_uses_override() -> None:
    assert resolve_max_bucket_size("lstm", override=5) == 5


def test_resolve_max_bucket_size_falls_back_to_table() -> None:
    assert resolve_max_bucket_size("lstm") == DEFAULT_MAX_BUCKET_SIZE_BY_ARCH["lstm"]


def test_resolve_max_bucket_size_unknown_arch_returns_safe_default() -> None:
    assert resolve_max_bucket_size("not-a-real-arch") == 4


def test_resolve_max_bucket_size_ignores_zero_or_negative_override() -> None:
    assert resolve_max_bucket_size("lstm", override=0) == DEFAULT_MAX_BUCKET_SIZE_BY_ARCH["lstm"]
    assert resolve_max_bucket_size("lstm", override=-1) == DEFAULT_MAX_BUCKET_SIZE_BY_ARCH["lstm"]


def test_resolve_batching_mode_passthrough_for_explicit_mode() -> None:
    for mode in BATCHING_MODES:
        if mode == "auto":
            continue
        assert resolve_batching_mode("lstm", mode=mode) == mode


def test_resolve_batching_mode_auto_uses_per_arch_table() -> None:
    assert resolve_batching_mode("dlinear", mode="auto") == "stacked"
    assert resolve_batching_mode("lstm", mode="auto") == "streams"
    # Unknown architectures default to streams under auto.
    assert resolve_batching_mode("unknown_arch", mode="auto") == "streams"


def test_resolve_batching_mode_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="unknown batching mode"):
        resolve_batching_mode("lstm", mode="garbage")


def test_route_bucket_returns_resolved_for_compatible_arch() -> None:
    assert route_bucket("dlinear", mode="stacked") == "stacked"
    assert route_bucket("lstm", mode="streams") == "streams"


def test_route_bucket_falls_back_when_stacked_request_incompatible() -> None:
    # lstm is not stacked-capable; an explicit "stacked" request
    # downgrades to "streams" with a logger warning.
    assert route_bucket("lstm", mode="stacked") == "streams"


def test_bucket_key_collects_topology_axes() -> None:
    cand = _candidate(architecture="lstm", hidden_size=128, num_layers=3, text_adapter_dim=64, fold_id="wf_fold_1")
    key = bucket_key_for_candidate(cand, text_encoder="finbert", target_mode="event_study")
    assert key.architecture == "lstm"
    assert key.hidden_size == 128
    assert key.num_layers == 3
    assert key.text_adapter_dim == 64
    assert key.text_encoder == "finbert"
    assert key.fold_id == "wf_fold_1"
    assert key.target_mode == "event_study"


def test_bucket_key_handles_none_text_encoder() -> None:
    cand = _candidate()
    key = bucket_key_for_candidate(cand, text_encoder=None, target_mode="event_study")
    assert key.text_encoder == "none"


def test_group_candidates_into_buckets_preserves_first_seen_order() -> None:
    # First candidate is hidden=64; second is hidden=128 (new bucket);
    # third is hidden=64 again (same bucket as first).
    candidates = [
        _candidate(hidden_size=64),
        _candidate(hidden_size=128),
        _candidate(hidden_size=64),
    ]
    buckets = group_candidates_into_buckets(
        candidates, text_encoder=None, target_mode="event_study"
    )
    # Two buckets total; the hidden=64 bucket comes first because its
    # first candidate appeared first.
    assert len(buckets) == 2
    first_key, first_cells = buckets[0]
    assert first_key.hidden_size == 64
    indices_in_first = [trial_index for trial_index, _ in first_cells]
    assert indices_in_first == [1, 3]
    second_key, second_cells = buckets[1]
    assert second_key.hidden_size == 128
    assert [t for t, _ in second_cells] == [2]


def test_group_candidates_into_buckets_splits_on_max_bucket_size() -> None:
    # 5 candidates with the same key + max_bucket_size=2 -> 3 buckets
    # of sizes 2, 2, 1.
    candidates = [_candidate(hidden_size=64) for _ in range(5)]
    buckets = group_candidates_into_buckets(
        candidates, text_encoder=None, target_mode="event_study", max_bucket_size=2
    )
    sizes = [len(cells) for _, cells in buckets]
    assert sizes == [2, 2, 1]


def test_format_bucket_log_line_carries_bucket_size_and_mode() -> None:
    key = BucketKey(
        architecture="lstm",
        hidden_size=64,
        num_layers=2,
        text_adapter_dim=0,
        text_encoder="none",
        fold_id="wf_fold_2",
        target_mode="event_study",
    )
    cells = [(1, _candidate()), (2, _candidate())]
    line = format_bucket_log_line(key, cells, routed_mode="streams")
    assert "arch=lstm" in line
    assert "bucket_size=2" in line
    assert "mode=streams" in line
    assert "fold=wf_fold_2" in line


def test_run_bucket_streams_short_circuits_on_empty_input() -> None:
    out = run_bucket_streams(
        [],
        train_one_cell=lambda *a, **k: {"unused": True},
        device=torch.device("cpu"),
    )
    assert out == []


def test_run_bucket_streams_cpu_runs_cells_sequentially() -> None:
    # On CPU device the runner has no streams to multiplex; it falls
    # back to sequential execution so per-cell determinism survives.
    seen: list[int] = []

    def _trainer(trial_index: int, _cand: dict[str, object], stream: object) -> dict[str, object]:
        seen.append(trial_index)
        assert stream is None
        return {"trial_index": trial_index}

    cells = [(1, _candidate()), (2, _candidate()), (3, _candidate())]
    out = run_bucket_streams(cells, train_one_cell=_trainer, device=torch.device("cpu"))
    assert seen == [1, 2, 3]
    assert [r["trial_index"] for r in out] == [1, 2, 3]


def test_run_bucket_streams_propagates_worker_exception() -> None:
    """A per-cell worker error is re-raised after the bucket joins.

    Direct attribute: the post-fix narrows ``BaseException`` to
    ``Exception``, so a ``RuntimeError`` from one cell still surfaces
    on the parent thread.
    """

    def _trainer(trial_index: int, _cand: dict[str, object], stream: object) -> dict[str, object]:
        if trial_index == 2:
            raise RuntimeError("synthetic per-cell failure")
        return {"trial_index": trial_index}

    cells = [(1, _candidate()), (2, _candidate())]
    with pytest.raises(RuntimeError, match="synthetic per-cell failure"):
        run_bucket_streams(cells, train_one_cell=_trainer, device=torch.device("cpu"))


def test_cell_early_stop_resets_stale_counter_on_improvement() -> None:
    from app.training.batched_sweep import _CellEarlyStop

    stop = _CellEarlyStop()
    stop.update(epoch=1, loss=1.0, patience=3)
    assert stop.best_loss == 1.0
    assert stop.best_epoch == 1
    assert stop.stale_epochs == 0
    assert stop.stopped is False
    # Strict improvement (more than 1e-6 better) resets stale_epochs.
    stop.update(epoch=2, loss=0.5, patience=3)
    assert stop.stale_epochs == 0
    assert stop.best_epoch == 2


def test_cell_early_stop_marks_stopped_after_patience_breached() -> None:
    from app.training.batched_sweep import _CellEarlyStop

    stop = _CellEarlyStop()
    stop.update(epoch=1, loss=1.0, patience=2)
    stop.update(epoch=2, loss=1.1, patience=2)
    assert stop.stale_epochs == 1
    assert stop.stopped is False
    stop.update(epoch=3, loss=1.2, patience=2)
    assert stop.stale_epochs == 2
    assert stop.stopped is True
