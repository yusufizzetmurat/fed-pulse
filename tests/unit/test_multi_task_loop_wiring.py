"""End-to-end wiring tests for the multi-task loss path (#273).

Covers the DataLoader-shape contract (``_make_partition_dataset`` +
``_unpack_batch``) and the train-step branch that swaps the single-task
CrossEntropy for :class:`MultiTaskLoss` when the model config carries
``multi_task_loss=True``.
"""

from __future__ import annotations

import datetime as _dt

import pytest
import torch
from torch.utils.data import DataLoader

from app.training.loop import (
    _MULTI_TASK_AUX_KEYS,
    _make_partition_dataset,
    _unpack_batch,
)


def _rand(shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype).normal_(0.0, 1.0) if dtype.is_floating_point else torch.zeros(shape, dtype=dtype)


def _make_aux(n_rows: int) -> dict[str, torch.Tensor]:
    return {
        "factor": torch.zeros(n_rows, dtype=torch.float32),
        "factor_mask": torch.zeros(n_rows, dtype=torch.bool),
        "certainty": torch.zeros(n_rows, dtype=torch.long),
        "certainty_mask": torch.zeros(n_rows, dtype=torch.bool),
        "topic": torch.zeros(n_rows, dtype=torch.long),
        "topic_mask": torch.zeros(n_rows, dtype=torch.bool),
    }


def test_make_partition_dataset_minimal_arity() -> None:
    """No text + no multi-task -> arity 2."""

    n = 8
    x = _rand((n, 5, 6))
    y = torch.zeros(n, dtype=torch.long)
    ds = _make_partition_dataset(x, y, None, None, None)
    loader = DataLoader(ds, batch_size=n, shuffle=False)
    batch = next(iter(loader))
    assert len(batch) == 2
    bx, by, text, missing, aux = _unpack_batch(batch)
    assert bx.shape == x.shape
    assert by.shape == y.shape
    assert text is None
    assert missing is None
    assert aux is None


def test_make_partition_dataset_text_arity() -> None:
    """Text path active -> arity 4 with (text, missing) packed."""

    n = 8
    x = _rand((n, 5, 6))
    y = torch.zeros(n, dtype=torch.long)
    text = _rand((n, 32))
    text_missing = torch.zeros((n, 1), dtype=torch.float32)
    ds = _make_partition_dataset(x, y, text, text_missing, None)
    loader = DataLoader(ds, batch_size=n, shuffle=False)
    batch = next(iter(loader))
    assert len(batch) == 4
    bx, by, btext, bmissing, aux = _unpack_batch(batch)
    assert bx.shape == x.shape
    assert btext is not None and btext.shape == text.shape
    assert bmissing is not None and bmissing.shape == text_missing.shape
    assert aux is None


def test_make_partition_dataset_multi_task_arity() -> None:
    """multi-task active, no text -> arity 8 with aux tensors packed."""

    n = 8
    x = _rand((n, 5, 6))
    y = torch.zeros(n, dtype=torch.long)
    aux_in = _make_aux(n)
    ds = _make_partition_dataset(x, y, None, None, aux_in)
    loader = DataLoader(ds, batch_size=n, shuffle=False)
    batch = next(iter(loader))
    assert len(batch) == 8
    bx, by, btext, bmissing, aux_out = _unpack_batch(batch)
    assert btext is None
    assert bmissing is None
    assert aux_out is not None
    for key in _MULTI_TASK_AUX_KEYS:
        assert key in aux_out
        assert aux_out[key].shape == aux_in[key].shape


def test_make_partition_dataset_text_plus_multi_task_arity() -> None:
    """text + multi-task -> arity 10."""

    n = 4
    x = _rand((n, 5, 6))
    y = torch.zeros(n, dtype=torch.long)
    text = _rand((n, 32))
    text_missing = torch.zeros((n, 1), dtype=torch.float32)
    aux_in = _make_aux(n)
    ds = _make_partition_dataset(x, y, text, text_missing, aux_in)
    loader = DataLoader(ds, batch_size=n, shuffle=False)
    batch = next(iter(loader))
    assert len(batch) == 10
    bx, by, btext, bmissing, aux_out = _unpack_batch(batch)
    assert btext is not None
    assert aux_out is not None
    for key in _MULTI_TASK_AUX_KEYS:
        assert aux_out[key].shape == aux_in[key].shape


def test_make_partition_dataset_rejects_missing_aux_key() -> None:
    """An aux dict missing a required key raises with a clear message."""

    n = 4
    x = _rand((n, 5, 6))
    y = torch.zeros(n, dtype=torch.long)
    bad_aux = {k: torch.zeros(n) for k in ("factor", "factor_mask")}  # incomplete
    with pytest.raises(ValueError, match="missing required key"):
        _make_partition_dataset(x, y, None, None, bad_aux)


def test_unpack_batch_rejects_unknown_arity() -> None:
    """A 3-tuple batch (or other invalid arity) raises."""

    with pytest.raises(ValueError, match="unexpected batch arity"):
        _unpack_batch((torch.zeros(1), torch.zeros(1), torch.zeros(1)))


def test_multi_task_loss_active_requires_classification_mode() -> None:
    """train_model raises when multi_task_loss=True + output_mode=regression.

    The guard fires early in the walk-forward branch (before model build
    or partition tensorisation), so a minimal call with synthetic
    sequence groups is enough to exercise it.
    """

    from app.models.config import ModelConfig
    from app.training.loop import train_model

    config = ModelConfig(
        output_mode="regression",
        multi_task_loss=True,
    )
    groups = [[_dummy_feature_vector(vol=0.01 * (i + 1), day=i + 1) for i in range(25)]]
    with pytest.raises(ValueError, match="output_mode='classification'"):
        train_model(
            model_config=config,
            train_sequence_groups=groups,
            val_sequence_groups=groups,
            test_sequence_groups=groups,
            epochs=1,
            save_checkpoint=False,
        )


def test_multi_task_loss_active_rejects_gated_infonce_combo() -> None:
    """multi_task_loss + gated_infonce in the same cell is explicitly unsupported.

    The guard fires after partition tensorisation but before the epoch
    loop, so the synthetic sequence groups must carry valid
    forward_realized_vol_10d values for the classification-mode build
    to succeed up to that point.
    """

    from app.models.config import ModelConfig
    from app.training.loop import train_model

    config = ModelConfig(
        output_mode="classification",
        multi_task_loss=True,
        fusion_mode="gated_infonce",
        n_classes=3,
        text_embedding_dim=32,
        text_adapter_dim=32,
    )
    groups = [[_dummy_feature_vector(vol=0.01 * (i + 1), day=i + 1) for i in range(25)]]
    with pytest.raises(ValueError, match="not yet supported"):
        train_model(
            model_config=config,
            train_sequence_groups=groups,
            val_sequence_groups=groups,
            test_sequence_groups=groups,
            epochs=1,
            save_checkpoint=False,
        )


def _dummy_feature_vector(*, vol: float = 0.02, day: int = 1):
    from app.models.config import FeatureVector

    return FeatureVector(
        date=_dt.date(2025, 1, day),
        sentiment_score=0.0,
        market_close=100.0,
        market_volatility=0.01,
        close_change_pct=0.0,
        volatility_change=0.0,
        elapsed_time=0.0,
        forward_realized_vol_10d=vol,
    )
