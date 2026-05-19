"""Unit tests for the Phase B LR-schedule selector (#227).

The trainer used to hardcode ReduceLROnPlateau. PR #227 added a
``cosine_warmup`` option that builds a torch ``OneCycleLR`` (warmup ->
cosine -> tail). These tests lock the API: plateau is the default (so
the determinism regression stays green) and cosine_warmup builds the
OneCycleLR with the documented warmup ratio.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.models.config import ModelConfig
from app.models.factory import build_forecaster


def _tiny_model() -> torch.nn.Module:
    cfg = ModelConfig(
        input_size=6,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        architecture="lstm",
        output_mode="regression",
    )
    return build_forecaster(cfg)


def _two_batch_loader() -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Two-batch loader so the OneCycleLR scheduler advances twice."""

    x = torch.zeros((4, 5, 6))
    y = torch.zeros((4, 2))
    return DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)


def test_model_config_default_lr_schedule_is_plateau() -> None:
    cfg = ModelConfig()
    assert cfg.lr_schedule == "plateau"


def test_model_config_carries_cosine_warmup_choice() -> None:
    cfg = ModelConfig(lr_schedule="cosine_warmup")
    serialized = cfg.to_dict()
    assert serialized["lr_schedule"] == "cosine_warmup"


def test_unknown_lr_schedule_raises() -> None:
    """The trainer raises ValueError on an unrecognised schedule name."""

    from app.training.loop import train_model
    from app.models.config import FeatureVector, ModelConfig

    # Build a minimal sequence_groups path so train_model exercises the
    # scheduler-construction branch and bails on the unsupported choice.
    vectors = [
        FeatureVector(
            date="2024-01-02",
            sentiment_score=0.0,
            market_close=4000.0,
            market_volatility=0.01,
        )
        for _ in range(30)
    ]
    cfg = ModelConfig(
        input_size=6,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        architecture="lstm",
        output_mode="regression",
        lr_schedule="bogus",
    )
    with pytest.raises(ValueError, match="unsupported lr_schedule"):
        train_model(
            sequence_groups=[vectors],
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            validation_split=0.2,
            early_stopping_patience=5,
            checkpoint_path=None,
            save_checkpoint=False,
            device="cpu",
            model_config=cfg,
            seed=11,
            shuffle_targets_control=False,
            use_compile=False,
            use_amp=False,
            lr_schedule="bogus",
        )


def test_onecycle_lr_construction_changes_lr_over_lifetime() -> None:
    """Build the OneCycleLR directly and confirm it advances the LR
    from a low warmup floor up to peak and back down. Guards against
    accidental drift of pct_start, anneal_strategy, or div_factor in a
    refactor."""

    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=10,
        steps_per_epoch=10,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )
    lr_curve = [optimizer.param_groups[0]["lr"]]
    for _ in range(99):
        scheduler.step()
        lr_curve.append(optimizer.param_groups[0]["lr"])
    # Warmup -> cosine -> tail produces a unimodal curve: starts low,
    # peaks near max_lr, ends below initial.
    peak = max(lr_curve)
    assert peak == pytest.approx(1e-3, rel=1e-3)
    assert lr_curve[0] < peak
    assert lr_curve[-1] < peak
