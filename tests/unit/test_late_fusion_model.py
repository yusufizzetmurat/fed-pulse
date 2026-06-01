"""Unit tests for the clean-room late-fusion model."""

from __future__ import annotations

import pytest
import torch

from app.data.late_fusion_model import (
    LateFusionModel,
    assert_text_gradient_flows,
    joint_loss,
)


def test_forward_shapes_full_fusion() -> None:
    model = LateFusionModel(text_dim=8, struct_dim=4)
    text = torch.randn(5, 8)
    struct = torch.randn(5, 4)
    dir_logit, magnitude = model(text, struct)
    assert dir_logit.shape == (5,)
    assert magnitude.shape == (5,)
    assert torch.isfinite(magnitude).all()


def test_market_only_and_text_only_configs() -> None:
    market_only = LateFusionModel(text_dim=8, struct_dim=4, use_text=False)
    text_only = LateFusionModel(text_dim=8, struct_dim=4, use_struct=False)
    text = torch.randn(3, 8)
    struct = torch.randn(3, 4)
    # both run without error and produce the right shapes
    assert market_only(text, struct)[0].shape == (3,)
    assert text_only(text, struct)[0].shape == (3,)
    # market-only must have no text branch
    assert not hasattr(market_only, "text_branch")
    assert not hasattr(text_only, "struct_branch")


def test_requires_at_least_one_branch() -> None:
    with pytest.raises(ValueError, match="at least one"):
        LateFusionModel(text_dim=8, struct_dim=4, use_text=False, use_struct=False)


def test_text_gradient_flows_in_full_model() -> None:
    model = LateFusionModel(text_dim=8, struct_dim=4)
    text = torch.randn(6, 8)
    struct = torch.randn(6, 4)
    total = assert_text_gradient_flows(model, text, struct)
    assert total > 0.0


def test_text_gradient_check_rejects_market_only() -> None:
    model = LateFusionModel(text_dim=8, struct_dim=4, use_text=False)
    with pytest.raises(ValueError, match="no text branch"):
        assert_text_gradient_flows(model, torch.randn(2, 8), torch.randn(2, 4))


def test_joint_loss_is_finite_and_positive() -> None:
    model = LateFusionModel(text_dim=8, struct_dim=4)
    text = torch.randn(4, 8)
    struct = torch.randn(4, 4)
    dir_logit, magnitude = model(text, struct)
    loss = joint_loss(
        dir_logit, magnitude, torch.ones(4), torch.rand(4)
    )
    assert torch.isfinite(loss)
    assert loss.item() > 0
