"""Pooling tests for the AutoModel encoder zoo."""

from __future__ import annotations
import torch
from app.data import fed_comms_encoders as e


def test_cls_pooling_takes_first_token() -> None:
    hidden = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    attn = torch.ones(2, 3)
    out = e._pool(hidden, attn, "cls")
    assert torch.allclose(out, hidden[:, 0])


def test_mean_pooling_respects_mask() -> None:
    hidden = torch.ones(1, 3, 2)
    hidden[0, 2] = 5.0  # this token is masked out
    attn = torch.tensor([[1.0, 1.0, 0.0]])
    out = e._pool(hidden, attn, "mean")
    assert torch.allclose(out, torch.ones(1, 2))  # masked token excluded → mean of ones


def test_encoder_registry_consistent() -> None:
    assert set(e.ENCODERS) == set(e._HF_IDS)
