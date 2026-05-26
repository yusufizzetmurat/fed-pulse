from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.data.embedding_comparator import _LinearHead, _mean_pool  # noqa: E402


def test_mean_pool_respects_attention_mask():
    last_hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    pooled = _mean_pool(last_hidden, attention_mask)
    assert pooled.shape == (2, 2)
    assert torch.allclose(pooled[0], torch.tensor([2.0, 3.0]))
    assert torch.allclose(pooled[1], torch.tensor([2.0, 0.0]))


def test_mean_pool_handles_all_zero_mask_without_division_by_zero():
    last_hidden = torch.zeros(1, 3, 4)
    attention_mask = torch.zeros(1, 3, dtype=torch.long)
    pooled = _mean_pool(last_hidden, attention_mask)
    assert pooled.shape == (1, 4)
    assert torch.all(torch.isfinite(pooled))


def test_linear_head_output_shape_matches_classes():
    head = _LinearHead(input_dim=8, num_classes=3)
    x = torch.randn(4, 8)
    logits = head(x)
    assert logits.shape == (4, 3)
