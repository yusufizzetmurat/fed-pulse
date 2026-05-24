"""Cover the symmetric InfoNCE loss (#235).

The contrastive objective is small enough that the unit tests can
pin its load-bearing behaviour exactly: identical inputs minimise
the loss (modulo the temperature-driven floor), random inputs sit
above that floor, and degenerate single-row batches return a graph-
attached zero rather than NaN.
"""

from __future__ import annotations

import math

import torch


def test_info_nce_loss_is_low_when_paired_rows_align() -> None:
    from app.training.info_nce_loss import InfoNCELoss

    torch.manual_seed(0)
    a = torch.randn(8, 16, requires_grad=True)
    # Perfect alignment: b is a scaled copy of a so every diagonal
    # pair is the row's nearest neighbour after L2 normalisation.
    b = a.detach().clone() * 1.5
    loss_aligned = InfoNCELoss(temperature=0.07)(a, b)
    # Misalign by shuffling b so the diagonals stop being the
    # nearest neighbours; the loss should now be much higher.
    perm = torch.tensor([4, 5, 6, 7, 0, 1, 2, 3])
    loss_misaligned = InfoNCELoss(temperature=0.07)(a, b[perm])
    assert float(loss_aligned) < 0.1
    assert float(loss_misaligned) > 1.0


def test_info_nce_loss_handles_single_row_batch_without_nan() -> None:
    """B=1 has no contrastive negatives; the loss must return a
    graph-attached zero so the optimiser still sees a finite scalar."""

    from app.training.info_nce_loss import InfoNCELoss

    a = torch.randn(1, 8, requires_grad=True)
    b = torch.randn(1, 8, requires_grad=True)
    loss = InfoNCELoss(temperature=0.07)(a, b)
    assert torch.isfinite(loss)
    assert float(loss) == 0.0
    # The graph attachment is the load-bearing part — if backward
    # raises, downstream training would crash whenever a fold ends
    # with a final mini-batch of size 1.
    loss.backward()


def test_info_nce_loss_rejects_temperature_zero_or_negative() -> None:
    from app.training.info_nce_loss import InfoNCELoss

    import pytest

    with pytest.raises(ValueError):
        InfoNCELoss(temperature=0.0)
    with pytest.raises(ValueError):
        InfoNCELoss(temperature=-0.1)


def test_info_nce_loss_rejects_mismatched_shapes() -> None:
    from app.training.info_nce_loss import InfoNCELoss

    import pytest

    loss_fn = InfoNCELoss(temperature=0.07)
    with pytest.raises(ValueError):
        loss_fn(torch.randn(4, 8), torch.randn(4, 16))
    with pytest.raises(ValueError):
        loss_fn(torch.randn(8), torch.randn(8))  # not 2-D


def test_info_nce_loss_is_symmetric_under_input_swap() -> None:
    """The Kong et al. (Eq. 7) objective averages both directions
    (text→market and market→text). Swapping the inputs must produce
    the same loss value."""

    from app.training.info_nce_loss import InfoNCELoss

    torch.manual_seed(1)
    a = torch.randn(6, 12)
    b = torch.randn(6, 12)
    loss_ab = InfoNCELoss(temperature=0.07)(a, b)
    loss_ba = InfoNCELoss(temperature=0.07)(b, a)
    assert math.isclose(float(loss_ab), float(loss_ba), rel_tol=1e-5)
