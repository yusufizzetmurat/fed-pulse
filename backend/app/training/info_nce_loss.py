"""Symmetric InfoNCE loss for multi-modal alignment (#235).

Kong et al. 2025 (M2VN) align a text representation with a market
representation by treating the (text_t, market_t) pair as a positive
example and every (text_t, market_t') with t != t' as a negative.
The InfoNCE objective is the symmetric NT-Xent loss with a fixed
temperature: minimise the cross-entropy between row-wise similarities
and the identity-permutation labels, then average the text→market
and market→text directions so neither modality dominates.

The temperature is a constructor argument, not a learned parameter.
Our training batches are small (B=16) — far below the ~256+ batch
sizes where a gradient-trained log-temperature converges stably (cf.
CLIP) — so the constant-τ recipe avoids the temperature collapse
modes that small-batch learned-τ runs are prone to.

The loss assumes both inputs already live in a shared latent
dimension (the projection heads in
``app.models.gated_infonce_fusion.GatedInfoNCEFusion`` do that step);
this module only computes the contrastive term. Vectors are L2-
normalised inside the forward so callers do not need to pre-normalise.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE between two L2-normalised representations.

    Inputs are ``(B, D)`` tensors representing one modality each
    (typically ``r_t`` for the market projection and ``t_t`` for the
    text projection). The loss pulls the diagonal of the similarity
    matrix toward the L2-normalised pair and pushes off-diagonal
    pairs apart, scaled by the inverse temperature.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(
                f"InfoNCELoss temperature must be > 0; got {temperature}"
            )
        self.temperature = float(temperature)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return the symmetric NT-Xent loss between ``a`` and ``b``.

        Both tensors must share shape ``(B, D)``. Returns a 0-d
        scalar suitable for adding to any other loss term in the
        training step. When the batch has only one row (B=1) the
        contrastive term degenerates (no negatives), so the loss
        returns a graph-attached zero rather than a NaN.
        """

        if a.shape != b.shape:
            raise ValueError(
                f"InfoNCELoss inputs must have matching shape; got {a.shape} vs {b.shape}"
            )
        if a.dim() != 2:
            raise ValueError(
                f"InfoNCELoss inputs must be 2-D (B, D); got {a.dim()}-D"
            )
        batch_size = a.size(0)
        if batch_size < 2:
            # No negatives in a single-row batch; emit zero with the
            # same graph attachment so the optimiser can still see it.
            return a.sum() * 0.0
        a_norm = F.normalize(a, dim=-1)
        b_norm = F.normalize(b, dim=-1)
        logits = torch.matmul(a_norm, b_norm.T) / self.temperature
        labels = torch.arange(batch_size, device=logits.device)
        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.T, labels)
        return (loss_ab + loss_ba) * 0.5
