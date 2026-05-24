"""Gated multi-modal fusion with shared latent projections (#235).

Implements the projection-heads + per-row sigmoid gate from Kong et
al. 2025 (M2VN) Eq. 10-11. Two modality encoders (market + text)
project to the same latent dim; a learned gate weights them per row;
the fused representation feeds the downstream classification head.

The module is intentionally small: it knows nothing about the
classification head, the InfoNCE loss, or the recurrent backbone.
Callers wire those pieces together — see
``app.models.multimodal_forecaster.MultiModalForecasterModel`` for
the canonical integration.

The forward emits a dict with four keys so the training loop can
read off both the fused representation (for classification) and the
two per-modality projections (for the InfoNCE alignment loss). The
gate tensor is also returned so the inference path can log
modality reliance per request.
"""

from __future__ import annotations

import torch
from torch import nn


class GatedInfoNCEFusion(nn.Module):
    """Projection heads + per-row gated fusion.

    Each modality projects to a shared ``latent_dim`` via a single
    linear layer. The gate is the sigmoid of a linear over the
    concatenated raw inputs, broadcast across the latent dim. The
    fused output is ``gate * market_proj + (1 - gate) * text_proj``,
    matching Kong et al. Eq. 11.
    """

    def __init__(
        self,
        market_dim: int,
        text_dim: int,
        *,
        latent_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.market_dim = int(market_dim)
        self.text_dim = int(text_dim)
        self.latent_dim = int(latent_dim)
        self.dropout = float(dropout)

        self.market_proj = nn.Linear(self.market_dim, self.latent_dim)
        self.text_proj = nn.Linear(self.text_dim, self.latent_dim)
        # The gate reads the raw modality outputs (before projection)
        # so it has access to the unsquashed information when deciding
        # how much to trust each side.
        self.gate = nn.Linear(self.market_dim + self.text_dim, self.latent_dim)
        self.dropout_layer = nn.Dropout(self.dropout) if self.dropout > 0.0 else nn.Identity()

    def forward(
        self,
        market_pooled: torch.Tensor,
        text_pooled: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Emit ``{r_t, t_t, fused, gate}``.

        ``r_t`` and ``t_t`` are the latent-dim projections per modality
        — what the InfoNCE alignment loss reads. ``fused`` is the
        gated combination, ``(B, latent_dim)``, ready for the
        classification head. ``gate`` is the per-row weighting tensor,
        useful for inference-time diagnostics ("which modality drove
        this prediction").
        """

        if market_pooled.dim() != 2 or text_pooled.dim() != 2:
            raise ValueError(
                "GatedInfoNCEFusion expects 2-D (B, D) tensors; "
                f"got market={market_pooled.shape}, text={text_pooled.shape}"
            )
        if market_pooled.shape[-1] != self.market_dim:
            raise ValueError(
                f"market_pooled last-dim must be {self.market_dim}; "
                f"got {market_pooled.shape[-1]}"
            )
        if text_pooled.shape[-1] != self.text_dim:
            raise ValueError(
                f"text_pooled last-dim must be {self.text_dim}; "
                f"got {text_pooled.shape[-1]}"
            )
        if market_pooled.shape[0] != text_pooled.shape[0]:
            raise ValueError(
                "market and text batches must have the same length; "
                f"got {market_pooled.shape[0]} vs {text_pooled.shape[0]}"
            )

        r_t = self.market_proj(self.dropout_layer(market_pooled))
        t_t = self.text_proj(self.dropout_layer(text_pooled))
        gate = torch.sigmoid(
            self.gate(torch.cat([market_pooled, text_pooled], dim=-1))
        )
        fused = gate * r_t + (1.0 - gate) * t_t
        return {"r_t": r_t, "t_t": t_t, "fused": fused, "gate": gate}
