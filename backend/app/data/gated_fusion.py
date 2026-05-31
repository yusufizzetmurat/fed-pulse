"""Gated text↔market fusion forecaster with an InfoNCE alignment objective.

The model fuses a (precomputed) Fed-communication text embedding with a market
feature vector through a learned scalar **gate**, then forecasts forward
realized variance at several horizons. The gate is *floored*: on days with no
fresh communication the gate is forced to zero, so the model collapses to the
pure market path — it can never do worse than market-only by leaning on absent
text. The gate value is itself an interpretable readout of "how much did text
matter here."

Alongside the supervised forecast loss, an **InfoNCE** term aligns each
communication's text embedding with an encoder of its realized forward-RV
outcome (CLIP-style symmetric contrastive loss with in-batch negatives),
computed only over rows that carry text. Text encoding is done upstream and
cached; this module consumes embeddings so training stays fast and the encoder
choice is swappable.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn


class GatedFusionForecaster(nn.Module):
    """Market baseline + gated text contribution → multi-horizon RV head."""

    def __init__(
        self,
        d_text: int,
        d_market: int,
        n_horizons: int,
        *,
        d_hidden: int = 128,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.text_proj = nn.Sequential(
            nn.Linear(d_text, d_hidden), nn.GELU(), nn.LayerNorm(d_hidden)
        )
        self.market_enc = nn.Sequential(
            nn.Linear(d_market, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
        )
        # scalar gate from [text rep, market rep, text-present flag]
        self.gate_net = nn.Linear(2 * d_hidden + 1, 1)
        self.head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Linear(d_hidden, n_horizons)
        )
        # outcome encoder used only for the InfoNCE alignment
        self.outcome_enc = nn.Sequential(
            nn.Linear(n_horizons, d_hidden), nn.GELU(), nn.LayerNorm(d_hidden)
        )

    def forward(
        self, text_emb: torch.Tensor, market_feat: torch.Tensor, text_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        zt = self.text_proj(text_emb)
        zm = self.market_enc(market_feat)
        mask = text_mask.float().unsqueeze(1)
        gate = torch.sigmoid(self.gate_net(torch.cat([zt, zm, mask], dim=1)))
        gate = gate * mask  # FLOOR: no fresh text → gate 0 → pure market path
        fused = zm + gate * zt
        pred = self.head(fused)
        return {"pred": pred, "gate": gate.squeeze(1), "z_text": zt}

    def encode_outcome(self, targets: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.outcome_enc(targets))


def info_nce_loss(
    z_text: torch.Tensor,
    z_outcome: torch.Tensor,
    text_mask: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric CLIP-style contrastive loss over text-present rows only."""

    keep = text_mask.bool()
    if int(keep.sum()) < 2:  # need ≥2 positives to contrast
        return z_text.new_zeros(())
    a = torch.nn.functional.normalize(z_text[keep], dim=1)
    b = torch.nn.functional.normalize(z_outcome[keep], dim=1)
    logits = a @ b.t() / temperature
    labels = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (
        torch.nn.functional.cross_entropy(logits, labels)
        + torch.nn.functional.cross_entropy(logits.t(), labels)
    )


def fusion_loss(
    model: GatedFusionForecaster,
    batch: dict[str, torch.Tensor],
    *,
    info_nce_weight: float = 0.1,
    huber_delta: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Supervised Huber forecast loss + λ·InfoNCE alignment.

    `batch` keys: text_emb (B,d_text), market_feat (B,d_market), text_mask (B,),
    targets (B,n_horizons). Targets are the (HAR-residual or raw) forward log-RV.
    """

    out = model(batch["text_emb"], batch["market_feat"], batch["text_mask"])
    sup = torch.nn.functional.huber_loss(out["pred"], batch["targets"], delta=huber_delta)
    z_out = model.encode_outcome(batch["targets"])
    nce = info_nce_loss(out["z_text"], z_out, batch["text_mask"], temperature=model.temperature)
    total = sup + info_nce_weight * nce
    return {"loss": total, "supervised": sup, "info_nce": nce, "gate": out["gate"]}


def build_model(
    d_text: int, d_market: int, n_horizons: int, **kwargs: Any
) -> GatedFusionForecaster:
    return GatedFusionForecaster(d_text, d_market, n_horizons, **kwargs)
