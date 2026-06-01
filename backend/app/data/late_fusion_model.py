"""Clean-room late-fusion model for the rebuild.

Independent of the suspect ``MultiModalForecasterModel``. A text branch and a
structured (market + SEP) branch are encoded separately and fused LATE (concat
at the penultimate layer), feeding two proper heads: direction (BCE logit) and
magnitude (softplus regression). The same class serves the experiment's three
configurations via the ``use_text`` / ``use_struct`` flags:

* full late fusion  -> use_text=True,  use_struct=True
* market-only        -> use_text=False, use_struct=True
* text-only          -> use_text=True,  use_struct=False

``assert_text_gradient_flows`` is the fault-class-#3 check: after one backward
pass the text branch's parameters must receive non-zero gradients, proving the
text signal is actually wired into the loss (not detached or masked away).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


class LateFusionModel(nn.Module):
    def __init__(
        self,
        text_dim: int,
        struct_dim: int,
        text_latent: int = 16,
        struct_latent: int = 16,
        trunk_dim: int = 32,
        dropout: float = 0.3,
        use_text: bool = True,
        use_struct: bool = True,
    ) -> None:
        super().__init__()
        if not (use_text or use_struct):
            raise ValueError("at least one of use_text / use_struct must be True")
        self.use_text = use_text
        self.use_struct = use_struct

        fused = 0
        if use_text:
            self.text_branch = nn.Sequential(
                nn.Linear(text_dim, text_latent), nn.GELU(), nn.Dropout(dropout)
            )
            fused += text_latent
        if use_struct:
            self.struct_branch = nn.Sequential(
                nn.Linear(struct_dim, struct_latent), nn.GELU(), nn.Dropout(dropout)
            )
            fused += struct_latent

        self.trunk = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, trunk_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dir_head = nn.Linear(trunk_dim, 1)
        self.mag_head = nn.Linear(trunk_dim, 1)

    def forward(
        self, text: torch.Tensor, struct: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parts: list[torch.Tensor] = []
        if self.use_text:
            parts.append(self.text_branch(text))
        if self.use_struct:
            parts.append(self.struct_branch(struct))
        fused = torch.cat(parts, dim=-1)
        hidden = self.trunk(fused)
        dir_logit = self.dir_head(hidden).squeeze(-1)
        # Linear magnitude output: the experiment standardizes the magnitude target
        # per fold, so the head predicts a z-scored value (a softplus would clamp
        # the scale and cannot match tiny raw |return| targets).
        magnitude = self.mag_head(hidden).squeeze(-1)
        return dir_logit, magnitude


def joint_loss(
    dir_logit: torch.Tensor,
    magnitude: torch.Tensor,
    dir_target: torch.Tensor,
    mag_target: torch.Tensor,
    mag_weight: float = 1.0,
) -> torch.Tensor:
    """BCE on direction + Huber on magnitude (both heads trained jointly)."""
    dir_loss = F.binary_cross_entropy_with_logits(dir_logit, dir_target)
    mag_loss = F.smooth_l1_loss(magnitude, mag_target)
    return dir_loss + mag_weight * mag_loss


def assert_text_gradient_flows(
    model: LateFusionModel, text: torch.Tensor, struct: torch.Tensor
) -> float:
    """Fault #3 gate: confirm the text branch receives non-zero gradients.

    Returns the summed absolute gradient of the first text-branch weight. Raises
    if the text branch is absent or its gradient is zero/None (which would mean
    text is detached from the loss).
    """
    if not model.use_text:
        raise ValueError("model has no text branch to check")
    model.zero_grad()
    dir_logit, magnitude = model(text, struct)
    dir_target = (torch.rand_like(dir_logit) > 0.5).float()
    mag_target = magnitude.detach().abs() + 0.1
    loss = joint_loss(dir_logit, magnitude, dir_target, mag_target)
    loss.backward()  # type: ignore[no-untyped-call]
    grad = next(model.text_branch.parameters()).grad
    if grad is None:
        raise AssertionError("text branch weight has no gradient (text detached)")
    total = float(grad.abs().sum().item())
    if total == 0.0:
        raise AssertionError("text branch gradient is exactly zero (text detached)")
    return total
