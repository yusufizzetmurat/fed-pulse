"""Multi-task loss for the four-branch classification head (#78).

The four axes (stance, factor, certainty, topic) have very different
label-coverage rates on the supervised corpus: stance ~100%, factor
and certainty <5% (gtfintechlab cross-bank rows + the gss_factor
source only), topic 0% upstream. The loss handles this with per-axis
masks — a row only contributes loss on axes where its label was
populated, so the optimiser never learns from a synthetic placeholder
that would otherwise lock the head in on a meaningless mean.

The total loss is a lambda-weighted sum:

    L = lambda_stance * stance_loss
      + lambda_factor * factor_loss
      + lambda_certainty * certainty_loss
      + lambda_topic * topic_loss

Each per-axis term is the mean of the row-wise loss over rows where
that axis's mask is True (0 when the mask is empty for that batch,
which avoids div-by-zero). Lambdas default to (1.0, 0.3, 0.3, 0.3) so
the headline stance F1 stays the dominant gradient signal; the
sparser axes contribute auxiliary gradients without overpowering
stance.

Class weights are passed per axis. Stance reuses the per-fold
``fit_class_weights`` output; certainty / topic class weights are
fitted on the train slice independently using the same helper. The
factor branch is a regression (SmoothL1, ``beta=0.02`` to mirror the
existing vol-regression loss) so it does not consume a class-weight
tensor.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MultiTaskLoss(nn.Module):
    """Per-axis weighted + masked loss for the multi-task head.

    Reads logits and targets from dicts keyed by axis name and applies
    the corresponding per-axis loss (CE for stance / certainty /
    topic; SmoothL1 for factor). Class weights are optional per axis
    — None gives uniform weighting on that axis.
    """

    def __init__(  # noqa: PLR0913 — per-axis class weights + lambdas surface as named kwargs by design
        self,
        *,
        stance_weight: torch.Tensor | None = None,
        certainty_weight: torch.Tensor | None = None,
        topic_weight: torch.Tensor | None = None,
        lambda_stance: float = 1.0,
        lambda_factor: float = 0.3,
        lambda_certainty: float = 0.3,
        lambda_topic: float = 0.3,
        factor_smooth_l1_beta: float = 0.02,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "_stance_weight",
            stance_weight if stance_weight is not None else torch.empty(0),
        )
        self.register_buffer(
            "_certainty_weight",
            certainty_weight if certainty_weight is not None else torch.empty(0),
        )
        self.register_buffer(
            "_topic_weight",
            topic_weight if topic_weight is not None else torch.empty(0),
        )
        self.lambda_stance = float(lambda_stance)
        self.lambda_factor = float(lambda_factor)
        self.lambda_certainty = float(lambda_certainty)
        self.lambda_topic = float(lambda_topic)
        self.factor_smooth_l1_beta = float(factor_smooth_l1_beta)

    def _stance_weight_or_none(self) -> torch.Tensor | None:
        return self._stance_weight if self._stance_weight.numel() > 0 else None

    def _certainty_weight_or_none(self) -> torch.Tensor | None:
        return self._certainty_weight if self._certainty_weight.numel() > 0 else None

    def _topic_weight_or_none(self) -> torch.Tensor | None:
        return self._topic_weight if self._topic_weight.numel() > 0 else None

    def forward(
        self,
        logits: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(total_loss, per_axis_breakdown)``.

        ``per_axis_breakdown`` is a detached dict of scalar tensors
        keyed by axis name so the training-loop logger can record per-
        axis loss trajectories. Axes whose mask is all-False contribute
        zero loss to the total and emit a zero-valued breakdown entry.
        """

        device = logits["stance"].device
        zero = torch.zeros((), device=device)

        stance_loss = self._masked_classification_loss(
            logits["stance"],
            targets["stance"],
            masks["stance_mask"],
            weight=self._stance_weight_or_none(),
        )
        factor_loss = self._masked_regression_loss(
            logits["factor"],
            targets["factor"],
            masks["factor_mask"],
        )
        certainty_loss = self._masked_classification_loss(
            logits["certainty"],
            targets["certainty"],
            masks["certainty_mask"],
            weight=self._certainty_weight_or_none(),
        )
        topic_loss = self._masked_classification_loss(
            logits["topic"],
            targets["topic"],
            masks["topic_mask"],
            weight=self._topic_weight_or_none(),
        )

        total = (
            self.lambda_stance * stance_loss
            + self.lambda_factor * factor_loss
            + self.lambda_certainty * certainty_loss
            + self.lambda_topic * topic_loss
        )
        if not total.requires_grad and stance_loss.requires_grad:
            total = total + zero
        return total, {
            "stance": stance_loss.detach(),
            "factor": factor_loss.detach(),
            "certainty": certainty_loss.detach(),
            "topic": topic_loss.detach(),
        }

    @staticmethod
    def _masked_classification_loss(
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        *,
        weight: torch.Tensor | None,
    ) -> torch.Tensor:
        """Mean CE loss over masked rows; zero when no rows are masked.

        ``F.cross_entropy`` would happily train against the placeholder
        class indices on masked-out rows; instead we slice the logits
        and targets by the boolean mask before computing the loss.
        Returns a graph-attached zero (``logits.sum() * 0``) when no
        rows survive so the optimiser still sees a tensor on the
        same device with the same gradient connectivity.
        """

        if mask.numel() == 0 or not mask.any():
            return logits.sum() * 0.0
        active_logits = logits[mask]
        active_target = target[mask]
        return F.cross_entropy(active_logits, active_target, weight=weight)

    def _masked_regression_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if mask.numel() == 0 or not mask.any():
            return pred.sum() * 0.0
        active_pred = pred[mask]
        active_target = target[mask]
        return F.smooth_l1_loss(
            active_pred, active_target, beta=self.factor_smooth_l1_beta
        )
