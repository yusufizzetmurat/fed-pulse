"""Multi-task loss for the three-branch classification head (#78).

The three axes (stance, factor, certainty) have very different
label-coverage rates on the supervised corpus: stance ~100%, factor
and certainty <5% (gtfintechlab cross-bank rows + the gss_factor
source only). The loss handles this with per-axis masks — a row only
contributes loss on axes where its label was populated, so the
optimiser never learns from a synthetic placeholder that would
otherwise lock the head in on a meaningless mean. The topic axis was
retired in ADR 0044 (no upstream source ships topic labels).

The total loss is a lambda-weighted sum:

    L = lambda_stance * stance_loss
      + lambda_factor * factor_loss
      + lambda_certainty * certainty_loss

Each per-axis term is the mean of the row-wise loss over rows where
that axis's mask is True (0 when the mask is empty for that batch,
which avoids div-by-zero). Lambdas default to (1.0, 0.3, 0.3) so
the headline stance F1 stays the dominant gradient signal; the
sparser axes contribute auxiliary gradients without overpowering
stance.

Class weights are passed per axis. Stance reuses the per-fold
``fit_class_weights`` output; certainty class weights are fitted on
the train slice independently using the same helper. The factor
branch is a regression (SmoothL1, ``beta=0.02`` to mirror the
existing vol-regression loss) so it does not consume a class-weight
tensor.

Regime-axis loss variant (#470). The 3-class vol-regime stance head
defaults to standard cross-entropy. When ``regime_loss_mode='ordinal_ce'``
is wired through the training loop, the stance branch instead trains
under :func:`ordinal_cross_entropy` — bin-distance-weighted CE where
each masked-in row's contribution is scaled by ``|true - argmax|``.
A ``calm -> high`` confusion therefore costs 2x a ``calm -> normal``
confusion, encoding the ordinal structure of the bucketed vol-regime
labels (``calm < normal < high``) into the loss without changing the
head architecture or introducing distribution-output complexity.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def ordinal_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Bin-distance-weighted cross-entropy for an ordinal classifier.

    For each row, the standard per-row CE is multiplied by
    ``1 + |target - argmax(logits)|`` so an adjacent-bin miss pays
    the unweighted CE and a far-bin miss pays proportionally more. A
    correct row (``argmax == target``) keeps its CE unchanged (the
    ``1 +`` floor avoids zeroing the gradient on already-correct rows).

    Picked over richer ordinal losses (CORN, soft-label smoothing,
    earth-mover) because it preserves the ``Linear(n_classes)`` head
    + integer-target contract the existing checkpoint shape locks in,
    introduces zero new hyperparameters, and routes the gradient
    asymmetrically through ``|true - pred|`` without rebuilding the
    output distribution. The ``argmax`` cost is a non-differentiable
    scaling factor (no gradient flows through it), so the variant
    behaves like a re-weighted CE under autograd — the gradient
    direction matches standard CE and only the magnitude is
    distance-scaled.

    ``weight`` is the optional per-class weight vector (passed through
    to :func:`torch.nn.functional.cross_entropy`). ``reduction='mean'``
    returns the mean over rows; ``'none'`` returns the per-row tensor
    so the caller (e.g. the sample-weighted multi-task path) can fold
    in its own weighting.
    """

    per_row_ce = F.cross_entropy(logits, target, weight=weight, reduction="none")
    distance = (logits.detach().argmax(dim=-1) - target).abs().to(per_row_ce.dtype)
    per_row = per_row_ce * (1.0 + distance)
    if reduction == "none":
        return per_row
    if reduction == "sum":
        return per_row.sum()
    return per_row.mean()


class MultiTaskLoss(nn.Module):
    """Per-axis weighted + masked loss for the multi-task head.

    Reads logits and targets from dicts keyed by axis name and applies
    the corresponding per-axis loss (CE for stance / certainty;
    SmoothL1 for factor). Class weights are optional per axis — None
    gives uniform weighting on that axis.

    ``regime_loss_mode`` selects the stance-axis loss kernel: ``'ce'``
    (default) keeps the unchanged cross-entropy path so every existing
    multi-task run reproduces byte-identically. ``'ordinal_ce'`` swaps
    in :func:`ordinal_cross_entropy` so the 3-class regime stance head
    pays distance-weighted CE — picked when the regime label space
    carries ordinal structure (``calm < normal < high``).
    """

    def __init__(  # noqa: PLR0913 — per-axis class weights + lambdas surface as named kwargs by design
        self,
        *,
        stance_weight: torch.Tensor | None = None,
        certainty_weight: torch.Tensor | None = None,
        lambda_stance: float = 1.0,
        lambda_factor: float = 0.3,
        lambda_certainty: float = 0.3,
        factor_smooth_l1_beta: float = 0.02,
        regime_loss_mode: str = "ce",
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
        self.lambda_stance = float(lambda_stance)
        self.lambda_factor = float(lambda_factor)
        self.lambda_certainty = float(lambda_certainty)
        self.factor_smooth_l1_beta = float(factor_smooth_l1_beta)
        if regime_loss_mode not in {"ce", "ordinal_ce"}:
            raise ValueError(
                f"regime_loss_mode must be 'ce' or 'ordinal_ce'; got {regime_loss_mode!r}"
            )
        self.regime_loss_mode = str(regime_loss_mode)

    def _stance_weight_or_none(self) -> torch.Tensor | None:
        buf = self.get_buffer("_stance_weight")
        return buf if buf.numel() > 0 else None

    def _certainty_weight_or_none(self) -> torch.Tensor | None:
        buf = self.get_buffer("_certainty_weight")
        return buf if buf.numel() > 0 else None

    def forward(
        self,
        logits: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        *,
        stance_sample_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(total_loss, per_axis_breakdown)``.

        ``per_axis_breakdown`` is a detached dict of scalar tensors
        keyed by axis name so the training-loop logger can record per-
        axis loss trajectories. Axes whose mask is all-False contribute
        zero loss to the total and emit a zero-valued breakdown entry.

        ``stance_sample_weight`` is an optional per-row float vector
        with the same length as the batch. When supplied, the stance
        branch becomes a weighted-mean CE where each masked-in row's
        contribution is scaled by its weight. When ``None`` (the
        default), the stance loss is the unweighted masked mean — the
        path every existing call site reaches today. The cross-bank
        supervision arm (``--cross-bank-supervision weighted``) is the
        only caller that passes a non-None vector; FOMC-only training
        runs leave it as ``None`` and reproduce the prior numerics
        byte-identically.
        """

        device = logits["stance"].device
        zero = torch.zeros((), device=device)

        stance_loss = self._masked_classification_loss(
            logits["stance"],
            targets["stance"],
            masks["stance_mask"],
            weight=self._stance_weight_or_none(),
            sample_weight=stance_sample_weight,
            loss_mode=self.regime_loss_mode,
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

        total = (
            self.lambda_stance * stance_loss
            + self.lambda_factor * factor_loss
            + self.lambda_certainty * certainty_loss
        )
        if not total.requires_grad and stance_loss.requires_grad:
            total = total + zero
        return total, {
            "stance": stance_loss.detach(),
            "factor": factor_loss.detach(),
            "certainty": certainty_loss.detach(),
        }

    @staticmethod
    def _masked_classification_loss(  # noqa: PLR0913 — per-axis kwargs surface by design
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        *,
        weight: torch.Tensor | None,
        sample_weight: torch.Tensor | None = None,
        loss_mode: str = "ce",
    ) -> torch.Tensor:
        """Mean CE loss over masked rows; zero when no rows are masked.

        ``F.cross_entropy`` would happily train against the placeholder
        class indices on masked-out rows; instead we slice the logits
        and targets by the boolean mask before computing the loss.
        Returns a graph-attached zero (``logits.sum() * 0``) when no
        rows survive so the optimiser still sees a tensor on the
        same device with the same gradient connectivity.

        ``sample_weight`` (optional per-row float vector, same length
        as ``mask``) scales each masked-in row's CE contribution. The
        return is a weighted mean: ``sum(w_i * ce_i) / sum(w_i)`` over
        masked-in rows. When the total weight collapses to zero (an
        all-cross-bank batch under ``weighted`` with weight=0.0), the
        result is the same graph-attached zero used for the empty-mask
        case so backward stays well-defined and contributes no
        gradient through the stance head.

        ``loss_mode='ordinal_ce'`` routes the per-row CE through
        :func:`ordinal_cross_entropy` so the row weight bakes in
        ``1 + |target - argmax|``; ``'ce'`` (the default) is the
        standard cross-entropy path the certainty branch always uses.
        """

        if mask.numel() == 0 or not mask.any():
            return logits.sum() * 0.0
        active_logits = logits[mask]
        active_target = target[mask]
        if sample_weight is None:
            if loss_mode == "ordinal_ce":
                return ordinal_cross_entropy(
                    active_logits, active_target, weight=weight, reduction="mean"
                )
            return F.cross_entropy(active_logits, active_target, weight=weight)
        active_sample_weight = sample_weight[mask]
        if loss_mode == "ordinal_ce":
            per_row = ordinal_cross_entropy(
                active_logits, active_target, weight=weight, reduction="none"
            )
        else:
            per_row = F.cross_entropy(
                active_logits, active_target, weight=weight, reduction="none"
            )
        weight_total = active_sample_weight.sum()
        # ``> 0`` keeps the zero-weight collapse on the same graph-
        # attached zero path the empty-mask branch uses; otherwise the
        # ``/ weight_total`` divisor would produce NaN gradients.
        if float(weight_total.detach().item()) <= 0.0:
            return logits.sum() * 0.0
        return (per_row * active_sample_weight).sum() / weight_total

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
