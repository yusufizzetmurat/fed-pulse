"""Multi-task loss for the three-branch classification head (#78).

The three axes (stance, certainty, time) have very different
label-coverage rates on the supervised corpus: stance ~100%, certainty
and time from the gtfintechlab cross-bank rows. The loss handles this
with per-axis masks — a row only contributes loss on axes where its
label was populated, so the optimiser never learns from a synthetic
placeholder that would otherwise lock the head in on a meaningless
mean. The factor axis was retired (market-derived GSS regression target
text cannot predict; 0% pool coverage). The topic axis was retired in
ADR 0044 (no upstream source ships topic labels).

The total loss is a lambda-weighted sum:

    L = lambda_stance * stance_loss
      + lambda_certainty * certainty_loss
      + lambda_time * time_loss

Each per-axis term is the mean of the row-wise loss over rows where
that axis's mask is True (0 when the mask is empty for that batch,
which avoids div-by-zero). Lambdas default to (1.0, 0.3, 0.3) so
the headline stance F1 stays the dominant gradient signal; the
sparser axes contribute auxiliary gradients without overpowering
stance.

Class weights are passed per axis. Stance reuses the per-fold
``fit_class_weights`` output; certainty and time class weights are
fitted on the train slice independently using the same helper. All
three axes are classification (CrossEntropy) heads.

Regime-axis loss variant (#470). The 3-class vol-regime stance head
defaults to standard cross-entropy. When ``regime_loss_mode='ordinal_ce'``
is wired through the training loop, the stance branch instead trains
under :func:`ordinal_cross_entropy` — bin-distance-weighted CE where
each masked-in row's contribution is scaled by ``|true - argmax|``.
A ``calm -> high`` confusion therefore costs 2x a ``calm -> normal``
confusion, encoding the ordinal structure of the bucketed vol-regime
labels (``calm < normal < high``) into the loss without changing the
head architecture or introducing distribution-output complexity.

Regime-axis loss extensions (#502). Two additional kernels round out
the imbalance-focused options. ``focal`` (Lin et al. 2017) multiplies
each row's CE by ``(1 - p_true) ** gamma`` so confident wrong
predictions dominate the gradient; the per-class weight tensor (when
present) composes multiplicatively. ``class_balanced`` (Cui et al.
2019) replaces the inverse-frequency class weight with the
effective-number weight ``(1 - beta) / (1 - beta ** n_c)`` — the
per-class counts come from the same train-slice machinery the standard
CE path already uses, only the weight derivation changes.
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


def focal_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    gamma: float = 2.0,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss (Lin et al. 2017): ``(1 - p_true) ** gamma`` weighted CE.

    ``gamma`` is the focusing parameter (literature default 2.0); a row
    where the model already assigns near-1 probability to the true class
    contributes essentially no loss, while a confident wrong prediction
    pays close to the unweighted CE. ``weight`` (optional per-class
    vector) composes multiplicatively with the per-sample focal weight
    so a rare-class miss is amplified by both factors.

    ``reduction='mean'`` returns the mean over rows; ``'none'`` returns
    the per-row tensor so the caller can fold in its own sample
    weighting.
    """

    log_probs = F.log_softmax(logits, dim=-1)
    target_expanded = target.unsqueeze(-1)
    log_p_true = log_probs.gather(-1, target_expanded).squeeze(-1)
    p_true = log_p_true.exp()
    # Detach the focal modulating factor — Lin et al. 2017 + the reference
    # RetinaNet implementation treat ``(1 - p_t) ** gamma`` as a fixed
    # per-sample weight so the gradient flows only through ``-log p_true``.
    # Keeping it in-graph introduces an extra term that destabilises
    # training on small batches (the FOMC corpus is the use case here).
    focal_weight = (1.0 - p_true).clamp(min=0.0).pow(float(gamma)).detach()
    per_row = focal_weight * (-log_p_true)
    if weight is not None:
        per_row = per_row * weight[target]
    if reduction == "none":
        return per_row
    if reduction == "sum":
        return per_row.sum()
    # Match ``F.cross_entropy(..., weight=..., reduction='mean')``
    # semantics: when class weights are present, normalise by their sum
    # over the targets so the loss scale is comparable to the unweighted
    # path, not by raw row count.
    if weight is not None:
        return per_row.sum() / weight[target].sum()
    return per_row.mean()


def class_balanced_weights(
    class_counts: torch.Tensor | list[int] | tuple[int, ...],
    *,
    beta: float = 0.999,
) -> torch.Tensor:
    """Effective-number class weights (Cui et al. 2019).

    Per-class weight is ``(1 - beta) / (1 - beta ** n_c)`` where ``n_c``
    is the per-class sample count on the train slice. Normalised so the
    weights sum to ``n_classes`` to match the convention used by
    :func:`app.training.loaders.fit_class_weights` (inverse-frequency
    path) and keep the loss magnitude comparable across modes.

    Classes with zero samples fall back to the value the limit
    ``(1 - beta) / (1 - beta ** n) -> 1 / n`` produces on a count of 1,
    so an empty class still receives a finite, well-behaved weight.
    """

    if isinstance(class_counts, torch.Tensor):
        counts = class_counts.detach().to(dtype=torch.float64, device="cpu")
    else:
        counts = torch.tensor(
            [float(c) for c in class_counts], dtype=torch.float64
        )
    if counts.numel() == 0:
        return torch.zeros(0, dtype=torch.float32)
    beta_v = float(beta)
    # Empty classes are floored to a count of 1 so the denominator
    # ``1 - beta ** n_c`` stays strictly positive.
    safe_counts = torch.where(counts > 0, counts, torch.ones_like(counts))
    effective_num = 1.0 - torch.pow(torch.tensor(beta_v, dtype=torch.float64), safe_counts)
    raw = (1.0 - beta_v) / effective_num
    total = raw.sum()
    if float(total.item()) <= 0.0:
        return torch.ones(counts.numel(), dtype=torch.float32)
    normalised = raw / total * float(counts.numel())
    return normalised.to(dtype=torch.float32)


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
    carries ordinal structure (``calm < normal < high``). ``'focal'``
    routes through :func:`focal_cross_entropy` with the configured
    ``focal_gamma`` so confident-wrong rows dominate. ``'class_balanced'``
    behaves like ``'ce'`` at the kernel level — the per-class effective-
    number weight is built upstream (see :func:`class_balanced_weights`)
    and passed in via ``stance_weight``.
    """

    _SUPPORTED_REGIME_LOSS_MODES = frozenset({"ce", "ordinal_ce", "focal", "class_balanced"})

    def __init__(  # noqa: PLR0913 — per-axis class weights + lambdas surface as named kwargs by design
        self,
        *,
        stance_weight: torch.Tensor | None = None,
        certainty_weight: torch.Tensor | None = None,
        time_weight: torch.Tensor | None = None,
        lambda_stance: float = 1.0,
        lambda_certainty: float = 0.3,
        lambda_time: float = 0.3,
        regime_loss_mode: str = "ce",
        focal_gamma: float = 2.0,
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
            "_time_weight",
            time_weight if time_weight is not None else torch.empty(0),
        )
        self.lambda_stance = float(lambda_stance)
        self.lambda_certainty = float(lambda_certainty)
        self.lambda_time = float(lambda_time)
        if regime_loss_mode not in self._SUPPORTED_REGIME_LOSS_MODES:
            raise ValueError(
                "regime_loss_mode must be one of "
                f"{sorted(self._SUPPORTED_REGIME_LOSS_MODES)}; got {regime_loss_mode!r}"
            )
        self.regime_loss_mode = str(regime_loss_mode)
        self.focal_gamma = float(focal_gamma)

    def _stance_weight_or_none(self) -> torch.Tensor | None:
        buf = self.get_buffer("_stance_weight")
        return buf if buf.numel() > 0 else None

    def _certainty_weight_or_none(self) -> torch.Tensor | None:
        buf = self.get_buffer("_certainty_weight")
        return buf if buf.numel() > 0 else None

    def _time_weight_or_none(self) -> torch.Tensor | None:
        buf = self.get_buffer("_time_weight")
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
            focal_gamma=self.focal_gamma,
        )
        certainty_loss = self._masked_classification_loss(
            logits["certainty"],
            targets["certainty"],
            masks["certainty_mask"],
            weight=self._certainty_weight_or_none(),
        )
        time_loss = self._masked_classification_loss(
            logits["time"],
            targets["time"],
            masks["time_mask"],
            weight=self._time_weight_or_none(),
        )

        total = (
            self.lambda_stance * stance_loss
            + self.lambda_certainty * certainty_loss
            + self.lambda_time * time_loss
        )
        if not total.requires_grad and stance_loss.requires_grad:
            total = total + zero
        return total, {
            "stance": stance_loss.detach(),
            "certainty": certainty_loss.detach(),
            "time": time_loss.detach(),
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
        focal_gamma: float = 2.0,
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

        ``loss_mode`` selects the per-row kernel: ``'ordinal_ce'`` routes
        through :func:`ordinal_cross_entropy`; ``'focal'`` routes through
        :func:`focal_cross_entropy` with the configured ``focal_gamma``;
        ``'ce'`` and ``'class_balanced'`` both use the standard
        :func:`F.cross_entropy` path (the class-balanced weighting is
        baked into the ``weight`` tensor upstream so the kernel itself
        stays vanilla CE).
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
            if loss_mode == "focal":
                return focal_cross_entropy(
                    active_logits,
                    active_target,
                    gamma=focal_gamma,
                    weight=weight,
                    reduction="mean",
                )
            return F.cross_entropy(active_logits, active_target, weight=weight)
        active_sample_weight = sample_weight[mask]
        if loss_mode == "ordinal_ce":
            per_row = ordinal_cross_entropy(
                active_logits, active_target, weight=weight, reduction="none"
            )
        elif loss_mode == "focal":
            per_row = focal_cross_entropy(
                active_logits,
                active_target,
                gamma=focal_gamma,
                weight=weight,
                reduction="none",
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
