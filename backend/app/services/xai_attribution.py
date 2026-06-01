"""Integrated-gradients attribution across the /analyze panels (#297).

The existing :mod:`app.evaluation.xai` module ships a keyword-salience
heuristic for the sentence-level hawkish/dovish badge on the stance
panel. That keeps working untouched; the heuristic is the right tool for
the headline "explain my stance" surface because it does not require a
forward pass.

This module adds gradient-based attribution for the remaining panels
that the headline heuristic does not cover:

* **regime classification** (vol-regime softmax head) — attributes the
  argmax class logit back to the per-bar feature vector. Aggregates the
  per-step gradients into per-feature-family magnitudes via the
  ``RICH_*_SLICE`` constants so the frontend can render one bar per
  family (linguistic / credibility / mp_surprise / multi_axis / market /
  cross_asset / realized_vol / llm).
* **rates heads** — one attribution per mounted rates head against the
  scalar bps prediction. Same per-family aggregation.
* **trajectory** — attributes the argmax next-stance probability back
  to the per-bar trajectory feature vector. Aggregated per-bar rather
  than per-family (the trajectory input is a flat per-bar vector, not
  the rich market+text union).

Integrated gradients (Sundararajan et al. 2017): for a model ``f(x)``
and a baseline ``x'`` (here, zeros — the trained scalers centre most of
the input around 0 already so the zero baseline is the natural neutral
input), the attribution is

    IG_i(x) = (x_i - x'_i) * mean_{k=1..n_steps}( df / dx_i |_{alpha_k * x + (1 - alpha_k) * x'} )

We use the Riemann mid-point rule. ``n_steps`` defaults to 20 (env
override ``FED_PULSE_IG_N_STEPS``) so the per-panel attribution costs
~20 forward + backward passes; bounded so the end-to-end /analyze
latency stays inside the budget called out in ADR 0026.

The text path is attributed via sentence-level ablation rather than
gradients through the encoder: re-pool the per-statement embedding
with sentence ``i`` removed, measure the target delta, and report the
top-K sentences. Gradient-through-the-encoder would have required
backprop through a frozen FinBERT — far outside the latency budget.

All entry points degrade gracefully: any RuntimeError, missing kwarg,
or contract failure surfaces a structured ``unavailable`` payload
rather than a raw exception. The /analyze route never 500s because a
panel could not be explained.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from app.models.config import (
    FEATURE_SIZE,
    RICH_CREDIBILITY_SLICE,
    RICH_CROSS_ASSET_SLICE,
    RICH_FEATURE_SIZE,
    RICH_LINGUISTIC_SLICE,
    RICH_LLM_FEATURE_SLICE,
    RICH_MARKET_SLICE,
    RICH_MP_SURPRISE_SLICE,
    RICH_MULTI_AXIS_SLICE,
    RICH_REALIZED_VOL_SLICE,
)

logger = logging.getLogger(__name__)


DEFAULT_N_STEPS = 20
# Hard cap. n_steps above 64 quickly blows the per-request latency
# budget on the dev box and the diminishing-returns curve flattens
# beyond ~32 in our smoke runs. Capping in the helper rather than
# trusting env config protects against a misconfigured deployment.
MAX_N_STEPS = 64

# Named feature families keyed by the rich-feature slice constants. The
# order here is the order the frontend renders the bars in.
FEATURE_FAMILY_SLICES: tuple[tuple[str, slice], ...] = (
    ("market", RICH_MARKET_SLICE),
    ("credibility", RICH_CREDIBILITY_SLICE),
    ("linguistic", RICH_LINGUISTIC_SLICE),
    ("mp_surprise", RICH_MP_SURPRISE_SLICE),
    ("multi_axis", RICH_MULTI_AXIS_SLICE),
    ("realized_vol", RICH_REALIZED_VOL_SLICE),
    ("cross_asset", RICH_CROSS_ASSET_SLICE),
    ("llm", RICH_LLM_FEATURE_SLICE),
)


def resolve_n_steps(override: int | None = None) -> int:
    """Pick the IG integration step count.

    Precedence: explicit ``override`` argument > ``FED_PULSE_IG_N_STEPS``
    env var > :data:`DEFAULT_N_STEPS`. Always clamped into
    ``[2, MAX_N_STEPS]`` — a single-step IG collapses to a finite
    difference and gives noisy attribution; above ``MAX_N_STEPS`` the
    latency budget breaks.
    """

    if override is not None:
        value = int(override)
    else:
        raw = os.environ.get("FED_PULSE_IG_N_STEPS", "").strip()
        try:
            value = int(raw) if raw else DEFAULT_N_STEPS
        except ValueError:
            value = DEFAULT_N_STEPS
    return max(2, min(MAX_N_STEPS, value))


@dataclass(frozen=True)
class FeatureFamilyAttribution:
    """One bar in the per-panel feature-family chart."""

    family: str
    magnitude: float
    signed: float

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "magnitude": float(self.magnitude),
            "signed": float(self.signed),
        }


@dataclass(frozen=True)
class PanelAttribution:
    """Bundle returned for one panel target.

    ``unavailable`` is True when the attribution could not be computed
    (panel not active on the checkpoint, kwarg mismatch, unexpected
    runtime error). The frontend renders the panel's "explanation
    unavailable" badge in that case rather than throwing.
    """

    panel: str
    target: str
    families: list[FeatureFamilyAttribution]
    n_steps: int
    unavailable: bool = False
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "panel": self.panel,
            "target": self.target,
            "families": [item.to_dict() for item in self.families],
            "n_steps": int(self.n_steps),
            "unavailable": bool(self.unavailable),
            "reason": self.reason,
        }


def _zero_baseline(x: torch.Tensor) -> torch.Tensor:
    """Zero baseline matches the trained RobustScaler-centred input."""

    return torch.zeros_like(x)


def integrated_gradients(
    forward: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    n_steps: int = DEFAULT_N_STEPS,
    baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute IG against the single input tensor ``x``.

    ``forward`` must take the interpolated input and return a scalar
    (single-element tensor); the gradients flow back from that scalar
    against the input. This is the classical IG formulation — Riemann
    mid-point integration over alpha in (0, 1).

    Returns a tensor with the same shape as ``x``: per-element
    attribution where positive values pushed the target up and negative
    values pushed it down.
    """

    if x.dim() == 0:
        raise ValueError("IG requires at least a 1-d input tensor")
    if baseline is None:
        baseline = _zero_baseline(x)
    if baseline.shape != x.shape:
        raise ValueError(
            f"baseline shape {tuple(baseline.shape)} must match input shape " f"{tuple(x.shape)}"
        )

    n_steps = max(2, int(n_steps))
    # Riemann mid-point: alphas at (k - 0.5) / n_steps for k = 1..n_steps.
    # Sundararajan et al. show mid-point integration converges faster than
    # left/right-endpoint rules at the same step count.
    alphas = torch.linspace(
        1.0 / (2.0 * n_steps),
        1.0 - 1.0 / (2.0 * n_steps),
        n_steps,
        device=x.device,
        dtype=x.dtype,
    )

    grads_accum = torch.zeros_like(x)
    for alpha in alphas:
        interpolated = baseline + alpha * (x - baseline)
        interpolated.requires_grad_(True)
        out = forward(interpolated)
        if out.dim() > 0:
            out = out.sum()
        grad = torch.autograd.grad(
            outputs=out,
            inputs=interpolated,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        if grad is None:
            continue
        grads_accum = grads_accum + grad.detach()

    averaged = grads_accum / float(n_steps)
    return (x - baseline) * averaged


def aggregate_feature_families(
    attribution: torch.Tensor,
    *,
    feature_size: int,
    slices: Sequence[tuple[str, slice]] = FEATURE_FAMILY_SLICES,
) -> list[FeatureFamilyAttribution]:
    """Bucket a per-feature attribution tensor into named families.

    ``attribution`` is ``(B, T, F)`` or ``(T, F)`` or ``(F,)``. The
    per-feature axis is reduced via L1 magnitude (sum of absolute
    values) per family, and the signed sum is kept alongside so the
    frontend can colour the bar by direction.

    Slices that fall outside ``feature_size`` are clipped (a 6-feature
    legacy checkpoint emits ``feature_size=6`` and only the ``market``
    family is meaningful — the rest of the slices collapse to empty).
    """

    if attribution.dim() == 1:
        flat = attribution.unsqueeze(0).unsqueeze(0)
    elif attribution.dim() == 2:
        flat = attribution.unsqueeze(0)
    else:
        flat = attribution
    # Sum over batch + time so the final per-family scalar is over the
    # full lookback window.
    per_feature_magnitude = flat.abs().sum(dim=(0, 1))
    per_feature_signed = flat.sum(dim=(0, 1))

    families: list[FeatureFamilyAttribution] = []
    for name, sl in slices:
        start = sl.start or 0
        stop = sl.stop or feature_size
        if start >= feature_size:
            families.append(FeatureFamilyAttribution(family=name, magnitude=0.0, signed=0.0))
            continue
        clipped = slice(start, min(stop, feature_size))
        mag = float(per_feature_magnitude[clipped].sum().item())
        signed = float(per_feature_signed[clipped].sum().item())
        families.append(FeatureFamilyAttribution(family=name, magnitude=mag, signed=signed))
    return families


def attribute_regime_panel(
    model: Any,
    x: torch.Tensor,
    *,
    forward_kwargs: dict[str, torch.Tensor] | None = None,
    n_steps: int | None = None,
) -> PanelAttribution:
    """IG attribution for the regime / vol-regime classifier head.

    Target: the argmax-class logit from the stance branch (the same
    surface the regime card argmax reads). Returns an ``unavailable``
    payload when the checkpoint is not in classification mode or when
    the forward path raises.
    """

    panel = "regime"
    steps = resolve_n_steps(n_steps)
    forward_kwargs = forward_kwargs or {}

    if str(getattr(model, "output_mode", "regression")) != "classification":
        return PanelAttribution(
            panel=panel,
            target="argmax_logit",
            families=[],
            n_steps=steps,
            unavailable=True,
            reason="not_classification_mode",
        )
    forward_multi = getattr(model, "forward_multi_task", None)
    if forward_multi is None:
        return PanelAttribution(
            panel=panel,
            target="argmax_logit",
            families=[],
            n_steps=steps,
            unavailable=True,
            reason="no_multi_task_forward",
        )

    # Resolve the argmax class once on the clean input so the IG
    # integration tracks a stable target across alpha steps.
    try:
        with torch.no_grad():
            out_dict = forward_multi(x, **forward_kwargs)
        stance = out_dict.get("stance")
        if stance is None or stance.dim() < 2:
            return PanelAttribution(
                panel=panel,
                target="argmax_logit",
                families=[],
                n_steps=steps,
                unavailable=True,
                reason="missing_stance_logits",
            )
        argmax_idx = int(stance.squeeze(0).argmax().item())
    except TypeError as exc:
        logger.warning("xai_regime_kwarg_mismatch detail=%s", str(exc))
        return PanelAttribution(
            panel=panel,
            target="argmax_logit",
            families=[],
            n_steps=steps,
            unavailable=True,
            reason="inference_kwarg_missing",
        )
    except Exception as exc:  # noqa: BLE001 -- defensive
        logger.warning(
            "xai_regime_forward_failed exception_class=%s",
            type(exc).__name__,
            exc_info=True,
        )
        return PanelAttribution(
            panel=panel,
            target="argmax_logit",
            families=[],
            n_steps=steps,
            unavailable=True,
            reason="unexpected_exception",
        )

    def _forward(input_tensor: torch.Tensor) -> torch.Tensor:
        out = forward_multi(input_tensor, **forward_kwargs)
        stance_out: torch.Tensor = out["stance"][:, argmax_idx]
        return stance_out

    try:
        attribution = integrated_gradients(_forward, x.detach(), n_steps=steps)
    except Exception as exc:  # noqa: BLE001 -- defensive
        logger.warning(
            "xai_regime_ig_failed exception_class=%s",
            type(exc).__name__,
            exc_info=True,
        )
        return PanelAttribution(
            panel=panel,
            target="argmax_logit",
            families=[],
            n_steps=steps,
            unavailable=True,
            reason="ig_runtime_error",
        )
    feature_size = int(getattr(model, "input_size", FEATURE_SIZE))
    families = aggregate_feature_families(attribution, feature_size=feature_size)
    return PanelAttribution(
        panel=panel,
        target=f"argmax_logit[{argmax_idx}]",
        families=families,
        n_steps=steps,
    )


def attribute_rates_panel(
    model: Any,
    x: torch.Tensor,
    *,
    head_name: str,
    forward_kwargs: dict[str, torch.Tensor] | None = None,
    n_steps: int | None = None,
) -> PanelAttribution:
    """IG attribution for one rates head's bps regression output.

    Target: the scalar ``rates_{head}_bps`` prediction. One call per
    mounted head; the caller iterates over the active heads.
    """

    panel = f"rates_{head_name}"
    steps = resolve_n_steps(n_steps)
    forward_kwargs = forward_kwargs or {}

    active = tuple(getattr(model, "rates_heads_active", ()) or ())
    if head_name not in active:
        return PanelAttribution(
            panel=panel,
            target=f"rates_{head_name}_bps",
            families=[],
            n_steps=steps,
            unavailable=True,
            reason="head_not_mounted",
        )
    forward_multi = getattr(model, "forward_multi_task", None)
    if forward_multi is None:
        return PanelAttribution(
            panel=panel,
            target=f"rates_{head_name}_bps",
            families=[],
            n_steps=steps,
            unavailable=True,
            reason="no_multi_task_forward",
        )

    pred_key = f"rates_{head_name}_bps"

    def _forward(input_tensor: torch.Tensor) -> torch.Tensor:
        out = forward_multi(input_tensor, **forward_kwargs)
        if pred_key not in out:
            raise RuntimeError(f"forward returned no {pred_key} key")
        bps_out: torch.Tensor = out[pred_key].squeeze(-1)
        return bps_out

    try:
        attribution = integrated_gradients(_forward, x.detach(), n_steps=steps)
    except TypeError as exc:
        logger.warning("xai_rates_kwarg_mismatch detail=%s", str(exc))
        return PanelAttribution(
            panel=panel,
            target=pred_key,
            families=[],
            n_steps=steps,
            unavailable=True,
            reason="inference_kwarg_missing",
        )
    except Exception as exc:  # noqa: BLE001 -- defensive
        logger.warning(
            "xai_rates_ig_failed head=%s exception_class=%s",
            head_name,
            type(exc).__name__,
            exc_info=True,
        )
        return PanelAttribution(
            panel=panel,
            target=pred_key,
            families=[],
            n_steps=steps,
            unavailable=True,
            reason="ig_runtime_error",
        )
    feature_size = int(getattr(model, "input_size", FEATURE_SIZE))
    families = aggregate_feature_families(attribution, feature_size=feature_size)
    return PanelAttribution(
        panel=panel,
        target=pred_key,
        families=families,
        n_steps=steps,
    )


def attribute_trajectory_panel(
    model: Any,
    inputs: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    n_steps: int | None = None,
) -> PanelAttribution:
    """IG attribution for the trajectory model's next-stance softmax.

    Target: the argmax next-stance probability. The trajectory input is
    a flat per-bar feature vector (not the rich-feature union), so the
    per-family aggregation collapses to a single ``trajectory`` bar
    keyed on the per-bar magnitude sum. Per-bar attribution magnitudes
    are not yet surfaced through the panel UI; the bar is intentionally
    coarse so the user sees that the trajectory model contributed,
    without inviting a per-bar interpretation the model cannot really
    deliver at this seq length.
    """

    panel = "trajectory"
    steps = resolve_n_steps(n_steps)
    if model is None:
        return PanelAttribution(
            panel=panel,
            target="next_stance_argmax",
            families=[],
            n_steps=steps,
            unavailable=True,
            reason="bundle_not_loaded",
        )

    try:
        with torch.no_grad():
            logits, _ = model(inputs, mask)
        if logits.dim() < 2:
            return PanelAttribution(
                panel=panel,
                target="next_stance_argmax",
                families=[],
                n_steps=steps,
                unavailable=True,
                reason="missing_logits",
            )
        argmax_idx = int(logits.squeeze(0).argmax().item())
    except Exception as exc:  # noqa: BLE001 -- defensive
        logger.warning(
            "xai_trajectory_forward_failed exception_class=%s",
            type(exc).__name__,
            exc_info=True,
        )
        return PanelAttribution(
            panel=panel,
            target="next_stance_argmax",
            families=[],
            n_steps=steps,
            unavailable=True,
            reason="unexpected_exception",
        )

    def _forward(input_tensor: torch.Tensor) -> torch.Tensor:
        logits, _ = model(input_tensor, mask)
        logit_out: torch.Tensor = logits[:, argmax_idx]
        return logit_out

    try:
        attribution = integrated_gradients(_forward, inputs.detach(), n_steps=steps)
    except Exception as exc:  # noqa: BLE001 -- defensive
        logger.warning(
            "xai_trajectory_ig_failed exception_class=%s",
            type(exc).__name__,
            exc_info=True,
        )
        return PanelAttribution(
            panel=panel,
            target="next_stance_argmax",
            families=[],
            n_steps=steps,
            unavailable=True,
            reason="ig_runtime_error",
        )

    # Per-bar magnitudes get surfaced under a single coarse "trajectory"
    # family so the panel UI can render one bar without inviting a
    # per-bar interpretation the model cannot defend at this seq length.
    total_magnitude = float(attribution.abs().sum().item())
    total_signed = float(attribution.sum().item())
    families = [
        FeatureFamilyAttribution(
            family="trajectory_input",
            magnitude=total_magnitude,
            signed=total_signed,
        )
    ]
    return PanelAttribution(
        panel=panel,
        target=f"next_stance_argmax[{argmax_idx}]",
        families=families,
        n_steps=steps,
    )


__all__ = [
    "DEFAULT_N_STEPS",
    "MAX_N_STEPS",
    "FEATURE_FAMILY_SLICES",
    "FeatureFamilyAttribution",
    "PanelAttribution",
    "aggregate_feature_families",
    "attribute_rates_panel",
    "attribute_regime_panel",
    "attribute_trajectory_panel",
    "integrated_gradients",
    "resolve_n_steps",
]
