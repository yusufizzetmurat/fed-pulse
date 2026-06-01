"""Per-fold rates-head target / scaler helpers (#292, #305).

The three rates heads (``2y`` / ``5y`` / ``terminal``) consume a strict-
forward 5-day yield change in basis points read off the FeatureVector
target row. Each fold fits an independent per-head standardiser on the
train slice (a ``(mean, std)`` pair); val and test partitions reuse the
train-fitted scaler so no look-ahead leaks into the standardisation
step.

Two target modes are wired (#305):

- ``raw`` (default, byte-identical to the pre-#305 path): the head
  predicts the observed ``yield_<tenor>_change_5d`` move in bps.
- ``fomc_attributable``: the head predicts the FOMC-attributable
  component of the observed move, defined as the 1-D projection of the
  observed move onto the strict-prior policy-surprise direction
  ``sign(mp_surprise_level)``. When the surprise magnitude is below
  :data:`SURPRISE_DIRECTION_EPSILON_BPS` (no-change meeting; direction
  ill-defined) the target is marked missing rather than coerced to zero.
  See :func:`fomc_attributable_projection` and ADR 0027.

The aux 3-class classification surface (easing / neutral / tightening)
is bucketed against per-fold tertile edges fitted on the same train
slice via :mod:`app.data.quantile_labels`. Both the regression scaler
and the classification edges land on the fold manifest sidecar so the
inference path and the API serialiser can read the same boundaries the
training loop saw.

Row alignment with the existing classification target (and the optional
log_rv tensor #304 added) is enforced by walking the SAME filter
``_build_partition_tensors`` does in classification mode: drop any
sequence whose target row carries a null ``forward_realized_vol_10d``
because the classifier still anchors the partition. Within the
surviving rows, a row whose rates target is missing /
non-finite contributes a NaN mask flag so the loss helper can ignore
it; the tensor stays row-aligned with ``y`` (and ``log_rv``) for the
TensorDataset arity invariant.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from app.data.quantile_labels import QuantileBinEdges, compute_bin_edges, label_for_value
from app.models.config import SEQUENCE_LENGTH, FeatureVector
from app.models.rates_heads import (
    RATES_HEAD_NAMES,
    RATES_HEAD_TARGET_COLUMNS,
)


# Canonical literal for the per-head target derivation. ``raw`` keeps the
# pre-#305 contract (predict the observed bps move); ``fomc_attributable``
# projects the observed move onto the strict-prior policy-surprise
# direction (see :func:`fomc_attributable_projection` + ADR 0027).
RATES_TARGET_MODES: tuple[str, ...] = ("raw", "fomc_attributable")
DEFAULT_RATES_TARGET_MODE: str = "raw"

# Minimum absolute mp_surprise_level (in bps) below which the surprise
# direction is treated as ill-defined and the FOMC-attributable target
# is marked missing. 1 bp is well below the FOMC's 25-bp standard move
# and an order of magnitude above floating-point noise in the strict-
# prior implied-move proxy that anchors the level construction.
SURPRISE_DIRECTION_EPSILON_BPS: float = 1.0


def fomc_attributable_projection(
    observed_move_bps: float | None,
    mp_surprise_level_bps: float | None,
    *,
    epsilon_bps: float = SURPRISE_DIRECTION_EPSILON_BPS,
) -> float | None:
    """Project the observed rates move onto the policy-surprise direction.

    The Kuttner-style 1-D projection:

    - ``u = sign(mp_surprise_level)`` is the strict-prior surprise
      direction (post-#350 / ADR 0024 construction; the surprise is the
      actual policy decision minus the implied next-move proxy at
      ``T-1``).
    - ``projected = observed_move_bps * u`` is the scalar coefficient
      of the projection in bps, signed positive when the observed move
      agrees with the surprise direction and negative when it opposes
      it.

    Returns ``None`` (target missing) when:

    - either input is ``None`` / non-finite, or
    - ``|mp_surprise_level| < epsilon_bps`` (no-change meeting; the
      direction is ill-defined and zero would be a label, not a no-op).

    The missing case is the load-bearing edge — coercing it to zero
    would inject a fake "no-attributable-move" label on every pause
    meeting and bias the regression toward the origin.
    """

    if observed_move_bps is None or mp_surprise_level_bps is None:
        return None
    try:
        obs = float(observed_move_bps)
        surp = float(mp_surprise_level_bps)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(obs) or not math.isfinite(surp):
        return None
    if abs(surp) < float(epsilon_bps):
        return None
    sign = 1.0 if surp > 0.0 else -1.0
    return obs * sign


@dataclass(frozen=True)
class RatesHeadScaler:
    """Standardiser (``mean`` / ``std`` in bps) for one rates head.

    ``mean`` and ``std`` are fitted on the train slice's raw bps values.
    The training loop applies ``(x - mean) / std`` to keep the MSE term
    on a unit-variance manifold so the joint-loss alpha behaves like a
    true mixing weight (otherwise a 30-bps standard deviation would
    dwarf the ~log(3) CE on the classification surface).
    """

    mean: float
    std: float


def _gather_rates_values_for_group(
    group: Sequence[FeatureVector],
    head_name: str,
    *,
    target_mode: str = DEFAULT_RATES_TARGET_MODE,
) -> list[float]:
    """Walk a sequence group and emit every supervised row's rates value.

    Mirrors :func:`app.training.loop._build_partition_log_rv_target` row
    selection so the tensor lands row-aligned with ``y``. The selection
    rule is the existing classifier's: drop a group whose leading
    target's ``forward_realized_vol_10d`` is null; for every kept group
    emit one value per row at index >= ``SEQUENCE_LENGTH`` whose
    forward-vol survives the filter.

    ``target_mode`` selects between the raw observed move
    (``target_<column>`` field) and the FOMC-attributable projection
    (``target_<column>_fomc_attributable`` field). Missing values
    (``None`` or non-finite) emit ``math.nan`` so the partition builder
    can mask them out row-by-row.
    """

    if len(group) < SEQUENCE_LENGTH + 1:
        return []
    leading_target = group[SEQUENCE_LENGTH]
    leading_vol = getattr(leading_target, "forward_realized_vol_10d", None)
    if leading_vol is None or (isinstance(leading_vol, float) and leading_vol != leading_vol):
        return []
    field = _rates_field_for(head_name, target_mode=target_mode)
    out: list[float] = []
    for idx in range(SEQUENCE_LENGTH, len(group)):
        target_row = group[idx]
        forward_vol = getattr(target_row, "forward_realized_vol_10d", None)
        if forward_vol is None:
            continue
        if isinstance(forward_vol, float) and forward_vol != forward_vol:
            continue
        value = getattr(target_row, field, None)
        if value is None:
            out.append(math.nan)
            continue
        try:
            f = float(value)
        except (TypeError, ValueError):
            out.append(math.nan)
            continue
        if not math.isfinite(f):
            out.append(math.nan)
            continue
        out.append(f)
    return out


def _rates_field_for(
    head_name: str,
    *,
    target_mode: str = DEFAULT_RATES_TARGET_MODE,
) -> str:
    """Return the FeatureVector field carrying the per-head bps target.

    ``target_mode='raw'`` returns ``target_<column>`` (the observed
    bps move). ``target_mode='fomc_attributable'`` returns
    ``target_<column>_fomc_attributable`` (the surprise-projected
    scalar). Anything else raises ``ValueError``.
    """

    name = str(head_name).lower()
    if name not in RATES_HEAD_TARGET_COLUMNS:
        raise ValueError(
            f"unknown rates head name {head_name!r}; " f"expected one of {RATES_HEAD_NAMES}"
        )
    mode = str(target_mode).lower()
    if mode not in RATES_TARGET_MODES:
        raise ValueError(
            f"unsupported rates_target_mode={target_mode!r}; "
            f"expected one of {RATES_TARGET_MODES}"
        )
    base = f"target_{RATES_HEAD_TARGET_COLUMNS[name]}"
    if mode == "fomc_attributable":
        return f"{base}_fomc_attributable"
    return base


def fit_rates_scaler(values: Sequence[float]) -> RatesHeadScaler:
    """Fit a ``(mean, std)`` standardiser on the train slice's bps values.

    NaN rows (missing target) are dropped before the fit; an empty input
    or a degenerate single-value input collapses to ``mean=0, std=1`` so
    the standardiser is well-defined on small fixtures (the loop still
    wraps the head in a row-mask so empty partitions never contribute).
    """

    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return RatesHeadScaler(mean=0.0, std=1.0)
    if len(finite) == 1:
        return RatesHeadScaler(mean=float(finite[0]), std=1.0)
    n = len(finite)
    mean_val = sum(finite) / n
    variance = sum((v - mean_val) ** 2 for v in finite) / n
    std_val = math.sqrt(max(variance, 0.0))
    if std_val < 1e-6:
        std_val = 1.0
    return RatesHeadScaler(mean=float(mean_val), std=float(std_val))


def build_partition_rates_targets(
    sequence_groups: Sequence[Sequence[FeatureVector]],
    *,
    head_names: Sequence[str],
    scalers: dict[str, RatesHeadScaler] | None = None,
    edges_by_head: dict[str, QuantileBinEdges] | None = None,
    target_mode: str = DEFAULT_RATES_TARGET_MODE,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, RatesHeadScaler],
    dict[str, QuantileBinEdges],
]:
    """Materialise per-head ``(target, mask, cls_target, cls_mask)`` tensors.

    Returns six dictionaries keyed on the rates-head short name (``2y``
    / ``5y`` / ``terminal``):

    - ``bps_target`` -- standardised regression target (``float32``).
      Rows whose raw value is missing emit ``0.0`` (masked out via
      ``bps_mask``).
    - ``bps_mask`` -- bool mask, ``True`` when the regression target is
      finite, ``False`` otherwise.
    - ``cls_target`` -- ``int64`` 3-class label
      (``0`` easing / ``1`` neutral / ``2`` tightening). Rows with no
      classifiable target carry ``-1`` and ``cls_mask`` is False.
    - ``cls_mask`` -- bool mask, ``True`` when the per-fold edges yield
      a valid class label.
    - ``scalers`` -- per-head ``RatesHeadScaler`` echoed back (caller
      passes ``None`` on the train slice and the helper fits a fresh
      scaler; val / test slices pass the train-fitted scaler so no
      look-ahead leaks into the standardisation).
    - ``edges_by_head`` -- per-head :class:`QuantileBinEdges` echoed
      back, fitted on the train slice when ``edges_by_head`` is
      ``None`` and re-used otherwise.

    All five tensors are 1-D and row-aligned with the per-partition
    ``y`` tensor :func:`_build_partition_tensors` emits in
    classification mode. Empty input ⇒ empty tensors.
    """

    head_list = [str(n).lower() for n in head_names]
    for name in head_list:
        if name not in RATES_HEAD_NAMES:
            raise ValueError(
                f"unknown rates head name {name!r}; " f"expected one of {RATES_HEAD_NAMES}"
            )
    mode = str(target_mode).lower()
    if mode not in RATES_TARGET_MODES:
        raise ValueError(
            f"unsupported rates_target_mode={target_mode!r}; "
            f"expected one of {RATES_TARGET_MODES}"
        )

    # Per-head row-wise raw values, with NaN holes for missing rows.
    raw_by_head: dict[str, list[float]] = {name: [] for name in head_list}
    for group in sequence_groups:
        # Per-head walk so a missing row in head A still emits a NaN at
        # the same row position for head B, keeping the per-head tensors
        # row-aligned with each other and with ``y``.
        per_head_rows: dict[str, list[float]] = {
            name: _gather_rates_values_for_group(group, name, target_mode=mode)
            for name in head_list
        }
        if not per_head_rows:
            continue
        row_count = max((len(rows) for rows in per_head_rows.values()), default=0)
        for name in head_list:
            rows = per_head_rows[name]
            if len(rows) < row_count:
                # Pad shorter heads with NaN so the per-head row count
                # agrees across heads even on degenerate fixtures.
                rows.extend([math.nan] * (row_count - len(rows)))
            raw_by_head[name].extend(rows)

    fitted_scalers: dict[str, RatesHeadScaler] = {}
    fitted_edges: dict[str, QuantileBinEdges] = {}
    bps_target: dict[str, torch.Tensor] = {}
    bps_mask: dict[str, torch.Tensor] = {}
    cls_target: dict[str, torch.Tensor] = {}
    cls_mask: dict[str, torch.Tensor] = {}

    for name in head_list:
        values = raw_by_head[name]
        if scalers is not None and name in scalers:
            scaler = scalers[name]
        else:
            # Train slice: fit a fresh scaler on the surviving values.
            scaler = fit_rates_scaler(values)
        fitted_scalers[name] = scaler

        if edges_by_head is not None and name in edges_by_head:
            edges = edges_by_head[name]
        else:
            # Train slice: fit fresh tertile edges on the surviving values.
            column = RATES_HEAD_TARGET_COLUMNS[name]
            edges = compute_bin_edges(
                [v for v in values if math.isfinite(v)],
                column=column,
            )
        fitted_edges[name] = edges

        n = len(values)
        target_tensor = torch.zeros(n, dtype=torch.float32)
        mask_tensor = torch.zeros(n, dtype=torch.bool)
        cls_tensor = torch.full((n,), -1, dtype=torch.int64)
        cls_mask_tensor = torch.zeros(n, dtype=torch.bool)
        for i, raw in enumerate(values):
            if raw is None or not math.isfinite(raw):
                continue
            standardised = (float(raw) - scaler.mean) / scaler.std
            target_tensor[i] = float(standardised)
            mask_tensor[i] = True
            label = label_for_value(float(raw), edges)
            if label is None:
                continue
            # quantile_labels emits signed ints -1 / 0 / +1; we shift to
            # 0 / 1 / 2 for cross-entropy compatibility.
            cls_idx = int(label) + 1
            if 0 <= cls_idx <= 2:
                cls_tensor[i] = cls_idx
                cls_mask_tensor[i] = True

        bps_target[name] = target_tensor
        bps_mask[name] = mask_tensor
        cls_target[name] = cls_tensor
        cls_mask[name] = cls_mask_tensor

    return bps_target, bps_mask, cls_target, cls_mask, fitted_scalers, fitted_edges


def inverse_standardise_bps(
    standardised: float,
    scaler: RatesHeadScaler,
) -> float:
    """Invert the train-fitted standardiser to recover raw bps units.

    Used by the inference path so the API response carries the bps
    prediction in the natural finance unit rather than the standardised
    training-time scale.
    """

    return float(standardised) * float(scaler.std) + float(scaler.mean)


__all__ = (
    "DEFAULT_RATES_TARGET_MODE",
    "RATES_TARGET_MODES",
    "RatesHeadScaler",
    "SURPRISE_DIRECTION_EPSILON_BPS",
    "build_partition_rates_targets",
    "fit_rates_scaler",
    "fomc_attributable_projection",
    "inverse_standardise_bps",
)
