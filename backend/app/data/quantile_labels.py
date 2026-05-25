"""Per-fold 3-class quantile labels for the rates-complex regression targets (#291).

The rates-complex heads (#292) ship regression as the primary output:
``yield_2y_change_5d``, ``yield_5y_change_5d``, and
``terminal_rate_change_5d`` are stored in raw basis points on every
event row. For the optional product-surface classification view, this
module computes per-fold tertile cutoffs on the *train slice* of each
fold and assigns one of three labels to every row in the fold:

- ``easing`` (= ``-1``) — change below the 33rd percentile of the train slice
- ``neutral`` (= ``0``) — change in ``[33rd, 67th)``
- ``tightening`` (= ``+1``) — change at or above the 67th percentile

The bin edges are computed strictly from the train slice (no
look-ahead) and the edge pair is persisted alongside the fold's metadata
so the training-package reader and the conformal calibrator both see
the same boundaries. The 33/67 quantiles match the convention pinned by
the existing vol-regime classifier (calm / normal / high).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# Numeric encoding of the three classes. Pinned so downstream consumers
# (loss weighting, confusion-matrix renderer) reuse the same convention
# without re-deriving the order. The vol-regime classifier uses 0 / 1 / 2
# for calm / normal / high; the rates direction class uses signed -1 /
# 0 / +1 because direction has natural sign and a signed encoding keeps
# directional-accuracy computations (sign(pred) == sign(obs)) consistent
# with the regression target.
EASING_LABEL = -1
NEUTRAL_LABEL = 0
TIGHTENING_LABEL = 1

# Quantile cut points. Matches the (1/3, 2/3) split used by the existing
# vol-regime classifier so the binning convention is uniform across the
# product's classification heads.
LOWER_QUANTILE = 1.0 / 3.0
UPPER_QUANTILE = 2.0 / 3.0


@dataclass(frozen=True)
class QuantileBinEdges:
    """Tertile cutoffs computed on a single fold's train slice.

    Both edges are inclusive on the lower side and exclusive on the
    upper side, matching the standard pandas ``qcut`` convention.

    ``column`` records which target the edges were computed on so the
    fold-manifest entry is self-describing.
    """

    column: str
    lower: float
    upper: float
    n_train_rows: int

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "column": self.column,
            "lower": float(self.lower),
            "upper": float(self.upper),
            "n_train_rows": int(self.n_train_rows),
        }


def _empirical_quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation empirical quantile (matches numpy default).

    Pulled out so the module stays import-light (no numpy dependency at
    fold-manifest write time) and so the cutoff convention is stable
    across pandas versions that change ``qcut`` interpolation defaults.
    """

    if not values:
        return float("nan")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1]; got {q!r}")
    cleaned = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    n = len(cleaned)
    if n == 0:
        return float("nan")
    if n == 1:
        return cleaned[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return cleaned[lo]
    weight = pos - lo
    return cleaned[lo] * (1.0 - weight) + cleaned[hi] * weight


def compute_bin_edges(
    train_values: Sequence[float],
    *,
    column: str,
) -> QuantileBinEdges:
    """Compute the (lower, upper) tertile cutoffs on a train slice.

    Rows whose target is ``None`` or non-finite are dropped before the
    quantile is computed; ``n_train_rows`` reports the count of surviving
    finite values.
    """

    finite = [
        float(v) for v in train_values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    if not finite:
        nan = float("nan")
        return QuantileBinEdges(column=column, lower=nan, upper=nan, n_train_rows=0)

    lower = _empirical_quantile(finite, LOWER_QUANTILE)
    upper = _empirical_quantile(finite, UPPER_QUANTILE)
    return QuantileBinEdges(
        column=column,
        lower=lower,
        upper=upper,
        n_train_rows=len(finite),
    )


def label_for_value(value: float | None, edges: QuantileBinEdges) -> int | None:
    """Map a raw bps change to one of the three class labels.

    Returns ``None`` when ``value`` is missing or the edges are not
    finite (the empty-train-slice case).
    """

    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    if not (math.isfinite(edges.lower) and math.isfinite(edges.upper)):
        return None
    if v < edges.lower:
        return EASING_LABEL
    if v < edges.upper:
        return NEUTRAL_LABEL
    return TIGHTENING_LABEL


def assign_labels(
    values: Sequence[float | None],
    edges: QuantileBinEdges,
) -> list[int | None]:
    """Apply :func:`label_for_value` to every entry in ``values``."""

    return [label_for_value(v, edges) for v in values]


def fold_manifest_entry(
    *,
    fold_id: str,
    edges_by_column: Mapping[str, QuantileBinEdges],
) -> dict[str, dict[str, float | int | str] | str]:
    """Build the fold-manifest payload for one fold.

    Returns a dict shaped like ``{"fold_id": <id>, "quantile_bin_edges":
    {<column>: {...}}}`` that the training-package writer merges into
    each fold's metadata block. The shape is JSON-serializable as-is.
    """

    return {
        "fold_id": fold_id,
        "quantile_bin_edges": {
            column: edges.to_dict() for column, edges in edges_by_column.items()
        },
    }


__all__ = (
    "EASING_LABEL",
    "LOWER_QUANTILE",
    "NEUTRAL_LABEL",
    "QuantileBinEdges",
    "TIGHTENING_LABEL",
    "UPPER_QUANTILE",
    "assign_labels",
    "compute_bin_edges",
    "fold_manifest_entry",
    "label_for_value",
)
