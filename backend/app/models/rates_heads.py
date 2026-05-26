"""Rates-complex regression heads (#292).

Three independent regression heads riding on the shared encoder pooled
output: ``yield_2y_change_5d``, ``yield_5y_change_5d``,
``terminal_rate_change_5d``. Each head predicts the post-event 5-day
yield change in basis points (the unit the events.parquet target column
already carries) and is optional per training-loop CLI flag.

The optional auxiliary 3-class classification surface (easing / neutral
/ tightening) per head is computed at the model level by binning the
regression prediction against the per-fold tertile edges saved from the
training partition (see :mod:`app.data.quantile_labels`). On the loss
side the aux classification term is wired through the joint
``rates_alpha * MSE + (1 - rates_alpha) * CE_aux`` mixing identical to
the dual-head pattern landed in #304 — the regression head is the
primary supervised target, the classification surface stays an
inference-time product surface unless an operator opts the head into
dual / classification training mode.

Canonical names
---------------
- ``RATES_HEAD_NAMES`` -- ordered tuple ``("2y", "5y", "terminal")`` used
  as the dictionary keys on every per-head structure (per-head loss /
  per-head MSE / per-head prediction / per-head metric).
- ``RATES_HEAD_TARGET_COLUMNS`` -- mapping from short name to the
  events.parquet column it reads. Used by the loader to pull the
  bps-units target onto the FeatureVector rows.
- ``RATES_HEAD_LABEL_NAMES`` -- the canonical three-class labels the
  aux classification surface emits, in numeric order.

The values are pinned here so the model factory, the loss helper, the
conformal extension, the API serialiser, and the frontend all use the
same vocabulary.
"""

from __future__ import annotations

from typing import Final

# Ordered short names. Used everywhere as the per-head dict key.
RATES_HEAD_NAMES: Final[tuple[str, str, str]] = ("2y", "5y", "terminal")

# Maps the per-head short name to the events.parquet column carrying the
# strict-forward 5-day yield change target in raw basis points (#291).
RATES_HEAD_TARGET_COLUMNS: Final[dict[str, str]] = {
    "2y": "yield_2y_change_5d",
    "5y": "yield_5y_change_5d",
    "terminal": "terminal_rate_change_5d",
}

# Class labels for the auxiliary 3-class direction surface, in numeric
# order. Matches the encoding pinned by :mod:`app.data.quantile_labels`
# (``EASING_LABEL = -1`` -> idx 0 in the model's output; ``NEUTRAL = 0``
# -> idx 1; ``TIGHTENING = 1`` -> idx 2). The conformal classification
# helper reads per-head softmax over these three classes.
RATES_HEAD_LABEL_NAMES: Final[tuple[str, str, str]] = (
    "easing",
    "neutral",
    "tightening",
)

# Number of classes per rates head's aux classification surface; pinned
# so the model factory and the per-head softmax helper agree without
# round-tripping through the labels tuple.
RATES_HEAD_N_CLASSES: Final[int] = 3


def resolve_rates_heads(spec: str | None) -> tuple[str, ...]:
    """Map the ``--rates-heads`` CLI choice to the active per-head tuple.

    ``None`` / ``"none"`` -> ``()`` (back-compat: no rates heads mount).
    ``"all"`` -> :data:`RATES_HEAD_NAMES`.
    ``"2y"`` / ``"5y"`` / ``"terminal"`` -> singleton tuple.
    Anything else raises ``ValueError`` so the CLI surfaces the typo
    instead of silently dropping every head.
    """

    if spec is None:
        return ()
    cleaned = str(spec).strip().lower()
    if cleaned in {"", "none"}:
        return ()
    if cleaned == "all":
        return RATES_HEAD_NAMES
    if cleaned in RATES_HEAD_NAMES:
        return (cleaned,)
    raise ValueError(
        f"unsupported --rates-heads={spec!r}; choose one of "
        f"{{none, 2y, 5y, terminal, all}}."
    )


__all__ = (
    "RATES_HEAD_LABEL_NAMES",
    "RATES_HEAD_NAMES",
    "RATES_HEAD_N_CLASSES",
    "RATES_HEAD_TARGET_COLUMNS",
    "resolve_rates_heads",
)
