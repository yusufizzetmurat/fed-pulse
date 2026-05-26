"""Map regression-head log-vol predictions onto the calm/normal/high regime axis.

Under the regression-canonical objective (ADR 0015, issue #322) the
forecaster's supervised head emits ``log(forward_realized_vol_10d)``.
The user-facing regime label space, however, is still the discrete
``calm | normal | high`` tertile that ``bucket_realized_regime`` and
the classifier head produce -- the API contract, the UI cards, and the
multi-axis dashboard all key off those three buckets.

This module bridges the two domains. The active checkpoint persists
its tertile cutoffs in **raw vol space** under
``ModelConfig.vol_regime_quantiles`` (two cutoffs delimiting three
bins, fit on the train slice). The helpers below take a log-vol point
(and optionally its per-fold residual std), exponentiate back to raw
vol, and either snap to the matching bucket or compute Gaussian-CDF
mass per bucket. The conformal manifest is never consulted here -- it
lives one layer up, so this module stays pure-function and trivially
unit-testable.
"""
from __future__ import annotations

import math
from typing import Literal

REGIME_LABELS: tuple[str, str, str] = ("calm", "normal", "high")

_SQRT_TWO = math.sqrt(2.0)


def _phi(z: float) -> float:
    # Standard-normal CDF via erf; avoids pulling scipy/numpy onto the
    # inference hot path.
    return 0.5 * (1.0 + math.erf(z / _SQRT_TWO))


def bucket_log_rv(
    log_rv_point: float,
    raw_vol_cutoffs: tuple[float, ...],
) -> Literal["calm", "normal", "high"] | None:
    """Map a log(forward realized vol) prediction to a calm/normal/high bucket.

    ``raw_vol_cutoffs`` is the active checkpoint's
    ``vol_regime_quantiles`` tuple -- tertile cutoffs in RAW vol space,
    fit on the train slice. The function exponentiates ``log_rv_point``
    back to raw vol and compares against the cutoffs, matching the
    existing ``bucket_realized_regime`` convention so the UI label
    space is uniform. Returns ``None`` when the cutoffs are missing or
    not exactly 2 entries (this helper only supports the 3-class regime
    layout; broader cutoff shapes are an out-of-scope future change).
    """
    if len(raw_vol_cutoffs) != 2:
        return None
    cutoff_low, cutoff_high = raw_vol_cutoffs
    if cutoff_low > cutoff_high:
        # Mis-fit cutoffs (should never happen on a real checkpoint) --
        # bail rather than silently invert the bucket order.
        return None
    raw_vol_point = math.exp(log_rv_point)
    if raw_vol_point < cutoff_low:
        return "calm"
    if raw_vol_point < cutoff_high:
        return "normal"
    return "high"


def derive_distribution(
    log_rv_point: float,
    log_rv_std: float,
    raw_vol_cutoffs: tuple[float, ...],
) -> dict[str, float] | None:
    """Gaussian-CDF mass per regime bucket, computed in log space.

    ``log_rv_std`` is the per-fold residual std on the regression head
    (use the persisted std from the run summary's ``log_rv_scaler`` or,
    when a conformal manifest exists, derive it from the 80%-quantile
    width divided by ``2 * z_{0.9}`` ~= ``2 * 1.2816``). Computes
    ``P(log_rv < log(cutoff))`` via the standard-normal CDF for each
    cutoff, then differences adjacent CDFs to get per-bin mass.
    Returns ``None`` when cutoffs are missing or ``log_rv_std <= 0``.
    Keys are ``{"calm", "normal", "high"}``; values sum to ``1.0``
    within ``1e-9``.
    """
    if len(raw_vol_cutoffs) != 2:
        return None
    if log_rv_std <= 0.0:
        return None
    cutoff_low, cutoff_high = raw_vol_cutoffs
    if cutoff_low <= 0.0 or cutoff_high <= 0.0:
        # log() blows up on a non-positive cutoff; treat as malformed.
        return None
    if cutoff_low > cutoff_high:
        return None
    log_cutoff_low = math.log(cutoff_low)
    log_cutoff_high = math.log(cutoff_high)
    cdf_low = _phi((log_cutoff_low - log_rv_point) / log_rv_std)
    cdf_high = _phi((log_cutoff_high - log_rv_point) / log_rv_std)
    mass_calm = cdf_low
    mass_normal = cdf_high - cdf_low
    mass_high = 1.0 - cdf_high
    # Floor at zero to absorb the ~1e-17 negative residuals erf() can
    # leave when the point is many sigmas outside a cutoff; the result
    # is then renormalised so the three values still sum to 1.0.
    mass_calm = max(mass_calm, 0.0)
    mass_normal = max(mass_normal, 0.0)
    mass_high = max(mass_high, 0.0)
    total = mass_calm + mass_normal + mass_high
    if total <= 0.0:
        return None
    return {
        "calm": mass_calm / total,
        "normal": mass_normal / total,
        "high": mass_high / total,
    }
