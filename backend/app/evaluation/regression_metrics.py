"""Regression-head walk-forward metrics with block-bootstrap CIs (#291).

The rates-complex heads added by #292 predict yield changes in raw basis
points rather than a discrete class label. Reporting parity with the
existing classification heads needs three metrics computed on the same
walk-forward partition:

- :func:`mae_bps` — mean absolute error in basis points (natural finance
  unit; comparable across folds without scale normalization).
- :func:`directional_accuracy` — share of rows where the predicted sign
  matches the observed sign. Zero-magnitude predictions or observations
  contribute via the :data:`ZERO_TOLERANCE_BPS` rule below.
- :func:`r_squared` — coefficient of determination against the
  observation mean. Comparable across folds when the target scale is
  stationary (basis points are scale-stable across the 2008-present
  window).

Each metric ships with a :func:`with_block_bootstrap_ci` wrapper that
emits a :class:`app.evaluation.bootstrap.BootstrapCI` over a row-level
resample. Block size defaults to 5 (the post-event horizon used by the
forward-yield targets) so the resampled rows share the temporal
auto-correlation structure of the original walk-forward partition.

The metrics are model-agnostic: the input is two equal-length numeric
sequences (``predicted``, ``observed``) plus an optional row weight.
This keeps the helpers usable from every training path (forecaster,
LSTM baseline, regression-ensemble aggregator) without pulling a model
dependency.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from app.evaluation.bootstrap import BootstrapCI, block_bootstrap_ci

# A predicted or observed value with absolute magnitude below this
# threshold (in bps) is treated as a "no-move" signal for the directional
# accuracy metric. Picked at 0.5 bps because the FOMC moves in 25-bp
# increments and rates desks colloquially treat moves below 1 bp as
# noise; the half-bp threshold sits well below any economically
# meaningful signal while excluding pure floating-point dither.
ZERO_TOLERANCE_BPS: float = 0.5

# Default block size for the moving-block bootstrap. Matches the 5-day
# post-event horizon used by the forward-yield targets so the resampled
# rows preserve the autocorrelation structure of the underlying
# event-window panel.
DEFAULT_BLOCK_SIZE: int = 5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_paired(predicted: Sequence[float], observed: Sequence[float]) -> int:
    if len(predicted) != len(observed):
        raise ValueError(
            f"predicted and observed must be equal length; "
            f"got {len(predicted)} vs {len(observed)}"
        )
    return len(predicted)


def _clean_pairs(
    predicted: Sequence[float], observed: Sequence[float]
) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for p, o in zip(predicted, observed):
        if p is None or o is None:
            continue
        try:
            pf = float(p)
            of = float(o)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(pf) and math.isfinite(of)):
            continue
        pairs.append((pf, of))
    return pairs


def _sign(value: float) -> int:
    """Three-way sign with a no-move band around zero."""

    if value > ZERO_TOLERANCE_BPS:
        return 1
    if value < -ZERO_TOLERANCE_BPS:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Point estimates
# ---------------------------------------------------------------------------


def mae_bps(predicted: Sequence[float], observed: Sequence[float]) -> float:
    """Mean absolute error of paired predictions, in basis points.

    Returns ``nan`` when no finite pair survives the cleaning pass.
    Inputs are assumed to already carry basis-point units; the function
    does not rescale.
    """

    _validate_paired(predicted, observed)
    pairs = _clean_pairs(predicted, observed)
    if not pairs:
        return float("nan")
    return sum(abs(p - o) for p, o in pairs) / len(pairs)


def directional_accuracy(
    predicted: Sequence[float], observed: Sequence[float]
) -> float:
    """Share of rows where the predicted sign matches the observed sign.

    Uses the three-way sign with a :data:`ZERO_TOLERANCE_BPS` no-move
    band, so a row where both predicted and observed are within the
    band counts as a match (both classified as "no move"). The
    convention treats a directional miss (predicted up, observed
    down) and a confidence miss (predicted up, observed flat) as
    equally wrong.

    Returns ``nan`` when no finite pair survives the cleaning pass.
    """

    _validate_paired(predicted, observed)
    pairs = _clean_pairs(predicted, observed)
    if not pairs:
        return float("nan")
    correct = sum(1 for p, o in pairs if _sign(p) == _sign(o))
    return correct / len(pairs)


def r_squared(predicted: Sequence[float], observed: Sequence[float]) -> float:
    """Coefficient of determination against the observation mean.

    Returns ``nan`` when fewer than two finite pairs survive (R^2 is
    undefined on a single observation). Returns 1.0 when the
    observations are constant and predictions match exactly; returns
    a large negative number when predictions are systematically biased
    relative to the observation mean.
    """

    _validate_paired(predicted, observed)
    pairs = _clean_pairs(predicted, observed)
    if len(pairs) < 2:
        return float("nan")
    obs = [o for _, o in pairs]
    mean_obs = sum(obs) / len(obs)
    ss_tot = sum((o - mean_obs) ** 2 for o in obs)
    ss_res = sum((o - p) ** 2 for p, o in pairs)
    if ss_tot <= 1e-18:
        # Constant observation series. Define R^2 = 1 when residuals
        # are exactly zero, else 0 (the typical sklearn convention is
        # NaN here, but a 0 / 1 binary keeps the metric monotone in
        # the "did we match a constant?" sense without polluting the
        # aggregator with NaN handling).
        return 1.0 if ss_res <= 1e-18 else 0.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Bootstrap CI wrappers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegressionMetricCI:
    """Point estimate + block-bootstrap CI for a regression-head metric."""

    name: str
    point: float
    lo: float
    hi: float
    coverage: float
    n_resamples: int
    block_size: int
    n_observations: int

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "point": self.point,
            "lo": self.lo,
            "hi": self.hi,
            "coverage": self.coverage,
            "n_resamples": self.n_resamples,
            "block_size": self.block_size,
            "n_observations": self.n_observations,
        }


def with_block_bootstrap_ci(
    *,
    name: str,
    predicted: Sequence[float],
    observed: Sequence[float],
    statistic: str,
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_resamples: int = 1000,
    coverage: float = 0.95,
    seed: int = 11,
) -> RegressionMetricCI:
    """Resample row-level pair indices via the moving-block bootstrap.

    ``statistic`` selects which point estimate to compute on each
    resample. Supported values: ``"mae_bps"``, ``"directional_accuracy"``,
    ``"r_squared"``. Each resample draws contiguous index blocks (length
    ``block_size``) with replacement until the resample length matches
    the original sample.
    """

    _validate_paired(predicted, observed)
    pairs = _clean_pairs(predicted, observed)
    if not pairs:
        nan = float("nan")
        return RegressionMetricCI(
            name=name,
            point=nan,
            lo=nan,
            hi=nan,
            coverage=coverage,
            n_resamples=n_resamples,
            block_size=block_size,
            n_observations=0,
        )

    point_fns = {
        "mae_bps": lambda preds, obs: mae_bps(preds, obs),
        "directional_accuracy": lambda preds, obs: directional_accuracy(preds, obs),
        "r_squared": lambda preds, obs: r_squared(preds, obs),
    }
    if statistic not in point_fns:
        raise ValueError(
            f"unsupported statistic={statistic!r}; "
            f"expected one of {sorted(point_fns)}"
        )

    point_fn = point_fns[statistic]
    cleaned_preds = [p for p, _ in pairs]
    cleaned_obs = [o for _, o in pairs]
    point = point_fn(cleaned_preds, cleaned_obs)

    # Reuse the block-bootstrap index generator from
    # :mod:`app.evaluation.bootstrap` so the resampling distribution
    # matches the classification-head metrics' CI construction.
    import random

    rng = random.Random(seed)
    samples: list[float] = []
    n = len(pairs)
    # Compute the metric on each resample.
    from app.evaluation.bootstrap import _resample_indices  # type: ignore[attr-defined]

    for _ in range(n_resamples):
        idx = _resample_indices(n, block_size, rng)
        resample_preds = [cleaned_preds[i] for i in idx]
        resample_obs = [cleaned_obs[i] for i in idx]
        samples.append(point_fn(resample_preds, resample_obs))

    samples = [s for s in samples if math.isfinite(s)]
    if not samples:
        nan = float("nan")
        return RegressionMetricCI(
            name=name,
            point=point,
            lo=nan,
            hi=nan,
            coverage=coverage,
            n_resamples=n_resamples,
            block_size=block_size,
            n_observations=n,
        )

    samples.sort()
    alpha = (1.0 - coverage) / 2.0
    lo_idx = int(alpha * len(samples))
    hi_idx = int((1.0 - alpha) * len(samples)) - 1
    lo_idx = max(0, min(len(samples) - 1, lo_idx))
    hi_idx = max(0, min(len(samples) - 1, hi_idx))
    return RegressionMetricCI(
        name=name,
        point=point,
        lo=samples[lo_idx],
        hi=samples[hi_idx],
        coverage=coverage,
        n_resamples=n_resamples,
        block_size=block_size,
        n_observations=n,
    )


def regression_metric_panel(
    *,
    predicted: Sequence[float],
    observed: Sequence[float],
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_resamples: int = 1000,
    coverage: float = 0.95,
    seed: int = 11,
) -> dict[str, RegressionMetricCI]:
    """Compute MAE-bps, directional accuracy, and R^2 with bootstrap CIs."""

    return {
        "mae_bps": with_block_bootstrap_ci(
            name="mae_bps",
            predicted=predicted,
            observed=observed,
            statistic="mae_bps",
            block_size=block_size,
            n_resamples=n_resamples,
            coverage=coverage,
            seed=seed,
        ),
        "directional_accuracy": with_block_bootstrap_ci(
            name="directional_accuracy",
            predicted=predicted,
            observed=observed,
            statistic="directional_accuracy",
            block_size=block_size,
            n_resamples=n_resamples,
            coverage=coverage,
            seed=seed,
        ),
        "r_squared": with_block_bootstrap_ci(
            name="r_squared",
            predicted=predicted,
            observed=observed,
            statistic="r_squared",
            block_size=block_size,
            n_resamples=n_resamples,
            coverage=coverage,
            seed=seed,
        ),
    }


# Re-export the underlying CI dataclass for callers that want to mix
# regression-head CIs with the existing classification-head CIs from
# :mod:`app.evaluation.bootstrap`.
__all__ = (
    "BootstrapCI",
    "DEFAULT_BLOCK_SIZE",
    "RegressionMetricCI",
    "ZERO_TOLERANCE_BPS",
    "block_bootstrap_ci",
    "directional_accuracy",
    "mae_bps",
    "r_squared",
    "regression_metric_panel",
    "with_block_bootstrap_ci",
)
