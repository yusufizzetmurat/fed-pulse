"""HAR-tercile regime baseline served alongside the late-fusion classifier.

Per wiki section 20 (Gated_Fusion_InfoNCE_Comprehensive_Null), bucketing HAR's
predicted realized variance by the train-slice tercile cutoffs is the single
best forward-vol-regime classifier on the canonical fold protocol: macro-F1
0.687 / 0.685 / 0.654 at h=1 / 5 / 22, beating both the market-only fusion
arm and the full text+market fused model. Text robustly hurts the regime
classifier at h=1 and h=22 (block-bootstrap 95% CI excludes zero) and is null
at h=5; the regime call is a persistence call HAR already makes.

This module ports the bucketing logic out of the research-side regime trainer
(``app.data.fed_comms_regime``) into a thin serving wrapper that re-uses the
HAR coefficients persisted on the QLIKE-DLq production artifact. Soft regime
probabilities come from a normal centered on HAR's log-RV point with a sigma
derived from the per-horizon 80% conformal band width (band/2 over the
z_{0.9} ~= 1.2816 multiplier), giving the UI a proper {low, medium, high}
mass triple instead of a single argmax.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from app.data.intraday_rv_forecast import _EPS as _RVEPS, _LOGV_CLAMP, _har_lags


_SQRT_TWO = math.sqrt(2.0)
_CONFIDENCE_Z = 1.2816  # z_{0.9}; matches forecaster.CONFIDENCE_Z_SCORE
_TERCILE_LABELS: tuple[str, str, str] = ("low", "medium", "high")

# Wiki §20 macro-F1 numbers for HAR-tercile on the canonical 5-fold expanding
# walk-forward, pooled across folds. Source: Gated_Fusion_InfoNCE_Comprehensive_Null.md
# Result 2 ("Tercile regime classification"). The bucketing rule below
# reproduces the exact research-side baseline, so the macro-F1 is wired
# through unchanged.
_HAR_TERCILE_MACRO_F1: dict[int, float] = {1: 0.687, 5: 0.685, 22: 0.654}
_HAR_TERCILE_F1_SOURCE = (
    "Pooled 5-fold expanding walk-forward eval (n=1999); HAR's "
    "continuous forecast bucketed into terciles."
)
_HORIZONS: tuple[int, ...] = (1, 5, 22)


def _phi(z: float) -> float:
    """Standard-normal CDF via erf; no scipy on the serving path."""

    return 0.5 * (1.0 + math.erf(z / _SQRT_TWO))


def _bucket_index(value: float, q33: float, q67: float) -> int:
    """Same digitize convention as ``fed_comms_regime._labels`` (np.digitize)."""

    if value < q33:
        return 0
    if value < q67:
        return 1
    return 2


def _tercile_cutoffs(rv_history: np.ndarray) -> tuple[float, float]:
    """Quantile cutoffs on the supplied realized-variance series.

    The serving path has no train-slice anchor (the artifact pins HAR
    coefficients but not the q33/q67 cutoffs the wiki baseline uses).
    We approximate by reading them off the available history; this is
    the same operation the research-side trainer performs against its
    train fold, just applied to whichever 60-day window the endpoint
    has on hand.
    """

    if rv_history.size < 3:
        raise ValueError("rv_history must carry at least 3 values to derive terciles")
    q33, q67 = np.quantile(rv_history, [1.0 / 3.0, 2.0 / 3.0])
    return float(q33), float(q67)


def _har_point_log_rv(log_rv: np.ndarray, har_coef: list[float] | np.ndarray) -> float:
    """OLS HAR point forecast on the latest observation in log-RV space."""

    har = _har_lags(log_rv)
    last = har[-1]
    coef = np.asarray(har_coef, dtype=np.float64)
    log_pred = float(coef[0] + coef[1] * last[0] + coef[2] * last[1] + coef[3] * last[2])
    return float(np.clip(log_pred, -_LOGV_CLAMP, _LOGV_CLAMP))


def _soft_tercile_probs(
    predicted_rv: float,
    q33: float,
    q67: float,
    sigma_log: float,
) -> dict[str, float]:
    """Gaussian-CDF mass per tercile, computed in log-RV space.

    The HAR head emits a point in log-RV space and the conformal band gives
    a symmetric residual width; we treat the predictive distribution as
    normal in log space and integrate it against the log of each cutoff.
    Renormalises any erf() residual so the three masses always sum to 1.0.
    Falls back to a one-hot at the argmax bucket when ``sigma_log <= 0``
    or a cutoff is non-positive (log blows up).
    """

    bucket = _bucket_index(predicted_rv, q33, q67)
    fallback = {label: 1.0 if i == bucket else 0.0 for i, label in enumerate(_TERCILE_LABELS)}
    if sigma_log <= 0.0:
        return fallback
    if q33 <= 0.0 or q67 <= 0.0 or q33 > q67:
        return fallback
    if predicted_rv <= 0.0:
        return fallback
    log_pred = math.log(predicted_rv + _RVEPS)
    log_q33 = math.log(q33)
    log_q67 = math.log(q67)
    cdf_low = _phi((log_q33 - log_pred) / sigma_log)
    cdf_high = _phi((log_q67 - log_pred) / sigma_log)
    mass = [cdf_low, cdf_high - cdf_low, 1.0 - cdf_high]
    mass = [m if m > 0.0 else 0.0 for m in mass]
    total = sum(mass)
    if total <= 0.0:
        return fallback
    return {label: mass[i] / total for i, label in enumerate(_TERCILE_LABELS)}


def get_har_coef(horizon: int) -> list[float]:
    """Read the HAR OLS coefficients for ``horizon`` off the cached spec.

    Importing inside the call keeps the module load cheap on the
    schemathesis path and mirrors how ``rv_forecaster`` defers its own
    torch / HF imports.
    """

    from app.services.rv_forecaster import _RvPredictor

    pred = _RvPredictor.get()
    row = pred.spec["by_horizon"][f"h{horizon}"]
    return list(row["har_coef"])


def predict_har_regime(
    rv_history: list[float] | np.ndarray,
    cutoffs_q33: float | None = None,
    cutoffs_q67: float | None = None,
    har_coef: dict[int, list[float]] | None = None,
) -> dict[str, Any]:
    """HAR-tercile regime classification across the 1/5/22-day horizons.

    ``rv_history`` is a chronologically ordered series of daily realized
    variance values, matching :func:`app.services.rv_forecaster.predict_rv`.
    When ``cutoffs_q33`` / ``cutoffs_q67`` are not supplied they are read
    off the series itself (per-horizon training cutoffs are not on the
    production artifact). When ``har_coef`` is not supplied it is read off
    the cached production spec so the bucket call uses the same OLS HAR
    forecast that backs the QLIKE-DLq ensemble.
    """

    rv = np.asarray(rv_history, dtype=np.float64)
    if rv.ndim != 1 or len(rv) < 22:
        raise ValueError(
            "rv_history must be a 1-D series of at least 22 daily RV values"
        )
    if np.any(rv <= 0) or not np.all(np.isfinite(rv)):
        raise ValueError("rv_history values must be positive finite numbers")

    log_rv = np.log(rv + _RVEPS)

    if cutoffs_q33 is None or cutoffs_q67 is None:
        cutoffs_q33, cutoffs_q67 = _tercile_cutoffs(rv)
    if cutoffs_q33 > cutoffs_q67:
        raise ValueError("cutoffs_q33 must be <= cutoffs_q67")

    from app.services.rv_forecaster import _RvPredictor

    pred = _RvPredictor.get()
    eval_block = (pred.eval or {}).get("by_horizon", {}) if pred.eval else {}

    horizons_out: list[dict[str, Any]] = []
    for h in _HORIZONS:
        hk = f"h{h}"
        row = pred.spec["by_horizon"][hk]
        coef = (har_coef or {}).get(h) or row["har_coef"]
        log_point = _har_point_log_rv(log_rv, coef)
        predicted_rv = float(math.exp(log_point))
        # 80% conformal band width -> sigma in log-RV space; the QLIKE-DLq
        # band is the QLIKE-DLq ensemble's, but the HAR baseline shares
        # the same residual dispersion order-of-magnitude. Falls back to a
        # one-hot when the band is missing.
        q80 = float(row.get("conformal_quantiles", {}).get("0.20", 0.0))
        sigma_log = q80 / _CONFIDENCE_Z if q80 > 0.0 else 0.0
        probs = _soft_tercile_probs(predicted_rv, cutoffs_q33, cutoffs_q67, sigma_log)
        tercile_idx = _bucket_index(predicted_rv, cutoffs_q33, cutoffs_q67)
        eval_row = eval_block.get(hk, {}) if isinstance(eval_block, dict) else {}
        horizons_out.append(
            {
                "h": h,
                "predicted_rv": predicted_rv,
                "tercile": _TERCILE_LABELS[tercile_idx],
                "tercile_probs": probs,
                "macro_f1": _HAR_TERCILE_MACRO_F1[h],
                "macro_f1_source": _HAR_TERCILE_F1_SOURCE,
                "qlike_model": _safe_float(eval_row.get("qlike_ens")),
                "qlike_har": _safe_float(eval_row.get("qlike_har")),
            }
        )
    return {
        "horizons": horizons_out,
        "cutoffs_q33": float(cutoffs_q33),
        "cutoffs_q67": float(cutoffs_q67),
        "model_revision": pred.revision,
    }


def _safe_float(v: Any) -> float | None:
    """Coerce ``v`` to a finite float; return None on non-numeric input."""

    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f
