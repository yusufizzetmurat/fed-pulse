from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Default nominal coverage 0.80 (alpha=0.20) matches the regression-head
# default. Picked so the prediction set is informative on a 3-class
# target — at alpha=0.05 the set frequently covers all three classes
# on borderline rows, which is correct but uninformative for a decision
# support surface.
DEFAULT_CLASSIFICATION_ALPHA = 0.2


@dataclass(frozen=True)
class ConformalManifest:
    """Split-conformal quantiles for a single forecaster checkpoint.

    `residual_quantile_close` is the (1 - alpha) quantile of `|y - y_hat|`
    measured on the calibration fold for the close head; the volatility head
    uses its own quantile. `nominal_coverage` is `1 - alpha`. Apply at
    inference time as `[y_hat - q, y_hat + q]` for symmetric two-sided bands.

    For classification-mode checkpoints, ``softmax_quantile`` carries the
    APS threshold (Romano et al. 2020) fitted on the same calibration
    partition using ``1 - softmax[y_true]`` as the non-conformity score.
    Pre-#216 manifests without the field load with ``softmax_quantile=None``
    and the inference path falls back to uncalibrated max-softmax confidence.

    #292 extension: ``rates_residual_quantiles`` maps the per-head short
    name (``2y`` / ``5y`` / ``terminal``) to the (1 - alpha) absolute-
    residual quantile in **raw bps** -- the inference path applies
    ``[y_hat - q, y_hat + q]`` as the conformal bps band. Per-head aux
    classification surfaces use ``rates_softmax_quantiles`` which maps
    the same short name to the APS threshold for the per-head
    (easing / neutral / tightening) classifier. Both default to empty
    dicts so pre-#292 manifests round-trip clean.
    """

    alpha: float
    nominal_coverage: float
    residual_quantile_close: float
    residual_quantile_volatility: float
    calibration_n: int
    notes: str | None = None
    softmax_quantile: float | None = None
    rates_residual_quantiles: dict[str, float] | None = None
    rates_softmax_quantiles: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "nominal_coverage": self.nominal_coverage,
            "residual_quantile_close": self.residual_quantile_close,
            "residual_quantile_volatility": self.residual_quantile_volatility,
            "calibration_n": self.calibration_n,
            "notes": self.notes,
            "softmax_quantile": self.softmax_quantile,
            "rates_residual_quantiles": (
                dict(self.rates_residual_quantiles)
                if self.rates_residual_quantiles
                else None
            ),
            "rates_softmax_quantiles": (
                dict(self.rates_softmax_quantiles)
                if self.rates_softmax_quantiles
                else None
            ),
        }


def split_conformal_quantile(residuals: Sequence[float], alpha: float) -> float:
    """Empirical (1 - alpha) quantile with the finite-sample correction.

    Source: Lei & Wasserman (2014), "Distribution-Free Prediction Bands". The
    correction multiplies (1 - alpha) by (n + 1) / n so the band carries the
    desired coverage on hold-out points. Residuals must be non-negative
    absolute errors.
    """

    cleaned = sorted(float(abs(r)) for r in residuals if math.isfinite(float(r)))
    n = len(cleaned)
    if n == 0:
        raise ValueError("Calibration residual set is empty.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1); got {alpha!r}.")
    rank = math.ceil((1.0 - alpha) * (n + 1))
    rank = max(1, min(n, rank))
    return cleaned[rank - 1]


def calibrate_split_conformal(
    *,
    close_predictions: Sequence[float],
    close_actuals: Sequence[float],
    volatility_predictions: Sequence[float],
    volatility_actuals: Sequence[float],
    alpha: float = 0.2,
    notes: str | None = None,
) -> ConformalManifest:
    if len(close_predictions) != len(close_actuals):
        raise ValueError("close_predictions and close_actuals must align in length.")
    if len(volatility_predictions) != len(volatility_actuals):
        raise ValueError("volatility_predictions and volatility_actuals must align in length.")
    close_resid = [
        actual - pred for pred, actual in zip(close_predictions, close_actuals)
    ]
    vol_resid = [
        actual - pred for pred, actual in zip(volatility_predictions, volatility_actuals)
    ]
    return ConformalManifest(
        alpha=float(alpha),
        nominal_coverage=1.0 - float(alpha),
        residual_quantile_close=split_conformal_quantile(close_resid, alpha),
        residual_quantile_volatility=split_conformal_quantile(vol_resid, alpha),
        calibration_n=len(close_predictions),
        notes=notes,
    )


def apply_conformal_bands(
    *,
    close_predictions: Sequence[float],
    volatility_predictions: Sequence[float],
    manifest: ConformalManifest,
    horizon_scale: bool = True,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return (close_lower, close_upper, vol_lower, vol_upper) using the
    manifest's residual quantiles. `horizon_scale=True` widens the band by
    sqrt(step) so multi-step forecasts inherit the usual variance scaling.

    Note: the marginal (1 - alpha) coverage guarantee from split-conformal
    holds only for step 1. With ``horizon_scale=True`` the multi-step bands
    are a random-walk heuristic, not a calibrated conformal interval — treat
    `manifest.nominal_coverage` as a single-step quantity. Pass
    ``horizon_scale=False`` if you need uniform width across the horizon.
    """

    close_lower: list[float] = []
    close_upper: list[float] = []
    vol_lower: list[float] = []
    vol_upper: list[float] = []
    for step_idx, (pred_close, pred_vol) in enumerate(
        zip(close_predictions, volatility_predictions), start=1
    ):
        scale = math.sqrt(step_idx) if horizon_scale else 1.0
        close_w = manifest.residual_quantile_close * scale
        vol_w = manifest.residual_quantile_volatility * scale
        close_lower.append(min(max(0.0, pred_close - close_w), pred_close))
        close_upper.append(pred_close + close_w)
        vol_lower.append(min(max(0.0, pred_vol - vol_w), pred_vol))
        vol_upper.append(pred_vol + vol_w)
    return close_lower, close_upper, vol_lower, vol_upper


def calibrate_rates_regression_conformal(
    *,
    predictions_bps: Sequence[float],
    actuals_bps: Sequence[float],
    alpha: float = 0.2,
) -> float:
    """Fit the (1 - alpha) absolute-residual quantile for one rates head.

    Inputs are paired predictions / observations in **raw bps**; the
    helper returns the conformal band half-width the inference path
    applies symmetrically as ``[y_hat - q, y_hat + q]``. Same Lei-
    Wasserman correction the close / volatility regression helper uses;
    rows with non-finite values are dropped silently after the length
    check.
    """

    if len(predictions_bps) != len(actuals_bps):
        raise ValueError(
            f"predictions_bps ({len(predictions_bps)}) and actuals_bps "
            f"({len(actuals_bps)}) must align in length."
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1); got {alpha!r}.")
    residuals: list[float] = []
    for pred, actual in zip(predictions_bps, actuals_bps):
        if pred is None or actual is None:
            continue
        try:
            pf = float(pred)
            af = float(actual)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(pf) and math.isfinite(af)):
            continue
        residuals.append(af - pf)
    if not residuals:
        raise ValueError("Calibration residual set is empty after filtering.")
    return split_conformal_quantile(residuals, alpha)


def calibrate_classification_conformal(
    *,
    softmax_scores: Sequence[Sequence[float]],
    true_classes: Sequence[int],
    alpha: float = DEFAULT_CLASSIFICATION_ALPHA,
) -> float:
    """Fit the APS threshold (Romano et al. 2020) on a calibration partition.

    The non-conformity score for row i is ``1 - softmax[i, true_classes[i]]``
    — high score when the model is uncertain about the truth, low when it
    is confident on the right class. The threshold is the (1 - alpha)
    finite-sample-corrected quantile of those scores via the same Lei-
    Wasserman rank formula the regression helper uses.

    Inputs must align: ``len(softmax_scores) == len(true_classes)``. Rows
    whose softmax does not include the true class index (e.g. truncated /
    malformed) are dropped silently after a length sanity check.
    """

    if len(softmax_scores) != len(true_classes):
        raise ValueError(
            f"softmax_scores ({len(softmax_scores)}) and true_classes "
            f"({len(true_classes)}) must align in length."
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1); got {alpha!r}.")
    nonconformity: list[float] = []
    for row, true_idx in zip(softmax_scores, true_classes):
        idx = int(true_idx)
        if idx < 0 or idx >= len(row):
            continue
        prob = float(row[idx])
        if not math.isfinite(prob):
            continue
        nonconformity.append(1.0 - prob)
    if not nonconformity:
        raise ValueError("Calibration softmax set is empty after filtering.")
    return split_conformal_quantile(nonconformity, alpha)


def predict_conformal_set(
    softmax_probs: Sequence[float],
    threshold: float,
) -> list[int]:
    """Build the APS prediction set for one row's softmax distribution.

    Includes every class ``j`` whose ``1 - softmax[j] <= threshold``,
    i.e. ``softmax[j] >= 1 - threshold``. When no class clears the
    threshold (pathological row), falls back to ``[argmax]`` rather
    than emitting an empty set — the empty-set case is mathematically
    valid under APS but useless as a decision-support surface, and
    the fallback keeps the marginal coverage guarantee asymptotically
    valid because the row contributes a singleton instead of zero.
    """

    if not softmax_probs:
        return []
    keep = float(1.0 - threshold)
    included = [i for i, p in enumerate(softmax_probs) if float(p) >= keep]
    if included:
        return included
    argmax_idx = max(range(len(softmax_probs)), key=lambda i: float(softmax_probs[i]))
    return [argmax_idx]


def empirical_classification_coverage(
    predicted_sets: Sequence[Sequence[int]],
    true_classes: Sequence[int],
) -> float:
    """Fraction of rows where ``true_classes[i] in predicted_sets[i]``."""

    if len(predicted_sets) != len(true_classes):
        raise ValueError(
            f"predicted_sets ({len(predicted_sets)}) and true_classes "
            f"({len(true_classes)}) must align in length."
        )
    if not predicted_sets:
        return float("nan")
    inside = sum(
        1 for s, y in zip(predicted_sets, true_classes) if int(y) in {int(x) for x in s}
    )
    return inside / len(predicted_sets)


def format_class_set_label(
    predicted_set: Sequence[int],
    class_names: Sequence[str],
) -> str:
    """Emit ``"{normal, high}"``-style label for the UI card.

    Renders class indices through ``class_names`` and wraps in braces.
    Empty input → ``"{}"``. Unknown indices fall through as ``"?"`` so
    a stale manifest still produces a readable string rather than
    raising in the response serializer.
    """

    labels = [
        str(class_names[i]) if 0 <= int(i) < len(class_names) else "?"
        for i in predicted_set
    ]
    return "{" + ", ".join(labels) + "}"


def empirical_coverage(
    *,
    predictions: Sequence[float],
    actuals: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> float:
    if not (len(predictions) == len(actuals) == len(lower) == len(upper)):
        raise ValueError("predictions, actuals, lower, upper must align in length.")
    if not predictions:
        return float("nan")
    inside = sum(
        1
        for actual, lo, hi in zip(actuals, lower, upper)
        if math.isfinite(actual) and lo <= actual <= hi
    )
    return inside / len(predictions)


def load_manifest(path: Path | str) -> ConformalManifest:
    """Read a JSON manifest. Residual quantile fields default to 0.0
    when absent (classification-only manifests written by
    ``save_manifest`` drop them); the inference loader treats a 0.0
    residual quantile as "no regression bands available" and falls
    back to gaussian-z.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Conformal manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Conformal manifest must be a JSON object: {path}")
    softmax_quantile_raw = payload.get("softmax_quantile")
    rates_residuals_raw = payload.get("rates_residual_quantiles")
    rates_softmax_raw = payload.get("rates_softmax_quantiles")
    rates_residuals: dict[str, float] | None = None
    rates_softmax: dict[str, float] | None = None
    if isinstance(rates_residuals_raw, Mapping):
        rates_residuals = {
            str(k): float(v)
            for k, v in rates_residuals_raw.items()
            if v is not None
        }
        if not rates_residuals:
            rates_residuals = None
    if isinstance(rates_softmax_raw, Mapping):
        rates_softmax = {
            str(k): float(v)
            for k, v in rates_softmax_raw.items()
            if v is not None
        }
        if not rates_softmax:
            rates_softmax = None
    return ConformalManifest(
        alpha=float(payload["alpha"]),
        nominal_coverage=float(payload["nominal_coverage"]),
        residual_quantile_close=float(payload.get("residual_quantile_close", 0.0)),
        residual_quantile_volatility=float(
            payload.get("residual_quantile_volatility", 0.0)
        ),
        calibration_n=int(payload["calibration_n"]),
        notes=payload.get("notes"),
        softmax_quantile=(
            float(softmax_quantile_raw) if softmax_quantile_raw is not None else None
        ),
        rates_residual_quantiles=rates_residuals,
        rates_softmax_quantiles=rates_softmax,
    )


def save_manifest(manifest: ConformalManifest, path: Path | str) -> Path:
    """Persist a manifest atomically via temp file + ``Path.replace``.

    The temp-and-rename pattern means a mid-write process crash leaves
    the original sidecar intact rather than producing a half-written
    JSON the inference loader would later fail on. Same destination
    path on success; the temp file is unlinked even on failure.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_dict()
    # Drop residual_quantile_* fields entirely when both are zero so a
    # classification-only manifest is not mistaken for a regression
    # band manifest at inference time (the inference loader treats
    # any non-None manifest as conformal, so leaving the zeros in
    # would produce zero-width prediction bands).
    if (
        payload.get("residual_quantile_close") == 0.0
        and payload.get("residual_quantile_volatility") == 0.0
    ):
        payload.pop("residual_quantile_close", None)
        payload.pop("residual_quantile_volatility", None)
    payload = {k: v for k, v in payload.items() if v is not None}
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    return path


def to_jsonable(manifest: ConformalManifest) -> dict[str, float | int | str | None]:
    return asdict(manifest)


def bootstrap_ci_columns(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_key: str = "samples",
    block_size: int = 6,
    n_resamples: int = 1000,
    coverage: float = 0.95,
    seed: int = 11,
) -> list[dict[str, float | int | str | None]]:
    """Augment aggregator rows with `ci_lo` / `ci_hi` columns derived from a
    moving-block bootstrap.  Each row must carry the raw `samples` list so the
    bootstrap can resample; rows without samples fall through with `None` CIs.
    """

    from app.evaluation.bootstrap import block_bootstrap_ci

    out: list[dict[str, float | int | str | None]] = []
    for row in rows:
        result: dict[str, float | int | str | None] = {k: v for k, v in row.items() if k != sample_key}
        samples = row.get(sample_key)
        if isinstance(samples, Sequence) and len(samples) > 1:
            ci = block_bootstrap_ci(
                list(samples),
                block_size=block_size,
                n_resamples=n_resamples,
                coverage=coverage,
                seed=seed,
            )
            result["ci_lo"] = float(ci.lo)
            result["ci_hi"] = float(ci.hi)
        else:
            result["ci_lo"] = None
            result["ci_hi"] = None
        out.append(result)
    return out
