from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ConformalManifest:
    """Split-conformal quantiles for a single forecaster checkpoint.

    `residual_quantile_close` is the (1 - alpha) quantile of `|y - y_hat|`
    measured on the calibration fold for the close head; the volatility head
    uses its own quantile. `nominal_coverage` is `1 - alpha`. Apply at
    inference time as `[y_hat - q, y_hat + q]` for symmetric two-sided bands.
    """

    alpha: float
    nominal_coverage: float
    residual_quantile_close: float
    residual_quantile_volatility: float
    calibration_n: int
    notes: str | None = None

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "alpha": self.alpha,
            "nominal_coverage": self.nominal_coverage,
            "residual_quantile_close": self.residual_quantile_close,
            "residual_quantile_volatility": self.residual_quantile_volatility,
            "calibration_n": self.calibration_n,
            "notes": self.notes,
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
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Conformal manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Conformal manifest must be a JSON object: {path}")
    return ConformalManifest(
        alpha=float(payload["alpha"]),
        nominal_coverage=float(payload["nominal_coverage"]),
        residual_quantile_close=float(payload["residual_quantile_close"]),
        residual_quantile_volatility=float(payload["residual_quantile_volatility"]),
        calibration_n=int(payload["calibration_n"]),
        notes=payload.get("notes"),
    )


def save_manifest(manifest: ConformalManifest, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_dict()
    payload = {k: v for k, v in payload.items() if v is not None}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def to_jsonable(manifest: ConformalManifest) -> dict[str, float | int | str | None]:
    return asdict(manifest)


def bootstrap_ci_columns(
    rows: Iterable[Mapping],
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
