from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CoveragePoint:
    nominal: float
    empirical: float
    n: int


def empirical_coverage(residuals: Sequence[float], band_half_widths: Sequence[float]) -> float:
    if len(residuals) != len(band_half_widths):
        raise ValueError("residuals and band_half_widths must have equal length")
    if not residuals:
        return float("nan")
    inside = sum(1 for r, h in zip(residuals, band_half_widths) if abs(r) <= h)
    return inside / len(residuals)


def coverage_curve(
    residuals: Sequence[float],
    band_half_widths: Sequence[float],
    *,
    nominal_levels: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
) -> list[CoveragePoint]:
    if len(residuals) != len(band_half_widths):
        raise ValueError("residuals and band_half_widths must have equal length")
    if not residuals:
        return [CoveragePoint(nominal=lvl, empirical=float("nan"), n=0) for lvl in nominal_levels]

    sorted_abs = sorted(abs(r) for r in residuals)
    n = len(sorted_abs)
    base_scaled = [abs(r) / h if h > 0 else float("inf") for r, h in zip(residuals, band_half_widths)]

    points: list[CoveragePoint] = []
    for nominal in nominal_levels:
        scale = _scale_for_nominal(nominal)
        inside = sum(1 for s in base_scaled if s <= scale)
        points.append(CoveragePoint(nominal=nominal, empirical=inside / n, n=n))
    return points


def _scale_for_nominal(nominal: float) -> float:
    if not 0 < nominal < 1:
        raise ValueError(f"nominal must be in (0, 1); got {nominal}")
    # Inverse standard normal CDF approximation (Acklam, 2003) — good to ~1e-9
    # over [1e-15, 1 - 1e-15]; we only ever evaluate inside (0.5, 0.99).
    p = (1.0 + nominal) / 2.0
    return _inverse_normal_cdf(p)


def _inverse_normal_cdf(p: float) -> float:
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)


def load_predictions_jsonl(path: Path) -> tuple[list[float], list[float]]:
    residuals: list[float] = []
    band_half_widths: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        predicted = row.get("predicted")
        actual = row.get("actual")
        half_width = row.get("band_half_width")
        if predicted is None or actual is None or half_width is None:
            continue
        residuals.append(float(actual) - float(predicted))
        band_half_widths.append(float(half_width))
    return residuals, band_half_widths


def write_reliability_diagram(
    residuals: Sequence[float],
    band_half_widths: Sequence[float],
    output_path: Path,
    *,
    nominal_levels: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
) -> Path:
    points = coverage_curve(residuals, band_half_widths, nominal_levels=nominal_levels)
    payload = {
        "n": len(residuals),
        "points": [{"nominal": p.nominal, "empirical": p.empirical, "n": p.n} for p in points],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
