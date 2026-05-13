from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from app.evaluation.calibration import (
    coverage_curve,
    empirical_coverage,
    write_reliability_diagram,
)


def test_empirical_coverage_inside_all() -> None:
    residuals = [0.1, -0.2, 0.0, 0.15]
    bands = [1.0, 1.0, 1.0, 1.0]
    assert empirical_coverage(residuals, bands) == 1.0


def test_empirical_coverage_outside_all() -> None:
    residuals = [2.0, -3.0, 4.0]
    bands = [1.0, 1.0, 1.0]
    assert empirical_coverage(residuals, bands) == 0.0


def test_empirical_coverage_half() -> None:
    residuals = [0.5, -1.5, 0.5, 1.5]
    bands = [1.0, 1.0, 1.0, 1.0]
    assert empirical_coverage(residuals, bands) == 0.5


def test_coverage_curve_monotone_nondecreasing() -> None:
    residuals = [0.1 * i - 0.5 for i in range(11)]
    bands = [0.5] * 11
    points = coverage_curve(residuals, bands)
    empirical = [p.empirical for p in points]
    for prev, curr in zip(empirical, empirical[1:]):
        assert curr >= prev


def test_coverage_curve_close_to_gaussian_for_gaussian_residuals() -> None:
    rng_seed = 11
    import random
    rng = random.Random(rng_seed)
    residuals = [rng.gauss(0.0, 1.0) for _ in range(2000)]
    bands = [1.0] * len(residuals)
    points = {p.nominal: p.empirical for p in coverage_curve(residuals, bands)}
    assert abs(points[0.8] - 0.8) < 0.05
    assert abs(points[0.95] - 0.95) < 0.03


def test_write_reliability_diagram(tmp_path: Path) -> None:
    residuals = [0.1, -0.2, 0.5, -0.5, 0.0]
    bands = [1.0] * 5
    out = tmp_path / "reliability.json"
    write_reliability_diagram(residuals, bands, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n"] == 5
    assert any(p["nominal"] == 0.8 for p in payload["points"])


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        empirical_coverage([0.1, 0.2], [1.0])
    with pytest.raises(ValueError):
        coverage_curve([0.1, 0.2], [1.0])


def test_inverse_normal_cdf_at_known_points() -> None:
    from app.evaluation.calibration import _scale_for_nominal

    assert math.isclose(_scale_for_nominal(0.8), 1.28155, abs_tol=1e-3)
    assert math.isclose(_scale_for_nominal(0.95), 1.95996, abs_tol=1e-3)
