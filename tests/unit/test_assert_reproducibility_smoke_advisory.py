"""#427: assert_reproducibility_smoke is advisory and never gates merges.

The reproduce-smoke job emits a WARN diff line when the observed metric
drifts outside the pinned tolerance but exits 0 so the merge proceeds.
This test pins that behaviour so a future change cannot silently flip
the gate back to a hard fail (re-introducing the trap that closed #427).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "assert_reproducibility_smoke.py"


def _write_reference(tmp_path: Path, *, value: float, tol: float) -> Path:
    payload = {
        "training_package_id": "tp_test",
        "head_mode": "dual",
        "metric": "regime_f1_macro",
        "reference_value": value,
        "tolerance": tol,
        "source_artefact": "test fixture",
        "source_artefact_cell": "synthetic",
    }
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run(reference_path: Path, observed: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--observed-value",
            str(observed),
            "--reference-path",
            str(reference_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_inside_tolerance_exits_zero_with_ok(tmp_path: Path) -> None:
    ref = _write_reference(tmp_path, value=0.4, tol=0.005)
    result = _run(ref, 0.4012)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_outside_tolerance_exits_zero_with_warn(tmp_path: Path) -> None:
    ref = _write_reference(tmp_path, value=0.142857, tol=0.005)
    result = _run(ref, 0.405372)
    assert result.returncode == 0, (
        "advisory under #427: out-of-tolerance must not gate the merge; "
        "got rc={}, stdout={}".format(result.returncode, result.stdout)
    )
    assert "WARN" in result.stdout
    assert "diff=" in result.stdout


def test_outside_tolerance_in_other_direction_also_exits_zero(tmp_path: Path) -> None:
    ref = _write_reference(tmp_path, value=0.5, tol=0.005)
    result = _run(ref, 0.1)
    assert result.returncode == 0
    assert "WARN" in result.stdout
