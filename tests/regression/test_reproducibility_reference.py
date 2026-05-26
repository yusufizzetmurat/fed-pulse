"""Schema + extraction guard for the reproducibility reference cell (#335).

The pinned reference JSON drives the ``reproduce-smoke`` CI workflow.
If a future change to the canonical-comparison sweep renames a cell or
drops a key the workflow asserts against, the assertion script would
fail in CI with a confusing error. This test pins the reference cell's
shape and the source-artefact path it points at so the failure surfaces
at PR review time instead of inside a 25-minute workflow run.

The test deliberately does NOT import any backend modules — the
reference is plain JSON the workflow consumes before any training code
runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = REPO_ROOT / "tests" / "regression" / "reproducibility_reference.json"
REQUIRED_KEYS = {
    "training_package_id",
    "head_mode",
    "metric",
    "reference_value",
    "tolerance",
    "source_artefact",
    "source_artefact_cell",
}


@pytest.fixture(scope="module")
def reference_payload() -> dict[str, object]:
    assert REFERENCE_PATH.exists(), f"reference JSON missing at {REFERENCE_PATH}"
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


def test_reference_json_carries_required_keys(reference_payload: dict[str, object]) -> None:
    missing = REQUIRED_KEYS - set(reference_payload)
    assert not missing, f"reference JSON missing required keys: {sorted(missing)}"


def test_reference_value_and_tolerance_are_floats(reference_payload: dict[str, object]) -> None:
    assert isinstance(reference_payload["reference_value"], (int, float))
    assert isinstance(reference_payload["tolerance"], (int, float))
    assert 0.0 < float(reference_payload["tolerance"]) < 1.0, (
        "tolerance must sit in (0, 1); a 0.5 tolerance trivially passes and a "
        "0.0 tolerance can never pass"
    )
    assert 0.0 <= float(reference_payload["reference_value"]) <= 1.0, (
        "reference value is a macro-F1; it must sit in [0, 1]"
    )


def test_source_artefact_cell_is_dotted_path(reference_payload: dict[str, object]) -> None:
    cell = str(reference_payload["source_artefact_cell"])
    parts = cell.split(".")
    assert len(parts) >= 3, (
        f"source_artefact_cell '{cell}' should be a dotted path with at least "
        "three segments (e.g. 'summary.dual.regime_f1_macro.mean')"
    )
    assert parts[0] == "summary", (
        f"source_artefact_cell '{cell}' should start at the 'summary' key of "
        "the canonical-comparison output"
    )


def test_source_artefact_cell_extracts_cleanly_when_present(
    reference_payload: dict[str, object],
) -> None:
    """If the canonical sweep artefact is on disk, the pinned cell must extract.

    The artefact is not in git (it is large + regenerated on every sweep
    revision), so the test skips when absent. When present — for instance
    on a developer box that just ran the canonical sweep — the cell must
    extract cleanly and match the pinned reference within the tolerance.
    """

    artefact_path = REPO_ROOT / str(reference_payload["source_artefact"])
    if not artefact_path.exists():
        pytest.skip(f"canonical sweep artefact absent at {artefact_path}")

    payload = json.loads(artefact_path.read_text(encoding="utf-8"))
    cursor: object = payload
    for part in str(reference_payload["source_artefact_cell"]).split("."):
        assert isinstance(cursor, dict), (
            f"source_artefact_cell traversal hit a non-dict at part '{part}' "
            f"of '{reference_payload['source_artefact_cell']}'"
        )
        assert part in cursor, (
            f"source_artefact_cell '{reference_payload['source_artefact_cell']}' "
            f"missing key '{part}' in {artefact_path.name}"
        )
        cursor = cursor[part]
    assert isinstance(cursor, (int, float)), (
        f"source_artefact_cell extracted a non-numeric value: {cursor!r}"
    )

    diff = abs(float(cursor) - float(reference_payload["reference_value"]))
    tolerance = float(reference_payload["tolerance"])
    assert diff <= tolerance, (
        f"pinned reference_value {reference_payload['reference_value']} drifted "
        f"from the source artefact cell value {cursor} by {diff:.6f} "
        f"(tolerance {tolerance}); re-pin the reference or investigate the "
        "sweep change"
    )
