"""Schema guard for the reproducibility reference cell (#335).

The pinned reference JSON drives the ``reproduce-smoke`` CI workflow.
If a future change renames a required key or drops a value the
workflow asserts against, the assertion script would fail in CI with a
confusing error. This test pins the reference cell's shape so the
failure surfaces at PR review time instead of inside a 25-minute
workflow run.

The reference describes the deterministic 1-seed x 1-fold x 1-epoch
cold-start cell the smoke must reproduce on every push to dev. It is
not a slice of the canonical-comparison sweep artefact (a 1-epoch
single-fold cell cannot equal the 5-seed x N-fold sweep mean by
construction), so the test deliberately does not extract a value out
of any on-disk sweep artefact.
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


def test_source_artefact_cell_is_non_empty_description(
    reference_payload: dict[str, object],
) -> None:
    cell = str(reference_payload["source_artefact_cell"])
    assert cell.strip(), "source_artefact_cell must carry a non-empty description"
    assert any(ch.isalpha() for ch in cell), (
        f"source_artefact_cell '{cell}' should describe the cell, not be a "
        "bare number or punctuation"
    )
