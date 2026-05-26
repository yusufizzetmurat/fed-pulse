"""Numerical-contract guard for the reproducibility CI smoke (#335).

Reads the pinned reference cell at
``tests/regression/reproducibility_reference.json`` and the metric the
just-finished smoke training emitted onto disk, then asserts the
observed value sits within the reference tolerance. Exits 1 with a
clear diff on mismatch so a future training-pipeline change cannot
silently drift the canonical headline.

CLI shape::

    python -m scripts.assert_reproducibility_smoke \
        --observed-value 0.4193 \
        [--reference-path tests/regression/reproducibility_reference.json]

The workflow extracts ``observed_value`` from the smoke run's sweep
report and pipes it in via ``--observed-value``. Keeping the metric
extraction out of this script lets it stay framework-agnostic — any
training pipeline that can write a single macro-F1 number to stdout
plugs in unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE_PATH = REPO_ROOT / "tests" / "regression" / "reproducibility_reference.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assert a smoke training's macro-F1 stays within the pinned "
            "reproducibility tolerance (#335 numerical-contract guard)."
        )
    )
    parser.add_argument(
        "--observed-value",
        type=float,
        required=True,
        help="Macro-F1 (or other pinned metric) from the just-finished smoke run.",
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=DEFAULT_REFERENCE_PATH,
        help="Path to the pinned reference JSON. Defaults to the in-repo location.",
    )
    return parser.parse_args()


def _load_reference(path: Path) -> dict[str, object]:
    if not path.exists():
        raise SystemExit(f"reference file missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"reference file is not valid JSON ({path}): {exc}") from exc
    required = {"reference_value", "tolerance", "metric"}
    missing = required - set(payload)
    if missing:
        raise SystemExit(f"reference file missing required keys {sorted(missing)}: {path}")
    return payload


def main() -> int:
    args = _parse_args()
    reference = _load_reference(args.reference_path)

    reference_value = float(reference["reference_value"])  # type: ignore[arg-type]
    tolerance = float(reference["tolerance"])  # type: ignore[arg-type]
    metric = str(reference["metric"])
    observed = float(args.observed_value)
    diff = observed - reference_value

    print(
        f"[reproduce-smoke] metric={metric} observed={observed:.6f} "
        f"reference={reference_value:.6f} tolerance=+/-{tolerance:.6f} "
        f"diff={diff:+.6f}",
        flush=True,
    )

    if abs(diff) > tolerance:
        print(
            "[reproduce-smoke] FAIL — observed metric drifted outside the "
            "pinned tolerance. Investigate before merging.",
            flush=True,
        )
        print(
            f"  metric:    {metric}\n"
            f"  observed:  {observed:.6f}\n"
            f"  reference: {reference_value:.6f}\n"
            f"  tolerance: +/-{tolerance:.6f}\n"
            f"  diff:      {diff:+.6f}\n"
            f"  source:    {reference.get('source_artefact')} "
            f"(cell {reference.get('source_artefact_cell')})",
            flush=True,
        )
        return 1

    print(
        "[reproduce-smoke] OK — observed metric inside the pinned tolerance.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
