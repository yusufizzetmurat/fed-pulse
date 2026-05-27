"""1-seed x 1-fold reproducibility smoke runner (#335).

Drives ``app.train_forecaster`` in dual-head mode against the canonical
training package for one seed and one fold, then extracts the
``regime_f1_macro`` metric from the sweep report and hands it to
``scripts.assert_reproducibility_smoke`` for tolerance checking.

This is the CI entrypoint behind ``make reproduce-smoke``. The
``make reproduce-all`` target stays the docker-compose / full-pipeline
flow; this script is the smaller native-python equivalent the
``reproduce-smoke`` GitHub Actions workflow calls so the contract can
run on the standard ``ubuntu-latest`` runner without a docker daemon.

Exits non-zero on any of: training failure, missing sweep report,
missing metric cell, or assertion failure. Each failure prints a clear
message so a reviewer reading the workflow log sees the root cause.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "1-seed x 1-fold canonical dual-head smoke + numerical-contract "
            "assertion (#335)."
        )
    )
    parser.add_argument(
        "--training-package-id",
        required=True,
        help="Training package id under data/processed/.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Single seed to train (default 11; first of the official set).",
    )
    parser.add_argument(
        "--fold-id",
        default="wf_fold_1",
        help="Walk-forward fold id from the package's fold manifest.",
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=REPO_ROOT / "tests" / "regression" / "reproducibility_reference.json",
        help="Path to the pinned reference JSON.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPO_ROOT / "backend" / "artifacts" / "reproduce_smoke" / "forecaster_sweep_results.json",
        help="Where the training run's sweep report is written.",
    )
    return parser.parse_args()


def _pull_training_package(training_package_id: str) -> None:
    """Pull the canonical training package via the existing reproduce_all script.

    Reuses the production-tested artefact pull so the smoke + the full
    fresh-machine reproduction share the same code path. Skips the pull
    when the package is already on disk so a re-run on the same runner
    does not re-download.
    """

    target = REPO_ROOT / "data" / "processed" / training_package_id
    if target.exists() and any(target.iterdir()):
        print(f"[reproduce-smoke] training package already on disk at {target}", flush=True)
        return

    canonical_target = REPO_ROOT / "data" / "processed" / "canonical"
    env = {**os.environ, "FED_PULSE_REPRODUCE_TP_ID": training_package_id}
    print(
        f"[reproduce-smoke] pulling training package {training_package_id} via "
        "scripts/reproduce_all.py (artefact pull only) ...",
        flush=True,
    )
    # ``reproduce_all.py`` pulls + runs a 1-epoch training; we only need
    # the pull here, so we exec the import surface directly.
    pull_cmd = [
        sys.executable,
        "-c",
        "import sys; sys.path.insert(0, 'scripts'); "
        "from reproduce_all import _ensure_training_package; "
        "_ensure_training_package()",
    ]
    result = subprocess.run(pull_cmd, cwd=str(REPO_ROOT), env=env)
    if result.returncode != 0:
        raise SystemExit(
            f"[reproduce-smoke] training-package pull failed (exit={result.returncode}); "
            "see logs above"
        )

    # ``reproduce_all._ensure_training_package`` copies into
    # ``data/processed/<FED_PULSE_REPRODUCE_TP_ID>``. Align it with the
    # canonical id the trainer expects so downstream consumers find the
    # parquet under the same name they trained against.
    if canonical_target.exists() and not target.exists():
        shutil.copytree(canonical_target, target)


def _run_smoke_training(
    *,
    training_package_id: str,
    seed: int,
    fold_id: str,
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "app.train_forecaster",
        "--training-package-id",
        training_package_id,
        "--seed",
        str(seed),
        "--epochs",
        "1",
        "--head-mode",
        "dual",
        "--output-mode",
        "classification",
        "--folds",
        fold_id,
        "--protocol",
        "walk-forward",
        "--report-path",
        str(report_path),
    ]
    print(f"[reproduce-smoke] training: {' '.join(cmd)}", flush=True)
    env = {**os.environ, "PYTHONPATH": str(BACKEND_DIR)}
    result = subprocess.run(cmd, cwd=str(BACKEND_DIR), env=env)
    if result.returncode != 0:
        raise SystemExit(
            f"[reproduce-smoke] smoke training failed (exit={result.returncode}); "
            "see backend logs above"
        )


def _extract_regime_f1_macro(report_path: Path) -> float:
    if not report_path.exists():
        raise SystemExit(f"[reproduce-smoke] sweep report missing at {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    trials = payload.get("trials") or []
    if not trials:
        raise SystemExit(
            f"[reproduce-smoke] sweep report at {report_path} has no trials"
        )
    # The smoke runs a single (seed, fold, hp) cell, so the trial set
    # collapses to one record. Pick the first selected trial when
    # available (mirrors the production trial-selection rule) else fall
    # back to the first record.
    trial = next(
        (record for record in trials if record.get("selected")),
        trials[0],
    )
    metrics = (trial.get("summary") or {}).get("metrics") or {}
    value = metrics.get("regime_f1_macro")
    if value is None:
        # The trainer emits regime_f1_macro inside val_metrics + test_metrics
        # on the walk-forward path; pick test when present so the smoke
        # reads the same surface the canonical-comparison sweep aggregates.
        test_metrics = (trial.get("summary") or {}).get("test_metrics") or {}
        value = test_metrics.get("regime_f1_macro")
    if value is None:
        raise SystemExit(
            "[reproduce-smoke] regime_f1_macro absent from the sweep report; "
            "check that the dual-head training surface still emits the metric"
        )
    return float(value)


def main() -> int:
    args = _parse_args()
    _pull_training_package(args.training_package_id)
    _run_smoke_training(
        training_package_id=args.training_package_id,
        seed=args.seed,
        fold_id=args.fold_id,
        report_path=args.report_path,
    )
    observed = _extract_regime_f1_macro(args.report_path)
    assert_cmd = [
        sys.executable,
        "-m",
        "scripts.assert_reproducibility_smoke",
        "--observed-value",
        f"{observed:.6f}",
        "--reference-path",
        str(args.reference_path),
    ]
    print(f"[reproduce-smoke] asserting: {' '.join(assert_cmd)}", flush=True)
    result = subprocess.run(assert_cmd, cwd=str(REPO_ROOT))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
