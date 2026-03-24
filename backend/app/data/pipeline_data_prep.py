from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path("/data") if Path("/data").exists() else BACKEND_ROOT.parent / "data"
DEFAULT_OUTPUT_ROOT = BACKEND_ROOT.parent / "artifacts" / "data_prep"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full data preparation pipeline (ingest -> normalize -> quality -> package -> run specs)."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Base data directory.")
    parser.add_argument("--dataset-version", required=True, help="Dataset version id.")
    parser.add_argument("--feature-version", required=True, help="Feature version id.")
    parser.add_argument("--training-package-id", default="", help="Optional explicit training package id.")
    parser.add_argument("--owner", default="unknown", help="Owner for generated run specs.")
    parser.add_argument("--near-threshold", type=float, default=0.97, help="Near-duplicate threshold.")
    parser.add_argument("--include-hf", action="store_true", help="Include Hugging Face ingestion.")
    parser.add_argument("--include-kaggle", action="store_true", help="Include Kaggle ingestion.")
    parser.add_argument("--include-scraped", action="store_true", help="Include scraped ingestion.")
    parser.add_argument(
        "--all-sources",
        action="store_true",
        help="Include all sources (overrides source-specific flags).",
    )
    parser.add_argument("--skip-generate-specs", action="store_true", help="Skip baseline spec generation.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    return parser.parse_args()


def _auto_training_package_id(dataset_version: str, feature_version: str) -> str:
    return f"tp_{dataset_version}_{feature_version}_epv1_v1.0"


def _run_step(command: list[str], *, dry_run: bool, log_file: Path) -> int:
    cmd_str = " ".join(shlex.quote(part) for part in command)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {cmd_str}\n")
    print(f"\n[data-prep] {cmd_str}")
    if dry_run:
        return 0
    process = subprocess.run(command, capture_output=True, text=True)
    with log_file.open("a", encoding="utf-8") as handle:
        if process.stdout:
            handle.write(process.stdout)
        if process.stderr:
            handle.write(process.stderr)
        handle.write(f"[exit_code] {process.returncode}\n")
    if process.stdout:
        print(process.stdout, end="")
    if process.returncode != 0 and process.stderr:
        print(process.stderr, end="", file=sys.stderr)
    return process.returncode


def main() -> int:
    args = _parse_args()
    data_dir = Path(args.data_dir)
    output_root = DEFAULT_OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    run_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_file = output_root / f"data_prep_{run_token}.log"

    include_hf = args.all_sources or args.include_hf
    include_kaggle = args.all_sources or args.include_kaggle
    include_scraped = args.all_sources or args.include_scraped
    if not (include_hf or include_kaggle or include_scraped):
        include_hf = include_kaggle = include_scraped = True

    training_package_id = args.training_package_id or _auto_training_package_id(
        args.dataset_version, args.feature_version
    )

    python_exec = sys.executable
    ingest_cmd = [
        python_exec,
        "-m",
        "app.data.source_ingestion",
        "--data-dir",
        str(data_dir),
    ]
    if include_hf:
        ingest_cmd.append("--include-hf")
    if include_kaggle:
        ingest_cmd.append("--include-kaggle")
    if include_scraped:
        ingest_cmd.append("--include-scraped")

    normalize_cmd = [
        python_exec,
        "-m",
        "app.data.label_normalization",
        "--input",
        str(data_dir / "raw" / "phase2" / "source_registry.jsonl"),
        "--output",
        str(data_dir / "interim" / "phase2" / "registry_labeled.jsonl"),
    ]
    quality_cmd = [
        python_exec,
        "-m",
        "app.data.quality_validation",
        "--input",
        str(data_dir / "interim" / "phase2" / "registry_labeled.jsonl"),
        "--output",
        str(data_dir / "interim" / "phase2" / "registry_quality_passed.jsonl"),
        "--near-threshold",
        str(args.near_threshold),
    ]
    package_cmd = [
        python_exec,
        "-m",
        "app.data.training_package_builder",
        "--input",
        str(data_dir / "interim" / "phase2" / "registry_quality_passed.jsonl"),
        "--dataset-version",
        args.dataset_version,
        "--feature-version",
        args.feature_version,
        "--training-package-id",
        training_package_id,
    ]
    generate_specs_cmd = [
        python_exec,
        "-m",
        "app.data.baseline_spec_generator",
        "--dataset-version",
        args.dataset_version,
        "--feature-version",
        args.feature_version,
        "--training-package-id",
        training_package_id,
        "--owner",
        args.owner,
    ]

    steps: list[list[str]] = [ingest_cmd, normalize_cmd, quality_cmd, package_cmd]
    if not args.skip_generate_specs:
        steps.append(generate_specs_cmd)

    with log_file.open("w", encoding="utf-8") as handle:
        handle.write(
            f"data_prep_started_at={datetime.now(timezone.utc).isoformat()}\n"
            f"dataset_version={args.dataset_version}\n"
            f"feature_version={args.feature_version}\n"
            f"training_package_id={training_package_id}\n"
            f"data_dir={data_dir}\n"
        )

    for step in steps:
        code = _run_step(step, dry_run=args.dry_run, log_file=log_file)
        if code != 0:
            print(f"\nData prep pipeline failed. See log: {log_file}", file=sys.stderr)
            return code

    print("\nData prep pipeline completed successfully.")
    print(f"Training package id: {training_package_id}")
    print(f"Log file: {log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
