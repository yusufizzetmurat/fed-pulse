"""CLI driver for the Phase 5 multi-run logit-average ensemble.

Reads a YAML manifest listing N training runs (one run = one
config x seed x architecture combination), pools their per-fold
per-trial logits, drops redundant components via the pairwise
Cohen-kappa guard in
:func:`app.evaluation.ensemble_aggregator.aggregate_multi_run_ensemble`,
and writes a markdown summary + JSON metrics file.

YAML schema
-----------

Top-level: a list of run-spec entries (or a mapping with key
``runs`` whose value is the list). Each entry carries the fields
of :class:`app.evaluation.ensemble_aggregator.RunSpec`:

.. code-block:: yaml

    runs:
      - run_id: rgm_lstm_seed11
        architecture: lstm
        encoder_alias: none
        seed: 11
        results_path: artifacts/experiments/rgm_lstm_seed11/forecaster_sweep_results.json
        weight: 1.0
      - run_id: rgm_transformer_seed11
        architecture: transformer
        encoder_alias: bert_base
        seed: 11
        results_path: artifacts/experiments/rgm_transformer_seed11/forecaster_sweep_results.json

Optional top-level keys:

- ``conformal_alpha``: float, default 0.2 (matches the headline
  classifier's per-fold conformal alpha).
- ``redundancy_kappa_threshold``: float, default 0.85.
- ``n_classes``: int, default 3.

Run with::

    python -m app.evaluation.ensemble_run_specs \\
        --run-spec-file path/to/phase5_specs.yaml \\
        --output-dir artifacts/experiments/phase5_ensemble

The CLI deliberately does no model loading or training; it operates
purely on the on-disk sweep JSONs each ``results_path`` points at.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from app.evaluation.ensemble_aggregator import (
    DEFAULT_CONFORMAL_ALPHA,
    DEFAULT_REDUNDANCY_KAPPA_THRESHOLD,
    MultiRunEnsembleResult,
    RunSpec,
    aggregate_multi_run_ensemble,
)


def _load_run_specs(path: Path) -> tuple[list[RunSpec], dict[str, Any]]:
    """Parse the YAML manifest into ``RunSpec`` instances + the
    top-level options block.

    Accepts either a top-level list or a mapping with ``runs`` keying
    the list — the mapping form is preferred when callers want to
    co-locate runtime options like ``conformal_alpha`` with the run
    list, but a bare list is supported for terse manifests.
    """

    if not path.exists():
        raise FileNotFoundError(f"run spec file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    options: dict[str, Any] = {}
    if isinstance(raw, list):
        run_entries = raw
    elif isinstance(raw, Mapping):
        run_entries_raw = raw.get("runs")
        if not isinstance(run_entries_raw, list):
            raise ValueError(
                f"run spec file {path!s} must carry a top-level list or a "
                "mapping with a 'runs' list"
            )
        run_entries = run_entries_raw
        for key in ("conformal_alpha", "redundancy_kappa_threshold", "n_classes"):
            if key in raw:
                options[key] = raw[key]
    else:
        raise ValueError(
            f"run spec file {path!s} must be a YAML list or mapping; got " f"{type(raw).__name__}"
        )

    specs: list[RunSpec] = []
    for i, entry in enumerate(run_entries):
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"run spec entry {i} in {path!s} must be a mapping; got " f"{type(entry).__name__}"
            )
        try:
            run_id = str(entry["run_id"])
            architecture = str(entry["architecture"])
            encoder_alias = str(entry["encoder_alias"])
            seed = int(entry["seed"])
            results_path = str(entry["results_path"])
        except KeyError as exc:
            raise ValueError(
                f"run spec entry {i} in {path!s} missing required field {exc}"
            ) from exc
        weight = float(entry.get("weight", 1.0))
        specs.append(
            RunSpec(
                run_id=run_id,
                architecture=architecture,
                encoder_alias=encoder_alias,
                seed=seed,
                results_path=results_path,
                weight=weight,
            )
        )
    return specs, options


def _render_markdown(specs: Sequence[RunSpec], result: MultiRunEnsembleResult) -> str:
    lines: list[str] = []
    lines.append("# Phase 5 multi-run logit-average ensemble")
    lines.append("")
    lines.append(
        f"Calibration strategy: `{result.calibration_strategy}` "
        f"(alpha={result.conformal_alpha:.2f})"
    )
    lines.append("")
    lines.append("## Component runs")
    lines.append("")
    lines.append("| run_id | architecture | encoder_alias | seed | weight | kept |")
    lines.append("| --- | --- | --- | ---: | ---: | :---: |")
    kept_set = set(result.kept_run_ids)
    for spec in specs:
        kept = "yes" if spec.run_id in kept_set else "no"
        lines.append(
            f"| `{spec.run_id}` | {spec.architecture} | {spec.encoder_alias} "
            f"| {spec.seed} | {spec.weight:.2f} | {kept} |"
        )
    lines.append("")
    if result.dropped_run_ids:
        lines.append("### Redundancy drops")
        lines.append("")
        lines.append("| dropped | redundant_with | kappa |")
        lines.append("| --- | --- | ---: |")
        for dropped, kept, kappa in result.dropped_run_ids:
            lines.append(f"| `{dropped}` | `{kept}` | {kappa:.3f} |")
        lines.append("")
    lines.append("## Per-fold breakdown")
    lines.append("")
    lines.append("| fold | n | macro-F1 | coverage | avg set size |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in result.per_fold:
        lines.append(
            f"| {row.fold_id} | {row.n_rows} | {row.breakdown.macro_f1:.4f} "
            f"| {row.coverage:.3f} | {row.avg_set_size:.2f} |"
        )
    lines.append("")
    lines.append("## Pooled")
    lines.append("")
    lines.append(f"- macro-F1: **{result.pooled_breakdown.macro_f1:.4f}**")
    lines.append(f"- weighted-F1: {result.pooled_breakdown.weighted_f1:.4f}")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 5 multi-run logit-average ensemble. Reads a YAML "
            "manifest of training-run specs and emits a markdown "
            "summary + JSON metrics for the pooled macro-F1 and "
            "per-fold conformal coverage."
        )
    )
    parser.add_argument(
        "--run-spec-file",
        type=Path,
        required=True,
        help="Path to a YAML file listing the runs to ensemble.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory the markdown + JSON outputs are written to.",
    )
    parser.add_argument(
        "--conformal-alpha",
        type=float,
        default=None,
        help=(
            "Per-fold conformal alpha. Defaults to the value in the YAML "
            f"if present, otherwise {DEFAULT_CONFORMAL_ALPHA}."
        ),
    )
    parser.add_argument(
        "--redundancy-kappa-threshold",
        type=float,
        default=None,
        help=(
            "Cohen-kappa cutoff above which a run is treated as redundant. "
            f"Defaults to the YAML value if present, else {DEFAULT_REDUNDANCY_KAPPA_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=None,
        help="Number of target classes. Defaults to the YAML value or 3.",
    )
    args = parser.parse_args(argv)

    specs, options = _load_run_specs(args.run_spec_file)
    if not specs:
        raise SystemExit(f"run spec file {args.run_spec_file!s} is empty")

    conformal_alpha = (
        args.conformal_alpha
        if args.conformal_alpha is not None
        else float(options.get("conformal_alpha", DEFAULT_CONFORMAL_ALPHA))
    )
    redundancy_threshold = (
        args.redundancy_kappa_threshold
        if args.redundancy_kappa_threshold is not None
        else float(options.get("redundancy_kappa_threshold", DEFAULT_REDUNDANCY_KAPPA_THRESHOLD))
    )
    n_classes = args.n_classes if args.n_classes is not None else int(options.get("n_classes", 3))

    result = aggregate_multi_run_ensemble(
        specs,
        calibration_strategy="conformal_per_fold",
        redundancy_kappa_threshold=redundancy_threshold,
        conformal_alpha=conformal_alpha,
        n_classes=n_classes,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "ensemble_phase5_results.json"
    md_path = args.output_dir / "ensemble_phase5_results.md"
    json_path.write_text(
        json.dumps(
            {
                "run_specs": [spec.to_dict() for spec in specs],
                "result": result.to_dict(),
            },
            indent=2,
        )
    )
    md_path.write_text(_render_markdown(specs, result))
    print(f"[ensemble_run_specs] wrote {json_path} + {md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
