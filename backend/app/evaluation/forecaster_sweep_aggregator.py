"""Aggregate forecaster sweep trials into a per-architecture headline table.

Consumes one or more ``forecaster_sweep_results.json`` payloads produced by
``backend/app/train_forecaster.py`` (see :func:`_run_sweep`), groups trials by
architecture, and emits

- a JSON summary with block-bootstrap CIs per architecture
- a markdown headline table sortable by combined-RMSE (lower is better)

The bootstrap CIs are computed via :func:`app.evaluation.bootstrap.block_bootstrap_ci`
to keep the protocol identical to the NLP bake-off aggregator.

Usage::

    python -m app.evaluation.forecaster_sweep_aggregator \\
        --artifact-dir backend/models/forecaster_sweep_results.json \\
        --output-dir   data/artifacts/forecaster_sweep
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from app.evaluation.bootstrap import BootstrapCI, block_bootstrap_ci


@dataclass(frozen=True)
class ArchitectureRow:
    architecture: str
    seeds: list[int]
    credibility_features: bool
    combined_rmse_values: list[float]
    close_rmse_values: list[float]
    volatility_rmse_values: list[float]
    combined_rmse_ci: BootstrapCI
    close_rmse_ci: BootstrapCI
    volatility_rmse_ci: BootstrapCI


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_report_files(artifact_dir: Path) -> Iterable[Path]:
    """Yield every sweep-result JSON under ``artifact_dir``.

    Accepts either a single file (matching ``*sweep_results.json``) or a
    directory. When given a directory the iteration is recursive and sorted
    so output is deterministic.
    """

    if artifact_dir.is_file():
        yield artifact_dir
        return
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"artifact_dir does not exist: {artifact_dir}")
    yield from sorted(artifact_dir.glob("**/*sweep_results.json"))


def _trial_seed(trial: dict) -> int | None:
    seed = trial.get("seed")
    if seed is None:
        return None
    try:
        return int(seed)
    except (TypeError, ValueError):
        return None


def _trial_architecture(trial: dict) -> str:
    architecture = trial.get("architecture")
    if architecture:
        return str(architecture)
    summary = trial.get("summary") or {}
    model_config = summary.get("model_config") or {}
    return str(model_config.get("architecture", "lstm"))


def _trial_metric(trial: dict, key: str) -> float | None:
    summary = trial.get("summary") or {}
    metrics = summary.get("metrics") or {}
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trial_credibility(trial: dict) -> bool:
    summary = trial.get("summary") or {}
    model_config = summary.get("model_config") or {}
    return bool(model_config.get("credibility_features", False))


def _collect_per_architecture(reports: list[dict]) -> dict[str, dict]:
    by_arch: dict[str, dict] = {}
    for report in reports:
        for trial in report.get("trials") or []:
            architecture = _trial_architecture(trial)
            bucket = by_arch.setdefault(
                architecture,
                {
                    "seeds": [],
                    "combined_rmse": [],
                    "close_rmse": [],
                    "volatility_rmse": [],
                    "credibility_features": _trial_credibility(trial),
                },
            )
            combined = _trial_metric(trial, "combined_rmse")
            if combined is None:
                continue
            seed = _trial_seed(trial)
            if seed is not None:
                bucket["seeds"].append(seed)
            bucket["combined_rmse"].append(combined)
            close = _trial_metric(trial, "close_rmse")
            if close is not None:
                bucket["close_rmse"].append(close)
            vol = _trial_metric(trial, "volatility_rmse")
            if vol is not None:
                bucket["volatility_rmse"].append(vol)
            # Once any credibility-on trial lands for an architecture the
            # bucket flips on so the headline label stays honest.
            bucket["credibility_features"] = bucket["credibility_features"] or _trial_credibility(trial)
    return by_arch


def _build_rows(
    by_arch: dict[str, dict],
    *,
    block_size: int,
    n_resamples: int,
    coverage: float,
    seed: int,
) -> list[ArchitectureRow]:
    rows: list[ArchitectureRow] = []
    for architecture, payload in by_arch.items():
        if not payload["combined_rmse"]:
            continue
        rows.append(
            ArchitectureRow(
                architecture=architecture,
                seeds=sorted(set(payload["seeds"])),
                credibility_features=bool(payload["credibility_features"]),
                combined_rmse_values=list(payload["combined_rmse"]),
                close_rmse_values=list(payload["close_rmse"]),
                volatility_rmse_values=list(payload["volatility_rmse"]),
                combined_rmse_ci=block_bootstrap_ci(
                    payload["combined_rmse"],
                    statistic="mean",
                    block_size=block_size,
                    n_resamples=n_resamples,
                    coverage=coverage,
                    seed=seed,
                ),
                close_rmse_ci=block_bootstrap_ci(
                    payload["close_rmse"] or [0.0],
                    statistic="mean",
                    block_size=block_size,
                    n_resamples=n_resamples,
                    coverage=coverage,
                    seed=seed,
                ),
                volatility_rmse_ci=block_bootstrap_ci(
                    payload["volatility_rmse"] or [0.0],
                    statistic="mean",
                    block_size=block_size,
                    n_resamples=n_resamples,
                    coverage=coverage,
                    seed=seed,
                ),
            )
        )
    # Lower combined-RMSE is better, so ascending sort gives rank order.
    rows.sort(key=lambda r: r.combined_rmse_ci.point)
    return rows


def render_markdown(rows: list[ArchitectureRow], *, coverage: float) -> str:
    if not rows:
        return "_no forecaster sweep rows found_\n"
    coverage_pct = int(round(coverage * 100))
    lines = [
        f"| Rank | Architecture | Credibility | n | combined-RMSE (mean, {coverage_pct}% CI) | close-RMSE | volatility-RMSE |",
        "|---:|---|:-:|---:|---|---|---|",
    ]
    for rank, row in enumerate(rows, start=1):
        n = len(row.combined_rmse_values)
        cr = row.combined_rmse_ci
        c = row.close_rmse_ci
        v = row.volatility_rmse_ci
        lines.append(
            f"| {rank} | `{row.architecture}` | {'on' if row.credibility_features else 'off'} | {n} | "
            f"{cr.point:.4f} [{cr.lo:.4f}, {cr.hi:.4f}] | "
            f"{c.point:.4f} [{c.lo:.4f}, {c.hi:.4f}] | "
            f"{v.point:.4f} [{v.lo:.4f}, {v.hi:.4f}] |"
        )
    return "\n".join(lines) + "\n"


def _row_to_json(row: ArchitectureRow) -> dict:
    return {
        "architecture": row.architecture,
        "seeds": row.seeds,
        "credibility_features": row.credibility_features,
        "combined_rmse": {
            "values": row.combined_rmse_values,
            "ci": asdict(row.combined_rmse_ci),
        },
        "close_rmse": {
            "values": row.close_rmse_values,
            "ci": asdict(row.close_rmse_ci),
        },
        "volatility_rmse": {
            "values": row.volatility_rmse_values,
            "ci": asdict(row.volatility_rmse_ci),
        },
    }


def aggregate(
    artifact_dir: Path,
    *,
    block_size: int = 1,
    n_resamples: int = 1000,
    coverage: float = 0.95,
    seed: int = 11,
) -> tuple[list[ArchitectureRow], str, dict]:
    """Read sweep reports under ``artifact_dir`` and emit (rows, md, json_payload)."""

    report_paths = list(_iter_report_files(artifact_dir))
    if not report_paths:
        raise FileNotFoundError(f"no sweep report files found under {artifact_dir}")
    reports = [_load_report(p) for p in report_paths]
    by_arch = _collect_per_architecture(reports)
    rows = _build_rows(
        by_arch,
        block_size=block_size,
        n_resamples=n_resamples,
        coverage=coverage,
        seed=seed,
    )
    markdown = render_markdown(rows, coverage=coverage)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifact_dir": str(artifact_dir),
        "source_reports": [str(p) for p in report_paths],
        "block_size": block_size,
        "n_resamples": n_resamples,
        "coverage": coverage,
        "bootstrap_seed": seed,
        "architectures": [_row_to_json(row) for row in rows],
    }
    return rows, markdown, payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate forecaster sweep trials into a per-architecture headline table."
    )
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rows, markdown, payload = aggregate(
        args.artifact_dir,
        block_size=args.block_size,
        n_resamples=args.n_resamples,
        coverage=args.coverage,
        seed=args.seed,
    )

    output_dir = args.output_dir or (
        args.artifact_dir.parent if args.artifact_dir.is_file() else args.artifact_dir
    ) / "forecaster_sweep_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"forecaster_sweep_summary_{timestamp}.json"
    md_path = output_dir / f"forecaster_sweep_summary_{timestamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    print(f"[forecaster_sweep_aggregator] {len(rows)} architectures -> {json_path}")
    print(f"[forecaster_sweep_aggregator] markdown -> {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
