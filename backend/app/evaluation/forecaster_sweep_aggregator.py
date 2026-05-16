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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

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
    # Mitigation 4: per-row train + val RMSE and the gap derivative.
    # ``train_rmse_values`` and ``val_rmse_values`` collect every trial
    # under the architecture (one per seed x HP combo); the aggregator
    # bootstraps the mean and emits a ``gap_flag`` when the mean gap
    # crosses 0.5. The headline "val" label maps to the holdout-side
    # RMSE the loop reports as ``combined_rmse``; "train" is the
    # final-state training-set RMSE captured by
    # ``TrainingRunSummary.train_metrics``. Defaults stay zero-valued
    # so older callers / unit fixtures that predate the train-metrics
    # column build a row without naming them.
    train_rmse_values: list[float] = field(default_factory=list)
    val_rmse_values: list[float] = field(default_factory=list)
    train_rmse_ci: BootstrapCI | None = None
    val_rmse_ci: BootstrapCI | None = None
    holdout_train_gap: float = 0.0
    gap_flag: str = "ok"
    target_mode: str = "real"


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"sweep report must be a JSON object: {path}")
    return payload


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


def _trial_seed(trial: dict[str, Any]) -> int | None:
    seed = trial.get("seed")
    if seed is None:
        return None
    try:
        return int(seed)
    except (TypeError, ValueError):
        return None


def _trial_architecture(trial: dict[str, Any]) -> str:
    architecture = trial.get("architecture")
    if architecture:
        return str(architecture)
    summary = trial.get("summary") or {}
    model_config = summary.get("model_config") or {}
    return str(model_config.get("architecture", "lstm"))


def _trial_metric(trial: dict[str, Any], key: str) -> float | None:
    summary = trial.get("summary") or {}
    metrics = summary.get("metrics") or {}
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trial_credibility(trial: dict[str, Any]) -> bool:
    summary = trial.get("summary") or {}
    model_config = summary.get("model_config") or {}
    return bool(model_config.get("credibility_features", False))


def _trial_train_metric(trial: dict[str, Any], key: str) -> float | None:
    summary = trial.get("summary") or {}
    train_metrics = summary.get("train_metrics") or {}
    value = train_metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trial_target_mode(trial: dict[str, Any]) -> str:
    summary = trial.get("summary") or {}
    mode = summary.get("target_mode")
    if isinstance(mode, str) and mode:
        return mode
    return "real"


def _bucket_key(architecture: str, target_mode: str) -> str:
    """Compose the dict key the aggregator groups by.

    Shuffled-target trials sit in a separate bucket so the
    memorisation-control row does not contaminate the headline table.
    The key is ``"<architecture>"`` for ``target_mode == "real"`` and
    ``"<architecture>::shuffled"`` otherwise.
    """

    if target_mode == "real":
        return architecture
    return f"{architecture}::{target_mode}"


def _collect_per_architecture(reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_arch: dict[str, dict[str, Any]] = {}
    for report in reports:
        for trial in report.get("trials") or []:
            architecture = _trial_architecture(trial)
            target_mode = _trial_target_mode(trial)
            key = _bucket_key(architecture, target_mode)
            bucket = by_arch.setdefault(
                key,
                {
                    "architecture": architecture,
                    "target_mode": target_mode,
                    "seeds": [],
                    "combined_rmse": [],
                    "close_rmse": [],
                    "volatility_rmse": [],
                    "train_combined_rmse": [],
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
            train_combined = _trial_train_metric(trial, "combined_rmse")
            if train_combined is not None:
                bucket["train_combined_rmse"].append(train_combined)
            # Once any credibility-on trial lands for an architecture the
            # bucket flips on so the headline label stays honest.
            bucket["credibility_features"] = bucket["credibility_features"] or _trial_credibility(trial)
    return by_arch


def _build_rows(
    by_arch: dict[str, dict[str, Any]],
    *,
    block_size: int,
    n_resamples: int,
    coverage: float,
    seed: int,
) -> list[ArchitectureRow]:
    rows: list[ArchitectureRow] = []
    for _bucket_key_str, payload in by_arch.items():
        if not payload["combined_rmse"]:
            continue
        architecture = str(payload.get("architecture", _bucket_key_str))
        target_mode = str(payload.get("target_mode", "real"))
        train_values = list(payload.get("train_combined_rmse") or [])
        val_values = list(payload["combined_rmse"])
        train_mean = (sum(train_values) / len(train_values)) if train_values else 0.0
        val_mean = sum(val_values) / len(val_values) if val_values else 0.0
        if train_mean > 0.0:
            gap = (val_mean - train_mean) / train_mean
        else:
            gap = 0.0
        gap_flag = "high" if gap > 0.5 else "ok"
        rows.append(
            ArchitectureRow(
                architecture=architecture,
                seeds=sorted(set(payload["seeds"])),
                credibility_features=bool(payload["credibility_features"]),
                combined_rmse_values=val_values,
                close_rmse_values=list(payload["close_rmse"]),
                volatility_rmse_values=list(payload["volatility_rmse"]),
                combined_rmse_ci=block_bootstrap_ci(
                    val_values,
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
                train_rmse_values=train_values,
                val_rmse_values=val_values,
                train_rmse_ci=block_bootstrap_ci(
                    train_values or [0.0],
                    statistic="mean",
                    block_size=block_size,
                    n_resamples=n_resamples,
                    coverage=coverage,
                    seed=seed,
                ),
                val_rmse_ci=block_bootstrap_ci(
                    val_values,
                    statistic="mean",
                    block_size=block_size,
                    n_resamples=n_resamples,
                    coverage=coverage,
                    seed=seed,
                ),
                holdout_train_gap=float(gap),
                gap_flag=gap_flag,
                target_mode=target_mode,
            )
        )
    # Lower combined-RMSE is better, so ascending sort gives rank order.
    # Real-target rows sort first so the headline table is the
    # production-relevant ranking; shuffled rows trail underneath in
    # the same ascending order.
    rows.sort(key=lambda r: (0 if r.target_mode == "real" else 1, r.combined_rmse_ci.point))
    return rows


def render_markdown(rows: list[ArchitectureRow], *, coverage: float) -> str:
    if not rows:
        return "_no forecaster sweep rows found_\n"
    coverage_pct = int(round(coverage * 100))

    real_rows = [row for row in rows if row.target_mode == "real"]
    shuffled_rows = [row for row in rows if row.target_mode != "real"]

    lines: list[str] = []
    if real_rows:
        lines.append(
            f"| Rank | Architecture | Credibility | n | mode | train-RMSE | holdout-RMSE | holdout/train gap | combined-RMSE (mean, {coverage_pct}% CI) | close-RMSE | volatility-RMSE |"
        )
        lines.append("|---:|---|:-:|---:|:-:|---|---|---:|---|---|---|")
        for rank, row in enumerate(real_rows, start=1):
            lines.append(_render_row_line(row, rank, coverage_pct))

    if shuffled_rows:
        lines.append("")
        lines.append("### Shuffled-targets control")
        lines.append("")
        lines.append(
            f"| Rank | Architecture | Credibility | n | mode | train-RMSE | holdout-RMSE | holdout/train gap | combined-RMSE (mean, {coverage_pct}% CI) | close-RMSE | volatility-RMSE |"
        )
        lines.append("|---:|---|:-:|---:|:-:|---|---|---:|---|---|---|")
        for rank, row in enumerate(shuffled_rows, start=1):
            lines.append(_render_row_line(row, rank, coverage_pct))

    return "\n".join(lines) + "\n"


def _render_row_line(row: ArchitectureRow, rank: int, coverage_pct: int) -> str:
    n = len(row.combined_rmse_values)
    cr = row.combined_rmse_ci
    c = row.close_rmse_ci
    v = row.volatility_rmse_ci
    train_ci = row.train_rmse_ci
    val_ci = row.val_rmse_ci
    gap_text = f"{row.holdout_train_gap:+.3f}"
    if row.gap_flag == "high":
        gap_text = f"{gap_text}!"
    return (
        f"| {rank} | `{row.architecture}` | "
        f"{'on' if row.credibility_features else 'off'} | {n} | "
        f"{row.target_mode} | "
        f"{train_ci.point:.4f} [{train_ci.lo:.4f}, {train_ci.hi:.4f}] | "
        f"{val_ci.point:.4f} [{val_ci.lo:.4f}, {val_ci.hi:.4f}] | "
        f"{gap_text} | "
        f"{cr.point:.4f} [{cr.lo:.4f}, {cr.hi:.4f}] | "
        f"{c.point:.4f} [{c.lo:.4f}, {c.hi:.4f}] | "
        f"{v.point:.4f} [{v.lo:.4f}, {v.hi:.4f}] |"
    )


def _row_to_json(row: ArchitectureRow) -> dict[str, Any]:
    return {
        "architecture": row.architecture,
        "target_mode": row.target_mode,
        "seeds": row.seeds,
        "credibility_features": row.credibility_features,
        "combined_rmse": {
            "values": row.combined_rmse_values,
            "ci": asdict(row.combined_rmse_ci) if row.combined_rmse_ci is not None else None,
        },
        "close_rmse": {
            "values": row.close_rmse_values,
            "ci": asdict(row.close_rmse_ci) if row.close_rmse_ci is not None else None,
        },
        "volatility_rmse": {
            "values": row.volatility_rmse_values,
            "ci": asdict(row.volatility_rmse_ci) if row.volatility_rmse_ci is not None else None,
        },
        "train_rmse": {
            "values": row.train_rmse_values,
            "ci": asdict(row.train_rmse_ci) if row.train_rmse_ci is not None else None,
        },
        "val_rmse": {
            "values": row.val_rmse_values,
            "ci": asdict(row.val_rmse_ci) if row.val_rmse_ci is not None else None,
        },
        "holdout_train_gap": row.holdout_train_gap,
        "gap_flag": row.gap_flag,
    }


def aggregate(
    artifact_dir: Path,
    *,
    block_size: int = 1,
    n_resamples: int = 1000,
    coverage: float = 0.95,
    seed: int = 11,
) -> tuple[list[ArchitectureRow], str, dict[str, Any]]:
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
