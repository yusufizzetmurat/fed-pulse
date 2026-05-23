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
    # Per-row train / val / test RMSE + the test/train gap derivative.
    # ``train_rmse_values`` is the final-state training-set RMSE;
    # ``val_rmse_values`` is the best early-stopping checkpoint's val
    # RMSE; ``test_rmse_values`` is the held-out test RMSE (the
    # headline number the markdown emits as ``test-RMSE``). On the
    # legacy 80/20 path no real held-out exists, so test_rmse_values
    # collapses to val_rmse_values and the gap stays comparable to
    # the pre-PR holdout_train_gap. Defaults stay zero-valued so
    # callers that predate the new columns build a row without
    # naming them.
    train_rmse_values: list[float] = field(default_factory=list)
    val_rmse_values: list[float] = field(default_factory=list)
    test_rmse_values: list[float] = field(default_factory=list)
    train_rmse_ci: BootstrapCI | None = None
    val_rmse_ci: BootstrapCI | None = None
    test_rmse_ci: BootstrapCI | None = None
    holdout_train_gap: float = 0.0
    test_train_gap: float = 0.0
    gap_flag: str = "ok"
    target_mode: str = "real"
    fold_id: str | None = None
    protocol: str = "single-fold"


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


def _trial_val_metric(trial: dict[str, Any], key: str) -> float | None:
    summary = trial.get("summary") or {}
    val_metrics = summary.get("val_metrics") or {}
    value = val_metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trial_test_metric(trial: dict[str, Any], key: str) -> float | None:
    """Return the held-out test RMSE.

    Falls back to the trial's headline ``metrics`` block on the legacy
    80/20 path where ``test_metrics`` is absent, so the aggregator's
    test-RMSE column has a value in both protocols.
    """

    summary = trial.get("summary") or {}
    test_metrics = summary.get("test_metrics") or summary.get("metrics") or {}
    value = test_metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trial_fold_id(trial: dict[str, Any]) -> str | None:
    if trial.get("fold_id"):
        return str(trial["fold_id"])
    summary = trial.get("summary") or {}
    fold_id = summary.get("fold_id")
    if fold_id:
        return str(fold_id)
    return None


def _trial_protocol(trial: dict[str, Any]) -> str:
    summary = trial.get("summary") or {}
    protocol = summary.get("protocol")
    if isinstance(protocol, str) and protocol:
        return protocol
    return "single-fold"


def _trial_target_mode(trial: dict[str, Any]) -> str:
    summary = trial.get("summary") or {}
    mode = summary.get("target_mode")
    if isinstance(mode, str) and mode:
        return mode
    return "real"


def _bucket_key(
    architecture: str, target_mode: str, fold_id: str | None
) -> str:
    """Compose the dict key the aggregator groups by.

    Shuffled-target trials sit in a separate bucket so the
    memorisation-control row does not contaminate the headline table.
    Walk-forward fold rows carry an explicit ``fold_id`` suffix so the
    aggregator emits one row per (architecture, fold) pair plus an
    all-folds aggregate row per architecture.
    """

    base = architecture if target_mode == "real" else f"{architecture}::{target_mode}"
    if fold_id:
        return f"{base}::{fold_id}"
    return base


def _collect_per_architecture(reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:  # noqa: C901
    by_arch: dict[str, dict[str, Any]] = {}
    for report in reports:
        for trial in report.get("trials") or []:
            architecture = _trial_architecture(trial)
            target_mode = _trial_target_mode(trial)
            fold_id = _trial_fold_id(trial)
            protocol = _trial_protocol(trial)
            key = _bucket_key(architecture, target_mode, fold_id)
            bucket = by_arch.setdefault(
                key,
                {
                    "architecture": architecture,
                    "target_mode": target_mode,
                    "fold_id": fold_id,
                    "protocol": protocol,
                    "seeds": [],
                    "combined_rmse": [],
                    "close_rmse": [],
                    "volatility_rmse": [],
                    "train_combined_rmse": [],
                    "val_combined_rmse": [],
                    "test_combined_rmse": [],
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
            val_combined = _trial_val_metric(trial, "combined_rmse")
            if val_combined is not None:
                bucket["val_combined_rmse"].append(val_combined)
            test_combined = _trial_test_metric(trial, "combined_rmse")
            if test_combined is not None:
                bucket["test_combined_rmse"].append(test_combined)
            # Once any credibility-on trial lands for an architecture the
            # bucket flips on so the headline label stays honest.
            bucket["credibility_features"] = bucket["credibility_features"] or _trial_credibility(trial)
    # Emit an all-folds aggregate row per (architecture, target_mode)
    # when any per-fold bucket is present. The all-folds row collects
    # every per-fold trial so the bootstrap CI is computed across
    # (seed, fold) cells.
    aggregate_keys: dict[str, dict[str, Any]] = {}
    for _key, bucket in list(by_arch.items()):
        fold_id = bucket.get("fold_id")
        if not fold_id:
            continue
        agg_key = _bucket_key(bucket["architecture"], bucket["target_mode"], None) + "::all-folds"
        agg_bucket = aggregate_keys.setdefault(
            agg_key,
            {
                "architecture": bucket["architecture"],
                "target_mode": bucket["target_mode"],
                "fold_id": "all-folds",
                "protocol": bucket["protocol"],
                "seeds": [],
                "combined_rmse": [],
                "close_rmse": [],
                "volatility_rmse": [],
                "train_combined_rmse": [],
                "val_combined_rmse": [],
                "test_combined_rmse": [],
                "credibility_features": bucket["credibility_features"],
            },
        )
        for field_name in (
            "seeds",
            "combined_rmse",
            "close_rmse",
            "volatility_rmse",
            "train_combined_rmse",
            "val_combined_rmse",
            "test_combined_rmse",
        ):
            agg_bucket[field_name].extend(bucket[field_name])
        agg_bucket["credibility_features"] = (
            agg_bucket["credibility_features"] or bucket["credibility_features"]
        )
    by_arch.update(aggregate_keys)
    return by_arch


def _build_rows(  # noqa: C901
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
        fold_id = payload.get("fold_id")
        protocol = str(payload.get("protocol", "single-fold"))
        train_values = list(payload.get("train_combined_rmse") or [])
        val_values_field = list(payload.get("val_combined_rmse") or [])
        test_values_field = list(payload.get("test_combined_rmse") or [])
        headline_values = list(payload["combined_rmse"])
        # Walk-forward path emits explicit test_metrics; fall back to
        # the trial's headline ``combined_rmse`` when test_metrics is
        # absent (legacy 80/20 reports). The aggregator's ``test-RMSE``
        # column always reflects the held-out RMSE on the walk-forward
        # path and the val RMSE on the legacy path.
        if not test_values_field:
            test_values_field = list(headline_values)
        if not val_values_field:
            val_values_field = list(headline_values)
        train_mean = (sum(train_values) / len(train_values)) if train_values else 0.0
        val_mean = sum(val_values_field) / len(val_values_field) if val_values_field else 0.0
        test_mean = sum(test_values_field) / len(test_values_field) if test_values_field else 0.0
        if train_mean > 0.0:
            holdout_gap = (val_mean - train_mean) / train_mean
            test_gap = (test_mean - train_mean) / train_mean
        else:
            holdout_gap = 0.0
            test_gap = 0.0
        gap_flag = "high" if test_gap > 0.5 else "ok"
        rows.append(
            ArchitectureRow(
                architecture=architecture,
                seeds=sorted(set(payload["seeds"])),
                credibility_features=bool(payload["credibility_features"]),
                combined_rmse_values=headline_values,
                close_rmse_values=list(payload["close_rmse"]),
                volatility_rmse_values=list(payload["volatility_rmse"]),
                combined_rmse_ci=block_bootstrap_ci(
                    headline_values,
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
                val_rmse_values=val_values_field,
                test_rmse_values=test_values_field,
                train_rmse_ci=block_bootstrap_ci(
                    train_values or [0.0],
                    statistic="mean",
                    block_size=block_size,
                    n_resamples=n_resamples,
                    coverage=coverage,
                    seed=seed,
                ),
                val_rmse_ci=block_bootstrap_ci(
                    val_values_field,
                    statistic="mean",
                    block_size=block_size,
                    n_resamples=n_resamples,
                    coverage=coverage,
                    seed=seed,
                ),
                test_rmse_ci=block_bootstrap_ci(
                    test_values_field,
                    statistic="mean",
                    block_size=block_size,
                    n_resamples=n_resamples,
                    coverage=coverage,
                    seed=seed,
                ),
                holdout_train_gap=float(holdout_gap),
                test_train_gap=float(test_gap),
                gap_flag=gap_flag,
                target_mode=target_mode,
                fold_id=fold_id if isinstance(fold_id, str) else None,
                protocol=protocol,
            )
        )

    # Deterministic ordering. Real-target rows sort first so the
    # headline table is the production-relevant ranking; shuffled rows
    # trail underneath. Architectures rank by their representative
    # test-RMSE (the all-folds-aggregate row when folds are present,
    # otherwise the single row's test-RMSE). Within an architecture
    # group folds sort as: single-fold None first (legacy path), then
    # ``wf_fold_N`` in lexicographic order, with the all-folds
    # aggregate row last so the eye flows fold-by-fold then summary.
    def _fold_sort_key(fold_id: str | None) -> tuple[int, str]:
        if fold_id is None:
            return (0, "")
        if fold_id == "all-folds":
            return (2, "")
        return (1, fold_id)

    # Precompute the per-architecture representative test-RMSE so the
    # sort key never reads off the rows list during the sort itself.
    # ``rows`` is the same list that ``list.sort`` reorders in place;
    # reading it during key extraction observes a partially-mutated
    # view on some Python builds.
    def _row_test_point(r: ArchitectureRow) -> float:
        return r.test_rmse_ci.point if r.test_rmse_ci is not None else r.combined_rmse_ci.point

    arch_rep: dict[tuple[str, str], float] = {}
    arch_has_all_folds: dict[tuple[str, str], bool] = {}
    for r in rows:
        key = (r.architecture, r.target_mode)
        point = _row_test_point(r)
        if r.fold_id == "all-folds":
            arch_rep[key] = point
            arch_has_all_folds[key] = True
        elif not arch_has_all_folds.get(key, False):
            # Track the minimum per-fold / single-row value until an
            # all-folds row claims the slot.
            if key in arch_rep:
                arch_rep[key] = min(arch_rep[key], point)
            else:
                arch_rep[key] = point

    rows.sort(
        key=lambda r: (
            0 if r.target_mode == "real" else 1,
            arch_rep.get((r.architecture, r.target_mode), float("inf")),
            r.architecture,
            _fold_sort_key(r.fold_id),
        )
    )
    return rows


_HEADER = (
    "| Rank | Architecture | Protocol | Fold | Credibility | n | mode | "
    "train-RMSE | val-RMSE | test-RMSE (mean, {coverage_pct}% CI) | "
    "test/train gap | close-RMSE | volatility-RMSE |"
)
_SEPARATOR = "|---:|---|:-:|:-:|:-:|---:|:-:|---|---|---|---:|---|---|"


def render_markdown(rows: list[ArchitectureRow], *, coverage: float) -> str:
    if not rows:
        return "_no forecaster sweep rows found_\n"
    coverage_pct = int(round(coverage * 100))

    real_rows = [row for row in rows if row.target_mode == "real"]
    shuffled_rows = [row for row in rows if row.target_mode != "real"]

    lines: list[str] = []
    if real_rows:
        lines.append(_HEADER.format(coverage_pct=coverage_pct))
        lines.append(_SEPARATOR)
        for rank, row in enumerate(real_rows, start=1):
            lines.append(_render_row_line(row, rank, coverage_pct))

    if shuffled_rows:
        lines.append("")
        lines.append("### Shuffled-targets control")
        lines.append("")
        lines.append(_HEADER.format(coverage_pct=coverage_pct))
        lines.append(_SEPARATOR)
        for rank, row in enumerate(shuffled_rows, start=1):
            lines.append(_render_row_line(row, rank, coverage_pct))

    return "\n".join(lines) + "\n"


def _render_row_line(row: ArchitectureRow, rank: int, coverage_pct: int) -> str:
    n = len(row.combined_rmse_values)
    c = row.close_rmse_ci
    v = row.volatility_rmse_ci
    train_ci = row.train_rmse_ci if row.train_rmse_ci is not None else row.combined_rmse_ci
    val_ci = row.val_rmse_ci if row.val_rmse_ci is not None else row.combined_rmse_ci
    test_ci = row.test_rmse_ci if row.test_rmse_ci is not None else row.combined_rmse_ci
    gap_text = f"{row.test_train_gap:+.3f}"
    if row.gap_flag == "high":
        gap_text = f"{gap_text}!"
    fold_label = row.fold_id if row.fold_id else "-"
    return (
        f"| {rank} | `{row.architecture}` | {row.protocol} | {fold_label} | "
        f"{'on' if row.credibility_features else 'off'} | {n} | "
        f"{row.target_mode} | "
        f"{train_ci.point:.4f} [{train_ci.lo:.4f}, {train_ci.hi:.4f}] | "
        f"{val_ci.point:.4f} [{val_ci.lo:.4f}, {val_ci.hi:.4f}] | "
        f"{test_ci.point:.4f} [{test_ci.lo:.4f}, {test_ci.hi:.4f}] | "
        f"{gap_text} | "
        f"{c.point:.4f} [{c.lo:.4f}, {c.hi:.4f}] | "
        f"{v.point:.4f} [{v.lo:.4f}, {v.hi:.4f}] |"
    )


def _row_to_json(row: ArchitectureRow) -> dict[str, Any]:
    return {
        "architecture": row.architecture,
        "target_mode": row.target_mode,
        "fold_id": row.fold_id,
        "protocol": row.protocol,
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
        "test_rmse": {
            "values": row.test_rmse_values,
            "ci": asdict(row.test_rmse_ci) if row.test_rmse_ci is not None else None,
        },
        "holdout_train_gap": row.holdout_train_gap,
        "test_train_gap": row.test_train_gap,
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
