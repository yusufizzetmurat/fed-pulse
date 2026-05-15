"""Aggregate per-bank cross-CB evaluations into a transfer matrix.

Walks every ``(model_alias, checkpoint, bank)`` cell, calls
:func:`app.evaluation.cross_bank_transfer.evaluate_cross_bank`, and produces:

- ``transfer_matrix.json`` — full nested payload (per checkpoint, per bank).
- ``transfer_matrix.csv`` — flat matrix: rows = (model, bank), with point
  estimates plus 95% block-bootstrap CIs *when multiple checkpoints exist
  for the same alias* (one per training seed). Single-checkpoint cells emit
  only point estimates and a ``ci_kind="point_estimate"`` marker.
- ``transfer_matrix.md`` — same as csv but markdown.

CLI syntax: ``--model-checkpoints alias=path[,alias=path…]``. Repeating the
same alias accumulates paths for that alias, so the natural invocation is

    --model-checkpoints "finbert=/p/s11,finbert=/p/s29,finbert=/p/s47,bge_large=/p/s11"

which produces a 3-checkpoint CI for ``finbert`` and a point estimate for
``bge_large``.

Earlier revisions ran a seed loop that re-evaluated the same checkpoint per
seed, producing artificially-tight CIs. That was a fabricated-statistic
liability and is removed; CI bands now come from genuine cross-checkpoint
variance only.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import DATA_DIR
from app.evaluation.bootstrap import block_bootstrap_ci
from app.evaluation.cross_bank_transfer import (
    CROSS_BANK_SOURCES,
    CrossBankResult,
    evaluate_cross_bank,
)

BANK_ALIASES: dict[str, str] = {
    "ecb": "gtfintechlab_european_central_bank",
    "boj": "gtfintechlab_bank_of_japan",
    "boe": "gtfintechlab_bank_of_england",
    "boc": "gtfintechlab_bank_of_canada",
    "rba": "gtfintechlab_reserve_bank_of_australia",
}


def _ensure_package_dir(package_dir: Path) -> Path:
    if not package_dir.exists():
        raise FileNotFoundError(f"Training package not found: {package_dir}")
    return package_dir


def _parse_model_checkpoints(spec: str) -> dict[str, list[str]]:
    """Parse ``alias=path[,alias=path…]`` into ``{alias: [path, …]}``.

    Repeating the same alias appends another path under it; the matrix
    builder then runs the eval per path and aggregates with a real CI.
    """

    out: dict[str, list[str]] = {}
    if not spec:
        return out
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"--model-checkpoints entry {piece!r} missing 'alias=path'")
        alias, path = piece.split("=", 1)
        alias = alias.strip()
        path = path.strip()
        if not alias or not path:
            raise ValueError(f"--model-checkpoints entry {piece!r} has empty alias or path")
        out.setdefault(alias, []).append(path)
    return out


def _split_banks(spec: str) -> list[str]:
    if not spec:
        return list(CROSS_BANK_SOURCES)
    out: list[str] = []
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in BANK_ALIASES:
            out.append(BANK_ALIASES[token])
        elif token in CROSS_BANK_SOURCES:
            out.append(token)
        else:
            raise ValueError(f"unknown bank token: {token!r}")
    return out


def _point_only(values: list[float]) -> dict[str, Any]:
    """Single-checkpoint summary — no CI to claim."""

    if not values:
        return {"point": float("nan"), "ci_kind": "missing", "n_checkpoints": 0}
    return {
        "point": float(values[0]),
        "ci_kind": "point_estimate",
        "n_checkpoints": 1,
    }


def _ci_summary(
    values: list[float],
    *,
    n_resamples: int,
    coverage: float,
    seed: int,
) -> dict[str, Any]:
    """Block-bootstrap CI across ≥2 distinct checkpoints."""

    ci = asdict(
        block_bootstrap_ci(
            values,
            statistic="mean",
            block_size=1,
            n_resamples=n_resamples,
            coverage=coverage,
            seed=seed,
        )
    )
    ci.update({"ci_kind": "block_bootstrap", "n_checkpoints": len(values)})
    return ci


def _aggregate_checkpoint_runs(
    results: list[CrossBankResult],
    *,
    n_resamples: int,
    coverage: float,
    seed: int,
) -> dict[str, Any]:
    if not results:
        return {}
    macro = [r.macro_f1 for r in results]
    weighted = [r.weighted_f1 for r in results]
    acc = [r.accuracy for r in results]
    summarise = _ci_summary if len(results) >= 2 else _point_only
    kwargs = {"n_resamples": n_resamples, "coverage": coverage, "seed": seed}
    return {
        "support": results[0].support,
        "macro_f1": summarise(macro, **kwargs) if len(results) >= 2 else _point_only(macro),
        "weighted_f1": summarise(weighted, **kwargs) if len(results) >= 2 else _point_only(weighted),
        "accuracy": summarise(acc, **kwargs) if len(results) >= 2 else _point_only(acc),
    }


def build_matrix(
    *,
    package_dir: Path,
    model_checkpoints: dict[str, list[str]],
    banks: list[str],
    n_resamples: int = 1000,
    coverage: float = 0.95,
    rng_seed: int = 11,
    predict_fn=None,
) -> dict[str, Any]:
    """Build the transfer-matrix payload.

    For each ``alias`` in ``model_checkpoints`` and each ``bank``, run the
    eval once per provided checkpoint and aggregate the resulting macro-F1 /
    accuracy / weighted-F1 across checkpoints. CI is computed only when ≥ 2
    distinct checkpoints feed the cell.
    """

    matrix: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "training_package_id": package_dir.name,
        "banks": banks,
        "models": list(model_checkpoints.keys()),
        "by_model": {},
    }

    for model_alias, checkpoints in model_checkpoints.items():
        per_model: dict[str, Any] = {"checkpoints": checkpoints, "per_bank": {}}
        for bank in banks:
            per_checkpoint: list[CrossBankResult] = []
            for checkpoint in checkpoints:
                try:
                    result = evaluate_cross_bank(
                        package_dir=package_dir,
                        bank_source=bank,
                        checkpoint=checkpoint,
                        predict_fn=predict_fn,
                    )
                except Exception as exc:  # noqa: BLE001 — surface per-cell failure
                    per_model.setdefault("failures", []).append(
                        {"bank": bank, "checkpoint": checkpoint, "error": str(exc)}
                    )
                    continue
                per_checkpoint.append(result)
            if per_checkpoint:
                per_model["per_bank"][bank] = {
                    "summary": _aggregate_checkpoint_runs(
                        per_checkpoint,
                        n_resamples=n_resamples,
                        coverage=coverage,
                        seed=rng_seed,
                    ),
                    "per_checkpoint": [r.to_dict() for r in per_checkpoint],
                }
        matrix["by_model"][model_alias] = per_model
    return matrix


def _scrub_nan(value: Any) -> Any:
    """Recursively convert non-JSON-spec floats (nan / +-inf) to None."""

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _scrub_nan(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_nan(v) for v in value]
    return value


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value}"
    return str(value)


def render_csv(matrix: dict[str, Any]) -> str:
    fieldnames = [
        "model", "bank", "support", "n_checkpoints", "ci_kind",
        "macro_f1_point", "macro_f1_lo", "macro_f1_hi",
        "weighted_f1_point", "weighted_f1_lo", "weighted_f1_hi",
        "accuracy_point", "accuracy_lo", "accuracy_hi",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for model_alias, payload in matrix.get("by_model", {}).items():
        for bank, cell in payload.get("per_bank", {}).items():
            summary = cell.get("summary") or {}
            macro = summary.get("macro_f1") or {}
            weighted = summary.get("weighted_f1") or {}
            acc = summary.get("accuracy") or {}
            writer.writerow(
                {
                    "model": model_alias,
                    "bank": bank,
                    "support": summary.get("support", 0),
                    "n_checkpoints": macro.get("n_checkpoints", 0),
                    "ci_kind": macro.get("ci_kind", "missing"),
                    "macro_f1_point": _csv_value(macro.get("point")),
                    "macro_f1_lo": _csv_value(macro.get("lo")),
                    "macro_f1_hi": _csv_value(macro.get("hi")),
                    "weighted_f1_point": _csv_value(weighted.get("point")),
                    "weighted_f1_lo": _csv_value(weighted.get("lo")),
                    "weighted_f1_hi": _csv_value(weighted.get("hi")),
                    "accuracy_point": _csv_value(acc.get("point")),
                    "accuracy_lo": _csv_value(acc.get("lo")),
                    "accuracy_hi": _csv_value(acc.get("hi")),
                }
            )
    return buffer.getvalue()


def render_markdown(matrix: dict[str, Any], *, coverage: float = 0.95) -> str:
    coverage_pct = int(round(coverage * 100))
    lines = [
        f"| Model | Bank | Support | n_ckpt | macro-F1 ({coverage_pct}% CI when n≥2) | weighted-F1 | accuracy |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for model_alias, payload in matrix.get("by_model", {}).items():
        for bank, cell in payload.get("per_bank", {}).items():
            summary = cell.get("summary") or {}
            macro = summary.get("macro_f1") or {}
            n_ckpt = macro.get("n_checkpoints", 0)
            point = macro.get("point")
            lo = macro.get("lo")
            hi = macro.get("hi")
            if n_ckpt >= 2 and lo is not None and hi is not None:
                macro_cell = f"{point:.4f} [{lo:.4f}, {hi:.4f}]"
            elif point is not None:
                macro_cell = f"{point:.4f} (point)"
            else:
                macro_cell = "—"
            weighted_point = (summary.get("weighted_f1") or {}).get("point")
            acc_point = (summary.get("accuracy") or {}).get("point")
            lines.append(
                f"| `{model_alias}` | `{bank}` | {summary.get('support', 0)} | {n_ckpt} | "
                f"{macro_cell} | {weighted_point if weighted_point is None else f'{weighted_point:.4f}'} | "
                f"{acc_point if acc_point is None else f'{acc_point:.4f}'} |"
            )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cross-CB transfer matrix.")
    parser.add_argument("--training-package-id", required=True)
    parser.add_argument(
        "--eval-banks",
        default="",
        help="Comma-separated bank tokens (ecb,boj,boe,boc,rba). Defaults to all 5.",
    )
    parser.add_argument(
        "--model-checkpoints",
        required=True,
        help=(
            "Comma-separated alias=path pairs. Repeat an alias to accumulate "
            "per-seed checkpoints (e.g. finbert=p1,finbert=p2,finbert=p3 yields "
            "a 3-checkpoint CI for that alias)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write transfer_matrix.{json,csv,md}.",
    )
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--rng-seed", type=int, default=11)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    package_dir = _ensure_package_dir(DATA_DIR / "processed" / args.training_package_id)
    model_checkpoints = _parse_model_checkpoints(args.model_checkpoints)
    if not model_checkpoints:
        raise SystemExit("No model checkpoints provided.")
    banks = _split_banks(args.eval_banks)

    matrix = build_matrix(
        package_dir=package_dir,
        model_checkpoints=model_checkpoints,
        banks=banks,
        n_resamples=args.n_resamples,
        coverage=args.coverage,
        rng_seed=args.rng_seed,
    )

    output_dir = Path(args.output_dir) if args.output_dir else (
        DATA_DIR / "artifacts" / "cross_bank" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scrubbed = _scrub_nan(matrix)
    (output_dir / "transfer_matrix.json").write_text(
        json.dumps(scrubbed, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "transfer_matrix.csv").write_text(render_csv(matrix), encoding="utf-8")
    (output_dir / "transfer_matrix.md").write_text(render_markdown(matrix, coverage=args.coverage), encoding="utf-8")
    print(f"[transfer_matrix] wrote artefacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
