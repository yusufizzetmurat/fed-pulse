"""Aggregate per-bank cross-CB evaluations into a transfer matrix.

Walks every (model, bank, seed) cell, calls
:func:`app.evaluation.cross_bank_transfer.evaluate_cross_bank`, and produces:

- ``transfer_matrix.json`` — full nested payload (per seed, per bank, per class).
- ``transfer_matrix.csv`` — flat matrix: rows = bank, cols = metric, with
  point estimates plus block-bootstrap 95% CIs across seeds.
- ``transfer_matrix.md`` — same as csv but markdown.

Models are passed as ``--model-checkpoints alias=path[,alias=path,…]``. Each
alias maps to a HF-loadable directory or repo id; the CLI ignores aliases
whose path does not exist (so unfinished checkpoints don't crash the run).
"""

from __future__ import annotations

import argparse
import csv
import json
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


def _ensure_package_dir(package_dir: Path) -> Path:
    if not package_dir.exists():
        raise FileNotFoundError(f"Training package not found: {package_dir}")
    return package_dir


def _parse_model_checkpoints(spec: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not spec:
        return out
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"--model-checkpoints entry {piece!r} missing 'alias=path'")
        alias, path = piece.split("=", 1)
        out[alias.strip()] = path.strip()
    return out


def _split_banks(spec: str) -> list[str]:
    if not spec:
        return list(CROSS_BANK_SOURCES)
    out: list[str] = []
    aliases = {
        "ecb": "gtfintechlab_european_central_bank",
        "boj": "gtfintechlab_bank_of_japan",
        "boe": "gtfintechlab_bank_of_england",
        "boc": "gtfintechlab_bank_of_canada",
        "rba": "gtfintechlab_reserve_bank_of_australia",
    }
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in aliases:
            out.append(aliases[token])
        elif token in CROSS_BANK_SOURCES:
            out.append(token)
        else:
            raise ValueError(f"unknown bank token: {token!r}")
    return out


def _aggregate_seed_runs(
    results: list[CrossBankResult],
    *,
    n_resamples: int,
    coverage: float,
    seed: int,
) -> dict[str, Any]:
    """Aggregate multiple-seed CrossBankResult instances into mean + CI."""

    if not results:
        return {}
    macro = [r.macro_f1 for r in results]
    weighted = [r.weighted_f1 for r in results]
    acc = [r.accuracy for r in results]
    return {
        "support": results[0].support,
        "macro_f1": asdict(
            block_bootstrap_ci(
                macro, statistic="mean", block_size=1, n_resamples=n_resamples,
                coverage=coverage, seed=seed,
            )
        ),
        "weighted_f1": asdict(
            block_bootstrap_ci(
                weighted, statistic="mean", block_size=1, n_resamples=n_resamples,
                coverage=coverage, seed=seed,
            )
        ),
        "accuracy": asdict(
            block_bootstrap_ci(
                acc, statistic="mean", block_size=1, n_resamples=n_resamples,
                coverage=coverage, seed=seed,
            )
        ),
        "n_seeds": len(results),
    }


def build_matrix(
    *,
    package_dir: Path,
    model_checkpoints: dict[str, str],
    banks: list[str],
    seeds: list[int],
    n_resamples: int = 1000,
    coverage: float = 0.95,
    rng_seed: int = 11,
    predict_fn=None,
) -> dict[str, Any]:
    """Build the transfer-matrix payload by iterating over (model, bank, seed)."""

    matrix: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "training_package_id": package_dir.name,
        "banks": banks,
        "models": list(model_checkpoints.keys()),
        "seeds": seeds,
        "by_model": {},
    }

    for model_alias, checkpoint in model_checkpoints.items():
        per_model: dict[str, Any] = {"checkpoint": checkpoint, "per_bank": {}}
        for bank in banks:
            per_seed: list[CrossBankResult] = []
            for seed_val in seeds:
                try:
                    result = evaluate_cross_bank(
                        package_dir=package_dir,
                        bank_source=bank,
                        checkpoint=checkpoint,
                        predict_fn=predict_fn,
                    )
                except Exception as exc:  # noqa: BLE001 — surface per-cell failure
                    per_model.setdefault("failures", []).append(
                        {"bank": bank, "seed": seed_val, "error": str(exc)}
                    )
                    continue
                per_seed.append(result)
            if per_seed:
                per_model["per_bank"][bank] = {
                    "summary": _aggregate_seed_runs(
                        per_seed,
                        n_resamples=n_resamples,
                        coverage=coverage,
                        seed=rng_seed,
                    ),
                    "per_seed": [r.to_dict() for r in per_seed],
                }
        matrix["by_model"][model_alias] = per_model
    return matrix


def render_csv(matrix: dict[str, Any]) -> str:
    """Flat CSV: one row per (model, bank). Columns: macro_f1 point/lo/hi + …"""

    fieldnames = [
        "model", "bank", "support",
        "macro_f1_mean", "macro_f1_lo", "macro_f1_hi",
        "weighted_f1_mean", "weighted_f1_lo", "weighted_f1_hi",
        "accuracy_mean", "accuracy_lo", "accuracy_hi",
        "n_seeds",
    ]
    import io

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
                    "macro_f1_mean": macro.get("point", float("nan")),
                    "macro_f1_lo": macro.get("lo", float("nan")),
                    "macro_f1_hi": macro.get("hi", float("nan")),
                    "weighted_f1_mean": weighted.get("point", float("nan")),
                    "weighted_f1_lo": weighted.get("lo", float("nan")),
                    "weighted_f1_hi": weighted.get("hi", float("nan")),
                    "accuracy_mean": acc.get("point", float("nan")),
                    "accuracy_lo": acc.get("lo", float("nan")),
                    "accuracy_hi": acc.get("hi", float("nan")),
                    "n_seeds": summary.get("n_seeds", 0),
                }
            )
    return buffer.getvalue()


def render_markdown(matrix: dict[str, Any], *, coverage: float = 0.95) -> str:
    coverage_pct = int(round(coverage * 100))
    lines = [
        f"| Model | Bank | Support | macro-F1 ({coverage_pct}% CI) | weighted-F1 | accuracy |",
        "|---|---|---:|---|---|---|",
    ]
    for model_alias, payload in matrix.get("by_model", {}).items():
        for bank, cell in payload.get("per_bank", {}).items():
            summary = cell.get("summary") or {}
            macro = summary.get("macro_f1") or {}
            weighted = summary.get("weighted_f1") or {}
            acc = summary.get("accuracy") or {}
            lines.append(
                f"| `{model_alias}` | `{bank}` | {summary.get('support', 0)} | "
                f"{macro.get('point', float('nan')):.4f} "
                f"[{macro.get('lo', float('nan')):.4f}, {macro.get('hi', float('nan')):.4f}] | "
                f"{weighted.get('point', float('nan')):.4f} | "
                f"{acc.get('point', float('nan')):.4f} |"
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
        "--seeds",
        default="11,29,47,71,97",
        help="Comma-separated seeds — only affects CI computation (eval is deterministic).",
    )
    parser.add_argument(
        "--model-checkpoints",
        required=True,
        help="Comma-separated alias=path pairs.",
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
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    matrix = build_matrix(
        package_dir=package_dir,
        model_checkpoints=model_checkpoints,
        banks=banks,
        seeds=seeds,
        n_resamples=args.n_resamples,
        coverage=args.coverage,
        rng_seed=args.rng_seed,
    )

    output_dir = Path(args.output_dir) if args.output_dir else (
        DATA_DIR / "artifacts" / "cross_bank" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "transfer_matrix.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    (output_dir / "transfer_matrix.csv").write_text(render_csv(matrix), encoding="utf-8")
    (output_dir / "transfer_matrix.md").write_text(render_markdown(matrix, coverage=args.coverage), encoding="utf-8")
    print(f"[transfer_matrix] wrote artefacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
