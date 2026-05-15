"""Aggregate :mod:`app.data.finetune_batch` outputs into a headline table.

Reads one or more ``aggregate.json`` files (or a directory containing them),
re-derives per-encoder block-bootstrap CIs from the per-seed numbers, and
writes both JSON and a markdown table sortable by macro-F1.

Usage::

    python -m app.evaluation.bakeoff_aggregator \
        --artifact-dir data/artifacts/phase3/ \
        --output-dir   data/artifacts/phase3/bakeoff_summary
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
class EncoderRow:
    encoder_key: str
    checkpoint: str
    seeds: list[int]
    macro_f1_values: list[float]
    weighted_f1_values: list[float]
    accuracy_values: list[float]
    macro_f1_ci: BootstrapCI
    weighted_f1_ci: BootstrapCI
    accuracy_ci: BootstrapCI


def _load_aggregate(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_aggregate_files(artifact_dir: Path) -> Iterable[Path]:
    if artifact_dir.is_file() and artifact_dir.name == "aggregate.json":
        yield artifact_dir
        return
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"artifact_dir does not exist: {artifact_dir}")
    yield from sorted(artifact_dir.glob("**/aggregate.json"))


def _collect_per_encoder(aggregates: list[dict]) -> dict[str, dict]:
    by_encoder: dict[str, dict] = {}
    for aggregate in aggregates:
        for encoder_key, payload in (aggregate.get("by_encoder") or {}).items():
            bucket = by_encoder.setdefault(
                encoder_key,
                {
                    "checkpoint": payload.get("checkpoint", ""),
                    "seeds": [],
                    "macro_f1": [],
                    "weighted_f1": [],
                    "accuracy": [],
                },
            )
            for seed_str, per_seed in (payload.get("per_seed") or {}).items():
                try:
                    seed_int = int(seed_str)
                except ValueError:
                    continue
                if seed_int in bucket["seeds"]:
                    continue
                bucket["seeds"].append(seed_int)
                bucket["macro_f1"].append(float(per_seed.get("macro_f1", 0.0)))
                bucket["weighted_f1"].append(float(per_seed.get("weighted_f1", 0.0)))
                bucket["accuracy"].append(float(per_seed.get("accuracy", 0.0)))
    return by_encoder


def _build_rows(
    by_encoder: dict[str, dict],
    *,
    block_size: int,
    n_resamples: int,
    coverage: float,
    seed: int,
) -> list[EncoderRow]:
    rows: list[EncoderRow] = []
    for encoder_key, payload in by_encoder.items():
        rows.append(
            EncoderRow(
                encoder_key=encoder_key,
                checkpoint=str(payload["checkpoint"]),
                seeds=list(payload["seeds"]),
                macro_f1_values=list(payload["macro_f1"]),
                weighted_f1_values=list(payload["weighted_f1"]),
                accuracy_values=list(payload["accuracy"]),
                macro_f1_ci=block_bootstrap_ci(
                    payload["macro_f1"],
                    statistic="mean",
                    block_size=block_size,
                    n_resamples=n_resamples,
                    coverage=coverage,
                    seed=seed,
                ),
                weighted_f1_ci=block_bootstrap_ci(
                    payload["weighted_f1"],
                    statistic="mean",
                    block_size=block_size,
                    n_resamples=n_resamples,
                    coverage=coverage,
                    seed=seed,
                ),
                accuracy_ci=block_bootstrap_ci(
                    payload["accuracy"],
                    statistic="mean",
                    block_size=block_size,
                    n_resamples=n_resamples,
                    coverage=coverage,
                    seed=seed,
                ),
            )
        )
    rows.sort(key=lambda r: r.macro_f1_ci.point, reverse=True)
    return rows


def render_markdown(rows: list[EncoderRow], *, coverage: float) -> str:
    if not rows:
        return "_no encoder rows found_\n"
    coverage_pct = int(round(coverage * 100))
    lines = [
        f"| Rank | Encoder | Checkpoint | n | macro-F1 (mean, {coverage_pct}% CI) | weighted-F1 | accuracy |",
        "|---:|---|---|---:|---|---|---|",
    ]
    for rank, row in enumerate(rows, start=1):
        n = len(row.macro_f1_values)
        mf = row.macro_f1_ci
        wf = row.weighted_f1_ci
        ac = row.accuracy_ci
        lines.append(
            f"| {rank} | `{row.encoder_key}` | `{row.checkpoint}` | {n} | "
            f"{mf.point:.4f} [{mf.lo:.4f}, {mf.hi:.4f}] | "
            f"{wf.point:.4f} [{wf.lo:.4f}, {wf.hi:.4f}] | "
            f"{ac.point:.4f} [{ac.lo:.4f}, {ac.hi:.4f}] |"
        )
    return "\n".join(lines) + "\n"


def _row_to_json(row: EncoderRow) -> dict:
    return {
        "encoder_key": row.encoder_key,
        "checkpoint": row.checkpoint,
        "seeds": row.seeds,
        "macro_f1": {
            "values": row.macro_f1_values,
            "ci": asdict(row.macro_f1_ci),
        },
        "weighted_f1": {
            "values": row.weighted_f1_values,
            "ci": asdict(row.weighted_f1_ci),
        },
        "accuracy": {
            "values": row.accuracy_values,
            "ci": asdict(row.accuracy_ci),
        },
    }


def aggregate(
    artifact_dir: Path,
    *,
    block_size: int = 1,
    n_resamples: int = 1000,
    coverage: float = 0.95,
    seed: int = 11,
) -> tuple[list[EncoderRow], str, dict]:
    aggregate_paths = list(_iter_aggregate_files(artifact_dir))
    if not aggregate_paths:
        raise FileNotFoundError(f"no aggregate.json files found under {artifact_dir}")
    aggregates = [_load_aggregate(p) for p in aggregate_paths]
    by_encoder = _collect_per_encoder(aggregates)
    rows = _build_rows(
        by_encoder,
        block_size=block_size,
        n_resamples=n_resamples,
        coverage=coverage,
        seed=seed,
    )
    markdown = render_markdown(rows, coverage=coverage)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifact_dir": str(artifact_dir),
        "source_aggregates": [str(p) for p in aggregate_paths],
        "block_size": block_size,
        "n_resamples": n_resamples,
        "coverage": coverage,
        "encoders": [_row_to_json(row) for row in rows],
    }
    return rows, markdown, payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate bake-off runs into a headline table.")
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

    output_dir = args.output_dir or (args.artifact_dir / "bakeoff_summary")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"bakeoff_summary_{timestamp}.json"
    md_path = output_dir / f"bakeoff_summary_{timestamp}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    print(f"[bakeoff_aggregator] {len(rows)} encoders → {json_path}")
    print(f"[bakeoff_aggregator] markdown → {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
