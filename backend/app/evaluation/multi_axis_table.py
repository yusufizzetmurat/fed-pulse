"""Per-axis macro-F1 table across encoders.

Reads the bake-off ``aggregate.json`` output and produces a per-axis ×
per-encoder breakdown with mean, std, and bootstrap CI columns. The
``per_axis`` block on each encoder is the new shape produced by the
multi-axis-aware fine-tune batch (one block per axis: stance / factor /
certainty / time / topic); the legacy single-axis shape is treated as a
stance-only row so older aggregates render the same table layout.

Each row is one (axis, encoder, metric) cell. Output ships as both CSV
(for downstream tooling) and markdown (for direct paste into the wiki
or the thesis docs).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from app.evaluation.bootstrap import block_bootstrap_ci


DEFAULT_METRICS: tuple[str, ...] = ("macro_f1", "weighted_f1", "accuracy")
DEFAULT_AXES: tuple[str, ...] = ("stance", "factor", "certainty", "time", "topic")
LEGACY_AXIS: str = "stance"


@dataclass(frozen=True)
class MultiAxisRow:
    axis: str
    encoder: str
    metric: str
    mean: float
    std: float | None
    n: int
    ci_lo: float | None
    ci_hi: float | None
    samples: tuple[float, ...] = ()


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if result != result else result


def _samples_from_per_seed(per_seed: Any, metric: str) -> list[float]:
    if not isinstance(per_seed, Mapping):
        return []
    out: list[float] = []
    for raw in per_seed.values():
        if isinstance(raw, Mapping):
            value = _coerce_float(raw.get(metric))
        else:
            value = _coerce_float(raw)
        if value is not None:
            out.append(value)
    return out


def _row_from_samples(
    *,
    axis: str,
    encoder: str,
    metric: str,
    samples: Sequence[float],
    block_size: int,
    n_resamples: int,
    coverage: float,
    seed: int,
) -> MultiAxisRow | None:
    if not samples:
        return None
    n = len(samples)
    mean = sum(samples) / n
    std: float | None = None
    if n > 1:
        variance = sum((s - mean) ** 2 for s in samples) / (n - 1)
        std = variance**0.5
    ci_lo: float | None = None
    ci_hi: float | None = None
    if n > 1:
        ci = block_bootstrap_ci(
            list(samples),
            block_size=block_size,
            n_resamples=n_resamples,
            coverage=coverage,
            seed=seed,
        )
        ci_lo = float(ci.lo)
        ci_hi = float(ci.hi)
    return MultiAxisRow(
        axis=axis,
        encoder=encoder,
        metric=metric,
        mean=float(mean),
        std=std,
        n=n,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        samples=tuple(samples),
    )


def _rows_for_encoder(
    encoder_name: str,
    encoder_block: Mapping[str, Any],
    *,
    metrics: Iterable[str],
    block_size: int,
    n_resamples: int,
    coverage: float,
    seed: int,
) -> list[MultiAxisRow]:
    per_axis = encoder_block.get("per_axis")
    rows: list[MultiAxisRow] = []
    metric_list = list(metrics)
    if isinstance(per_axis, Mapping) and per_axis:
        # New shape: each axis carries its own per-seed metric block.
        for axis_name, axis_block in per_axis.items():
            if not isinstance(axis_block, Mapping):
                continue
            for metric in metric_list:
                samples = _samples_from_per_seed(axis_block.get("per_seed"), metric)
                row = _row_from_samples(
                    axis=str(axis_name),
                    encoder=encoder_name,
                    metric=metric,
                    samples=samples,
                    block_size=block_size,
                    n_resamples=n_resamples,
                    coverage=coverage,
                    seed=seed,
                )
                if row is not None:
                    rows.append(row)
        return rows

    # Legacy shape: encoder block has a per_seed map directly. Treat the
    # whole table as stance-axis values; downstream tooling renders one
    # row per (stance, encoder, metric) without losing the older table.
    for metric in metric_list:
        samples = _samples_from_per_seed(encoder_block.get("per_seed"), metric)
        row = _row_from_samples(
            axis=LEGACY_AXIS,
            encoder=encoder_name,
            metric=metric,
            samples=samples,
            block_size=block_size,
            n_resamples=n_resamples,
            coverage=coverage,
            seed=seed,
        )
        if row is not None:
            rows.append(row)
    return rows


def build_rows(
    aggregate: Mapping[str, Any],
    *,
    metrics: Iterable[str] = DEFAULT_METRICS,
    block_size: int = 1,
    n_resamples: int = 1000,
    coverage: float = 0.95,
    seed: int = 11,
) -> list[MultiAxisRow]:
    """Return the (axis, encoder, metric) rows for one aggregate.json payload."""

    by_encoder = aggregate.get("by_encoder") if isinstance(aggregate, Mapping) else None
    if not isinstance(by_encoder, Mapping):
        return []
    rows: list[MultiAxisRow] = []
    for encoder_name, encoder_block in by_encoder.items():
        if not isinstance(encoder_block, Mapping):
            continue
        rows.extend(
            _rows_for_encoder(
                str(encoder_name),
                encoder_block,
                metrics=metrics,
                block_size=block_size,
                n_resamples=n_resamples,
                coverage=coverage,
                seed=seed,
            )
        )
    return rows


def render_markdown(rows: Sequence[MultiAxisRow], *, coverage: float = 0.95) -> str:
    """Render the per-axis table as markdown grouped by axis."""

    if not rows:
        return "_no rows_\n"
    by_axis: dict[str, list[MultiAxisRow]] = {}
    for row in rows:
        by_axis.setdefault(row.axis, []).append(row)
    band = int(round(coverage * 100))
    out: list[str] = []
    for axis in sorted(by_axis.keys()):
        out.append(f"### Axis: `{axis}`")
        out.append("")
        out.append(f"| Encoder | Metric | n | mean | std | {band}% CI lo | {band}% CI hi |")
        out.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in by_axis[axis]:
            std = "—" if row.std is None else f"{row.std:.4f}"
            lo = "—" if row.ci_lo is None else f"{row.ci_lo:.4f}"
            hi = "—" if row.ci_hi is None else f"{row.ci_hi:.4f}"
            out.append(
                f"| {row.encoder} | {row.metric} | {row.n} | "
                f"{row.mean:.4f} | {std} | {lo} | {hi} |"
            )
        out.append("")
    return "\n".join(out)


def write_csv(rows: Sequence[MultiAxisRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["axis", "encoder", "metric", "n", "mean", "std", "ci_lo", "ci_hi"])
        for row in rows:
            writer.writerow(
                [
                    row.axis,
                    row.encoder,
                    row.metric,
                    row.n,
                    row.mean,
                    "" if row.std is None else row.std,
                    "" if row.ci_lo is None else row.ci_lo,
                    "" if row.ci_hi is None else row.ci_hi,
                ]
            )


def write_json(rows: Sequence[MultiAxisRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rows": [{k: v for k, v in asdict(row).items() if k != "samples"} for row in rows]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the per-axis × per-encoder bake-off table.")
    parser.add_argument("--aggregate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    aggregate = json.loads(Path(args.aggregate).read_text(encoding="utf-8"))
    rows = build_rows(
        aggregate,
        block_size=int(args.block_size),
        n_resamples=int(args.n_resamples),
        coverage=float(args.coverage),
        seed=int(args.seed),
    )
    output_dir = Path(args.output_dir)
    write_csv(rows, output_dir / "table.csv")
    write_json(rows, output_dir / "table.json")
    (output_dir / "table.md").write_text(
        render_markdown(rows, coverage=float(args.coverage)), encoding="utf-8"
    )
    print(f"[multi_axis_table] wrote {len(rows)} rows to {output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
