"""Confounder-control ablation runner (#495).

Mirrors the per-family ablation pattern (``scripts/run_per_family_ablation.py``)
but the cells are confounder controls, not feature-family zero-outs.
Each cell appends a per-event control block to the rich-feature input
and re-trains the dual-head classifier on the canonical fold protocol;
the resulting per-cell deltas measure whether the encoder's edge on
the headline regime-F1 survives after the controls are admitted.

Cells:

- ``baseline``       -- canonical rich-feature run; no control block is
                        appended (byte-identical to the per-family
                        ``baseline`` cell).
- ``year_fe``        -- one-hot indicator over the observed year range
                        on the events parquet (``min_year .. max_year``
                        inclusive).
- ``meeting_type_fe`` -- one-hot indicator over the canonical event-kind
                        vocabulary ``(statement, minutes, press_conference,
                        speech, testimony, macro_release)``.
- ``doc_length``     -- ``log(1 + token_count)`` as a single scalar.
- ``all_three``      -- the three blocks above concatenated in the
                        documented order ``[year_fe, meeting_type_fe,
                        doc_length]``.

The runner attaches the control block to every FeatureVector in every
loaded sequence before ``train_model`` sees the partition. Sequences
are aligned positionally with ``WalkForwardSplit.{train,val,test}_event_dates``
and the per-event metadata read off ``events.parquet``; rows with
the same event_date are matched in ``text_hash`` order to mirror the
loader's sort. Cells that need per-event metadata not exposed on
``WalkForwardSplit`` (currently ``event_kind`` + ``token_count``) read
it off the events parquet at runner startup.

Output JSON shape (``backend/artifacts/experiments/confounder_ablation.json``)::

    {
      "cells": [...],
      "seeds": [...],
      "fold_ids": [...],
      "training_package_id": "...",
      "epochs": ...,
      "head_mode": "dual",
      "regression_alpha": 0.5,
      "year_range": [min_year, max_year],
      "meeting_kinds": [...],
      "trials": {
        "<cell>": [ {"seed": ..., "folds": [{"fold_id": ..., "metrics": {...}}]}, ... ]
      },
      "summary": {
        "<cell>": {
          "regime_f1_macro": {"mean": ..., "std": ..., "n": ...} | None,
          "regression_rmse_log_rv": {...} | None,
          "delta_vs_baseline": {
              "regime_f1_macro": float | None,
              "regression_rmse_log_rv": float | None,
          }
        }
      }
    }

Usage::

    docker compose --profile gpu run --rm backend-gpu \\
        python -m scripts.run_confounder_ablation \\
        --training-package-id <id> \\
        --seeds 11 29 47 71 97 \\
        --epochs 40 \\
        --output artifacts/experiments/confounder_ablation.json

The runner is re-runnable; the output path is overwritten on each call.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from app.config import BACKEND_ROOT
from app.training.runtime_compat import ensure_compile_safe


# Canonical event-kind vocabulary the events.parquet builder emits
# (see ``backend/app/data/event_dataset_builder.py:_EVENT_KIND_MAP``).
# Order is pinned so the one-hot column positions stay reproducible.
CANONICAL_MEETING_KINDS: tuple[str, ...] = (
    "statement",
    "minutes",
    "press_conference",
    "speech",
    "testimony",
    "macro_release",
)

CANONICAL_CELLS: tuple[str, ...] = (
    "baseline",
    "year_fe",
    "meeting_type_fe",
    "doc_length",
    "all_three",
)


@dataclass(frozen=True)
class EventMetadata:
    """Per-event control values read off the events parquet row."""

    text_hash: str
    event_date: str
    event_kind: str
    token_count: int


@dataclass(frozen=True)
class ConfounderSpec:
    """Resolved per-run spec: which blocks to attach + their widths."""

    cell: str
    use_year_fe: bool
    use_meeting_fe: bool
    use_doc_length: bool
    year_range: tuple[int, int]  # inclusive bounds; (0, -1) when unused

    @property
    def width(self) -> int:
        size = 0
        if self.use_year_fe:
            size += self.year_range[1] - self.year_range[0] + 1
        if self.use_meeting_fe:
            size += len(CANONICAL_MEETING_KINDS)
        if self.use_doc_length:
            size += 1
        return size


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-package-id",
        required=True,
        help="Training-package ID under ``backend/artifacts/training_packages/<id>``.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Defaults to "
            "``artifacts/experiments/confounder_ablation.json``."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[11, 29, 47, 71, 97],
        help="Official seed set.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=None,
        help=(
            "Subset of walk-forward fold IDs. Defaults to every fold "
            "in the package's fold_manifest_expanding_walk_forward.json."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Epochs per cell.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="Hidden size shared across every cell.",
    )
    parser.add_argument(
        "--head-mode",
        choices=("classification", "regression", "dual"),
        default="dual",
        help="Head mode for every cell (defaults to ``dual``).",
    )
    parser.add_argument(
        "--regression-alpha",
        type=float,
        default=0.5,
        help="alpha for head_mode='dual' joint loss.",
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        choices=CANONICAL_CELLS,
        default=list(CANONICAL_CELLS),
        help="Subset of cells to run (default: all five).",
    )
    return parser.parse_args()


def _resolve_output_path(arg: Path | None) -> Path:
    if arg is not None:
        return arg
    base = BACKEND_ROOT.parent / "artifacts" / "experiments"
    base.mkdir(parents=True, exist_ok=True)
    return base / "confounder_ablation.json"


# ---------------------------------------------------------------------------
# Events parquet -> per-event metadata
# ---------------------------------------------------------------------------


def _load_event_metadata(training_package_id: str) -> list[EventMetadata]:
    """Read events.parquet and return per-event metadata.

    Rows are returned sorted by ``(event_date, text_hash)`` so the order
    matches the loader's ordering inside
    ``_load_package_sequences_with_metadata``. Empty text_hash / event_date
    rows are skipped; duplicate text_hashes are deduplicated (first wins,
    same as the loader's ``seen`` set).

    Before the dedup pass the records are pre-sorted by
    ``(horizon, source)`` to mirror the loader's ``_row_rank`` ordering.
    Without this the runner and the loader would pick different
    canonical rows for any text_hash that appears at multiple horizons
    (and therefore derive ``event_kind`` / ``token_count`` from a row
    the loader discarded).
    """

    from app.training.loaders import _read_events_frame, _resolve_training_package_dir

    package_dir = _resolve_training_package_dir(training_package_id)
    frame = _read_events_frame(package_dir)
    if frame.empty:
        return []
    records = frame.to_dict("records")

    def _row_rank(row: dict[str, object]) -> tuple[int, str]:
        horizon = row.get("horizon")
        try:
            horizon_int = int(horizon) if horizon is not None else 10_000
        except (TypeError, ValueError):
            horizon_int = 10_000
        return (horizon_int, str(row.get("source", "")))

    records.sort(key=_row_rank)
    seen: set[str] = set()
    out: list[EventMetadata] = []
    for row in records:
        text_hash = str(row.get("text_hash", "") or "").strip()
        if not text_hash or text_hash in seen:
            continue
        seen.add(text_hash)
        event_date_raw = str(row.get("event_date", "") or "")[:10]
        if not event_date_raw:
            continue
        event_kind = str(row.get("event_kind", "") or "").strip()
        token_count_raw = row.get("token_count", 0)
        try:
            token_count = int(token_count_raw) if token_count_raw is not None else 0
        except (TypeError, ValueError):
            token_count = 0
        out.append(
            EventMetadata(
                text_hash=text_hash,
                event_date=event_date_raw,
                event_kind=event_kind,
                token_count=max(0, token_count),
            )
        )
    out.sort(key=lambda meta: (meta.event_date, meta.text_hash))
    return out


def _resolve_year_range(events: Sequence[EventMetadata]) -> tuple[int, int]:
    """Observed (min_year, max_year) range from events parquet event_dates."""

    years = [int(meta.event_date[:4]) for meta in events if len(meta.event_date) >= 4]
    if not years:
        raise ValueError(
            "Cannot derive year-FE range from an empty events.parquet metadata list."
        )
    return min(years), max(years)


# ---------------------------------------------------------------------------
# Cell -> ConfounderSpec dispatch
# ---------------------------------------------------------------------------


def _resolve_spec(cell: str, year_range: tuple[int, int]) -> ConfounderSpec:
    """Translate a cell label into a ConfounderSpec."""

    if cell == "baseline":
        return ConfounderSpec(
            cell=cell,
            use_year_fe=False,
            use_meeting_fe=False,
            use_doc_length=False,
            year_range=year_range,
        )
    if cell == "year_fe":
        return ConfounderSpec(
            cell=cell,
            use_year_fe=True,
            use_meeting_fe=False,
            use_doc_length=False,
            year_range=year_range,
        )
    if cell == "meeting_type_fe":
        return ConfounderSpec(
            cell=cell,
            use_year_fe=False,
            use_meeting_fe=True,
            use_doc_length=False,
            year_range=year_range,
        )
    if cell == "doc_length":
        return ConfounderSpec(
            cell=cell,
            use_year_fe=False,
            use_meeting_fe=False,
            use_doc_length=True,
            year_range=year_range,
        )
    if cell == "all_three":
        return ConfounderSpec(
            cell=cell,
            use_year_fe=True,
            use_meeting_fe=True,
            use_doc_length=True,
            year_range=year_range,
        )
    raise ValueError(f"unknown cell label: {cell!r}")


# ---------------------------------------------------------------------------
# Per-event confounder vector
# ---------------------------------------------------------------------------


def _build_year_one_hot(event_date: str, year_range: tuple[int, int]) -> list[float]:
    width = year_range[1] - year_range[0] + 1
    vector = [0.0] * width
    try:
        year = int(event_date[:4])
    except (TypeError, ValueError):
        return vector
    if year < year_range[0] or year > year_range[1]:
        return vector
    vector[year - year_range[0]] = 1.0
    return vector


def _build_meeting_kind_one_hot(event_kind: str) -> list[float]:
    vector = [0.0] * len(CANONICAL_MEETING_KINDS)
    try:
        idx = CANONICAL_MEETING_KINDS.index(event_kind)
    except ValueError:
        return vector
    vector[idx] = 1.0
    return vector


def _build_doc_length_scalar(token_count: int) -> list[float]:
    return [math.log1p(max(0, int(token_count)))]


def build_confounder_vector(meta: EventMetadata, spec: ConfounderSpec) -> list[float]:
    """Compose the per-event control vector under ``spec``.

    Block order is fixed: year_fe -> meeting_type_fe -> doc_length.
    Blocks the spec does not request are simply omitted so the cell's
    per-bar width equals ``spec.width``.
    """

    out: list[float] = []
    if spec.use_year_fe:
        out.extend(_build_year_one_hot(meta.event_date, spec.year_range))
    if spec.use_meeting_fe:
        out.extend(_build_meeting_kind_one_hot(meta.event_kind))
    if spec.use_doc_length:
        out.extend(_build_doc_length_scalar(meta.token_count))
    return out


# ---------------------------------------------------------------------------
# Sequence-attachment adapter
# ---------------------------------------------------------------------------


def _attach_confounder_block(
    sequences: list[list[Any]],
    event_dates: list[str],
    *,
    metadata_by_date: dict[str, list[EventMetadata]],
    spec: ConfounderSpec,
) -> None:
    """Write the cell's confounder vector onto every bar of every sequence.

    Sequence ``sequences[i]`` aligns with ``event_dates[i]``; multiple
    sequences may share an ``event_date`` (e.g. the same FOMC date carries
    both a ``statement`` and a ``press_conference`` row). For each
    same-date block in ``event_dates`` we consume the metadata in
    text_hash order — the loader sorts items by ``(event_date, text_hash)``
    so this preserves the positional alignment. When more sequences share
    a date than there are metadata rows (a degenerate fixture), the
    overflow falls back to the last metadata row for that date so the
    block stays the right width.
    """

    if spec.width == 0:
        return
    if len(sequences) != len(event_dates):
        raise ValueError(
            "sequences/event_dates length mismatch: "
            f"{len(sequences)} sequences vs {len(event_dates)} event_dates. "
            "The walk-forward loader must emit a date per supervised "
            "sequence; a silent zip-truncation would produce ragged "
            "per-bar widths once as_rich_list runs."
        )
    cursor: dict[str, int] = {}
    for sequence, event_date in zip(sequences, event_dates, strict=False):
        bucket = metadata_by_date.get(event_date, [])
        if not bucket:
            vector = [0.0] * spec.width
        else:
            idx = cursor.get(event_date, 0)
            meta = bucket[min(idx, len(bucket) - 1)]
            cursor[event_date] = idx + 1
            vector = build_confounder_vector(meta, spec)
        # Pad / truncate defensively so a malformed meta never widens
        # the per-bar tensor past ``spec.width``.
        if len(vector) < spec.width:
            vector = vector + [0.0] * (spec.width - len(vector))
        elif len(vector) > spec.width:
            vector = vector[: spec.width]
        for fv in sequence:
            fv.confounder_features = list(vector)


# ---------------------------------------------------------------------------
# Trial helpers (mirror the per-family runner shape)
# ---------------------------------------------------------------------------


def _trial_metrics(summary: Any) -> dict[str, float | None]:
    test = getattr(summary, "test_metrics", None) or getattr(summary, "metrics", None)
    if test is None:
        return {}
    return {
        "regime_f1_macro": getattr(test, "regime_f1_macro", None),
        "regime_accuracy": getattr(test, "regime_accuracy", None),
        "regime_loss": getattr(test, "regime_loss", None),
        "regression_rmse_log_rv": getattr(test, "regression_rmse_log_rv", None),
        "regression_mae_log_rv": getattr(test, "regression_mae_log_rv", None),
        "regression_loss": getattr(test, "regression_loss", None),
    }


def _summary_stats(values: list[float]) -> dict[str, float] | None:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return None
    return {
        "mean": statistics.fmean(finite),
        "std": statistics.pstdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "max": max(finite),
        "n": len(finite),
    }


def _resolve_fold_ids(training_package_id: str, override: list[str] | None) -> list[str]:
    if override:
        return list(override)
    from app.training.loaders import _read_fold_manifest, _resolve_training_package_dir

    package_dir = _resolve_training_package_dir(training_package_id)
    manifest = _read_fold_manifest(package_dir)
    if not manifest:
        raise RuntimeError(
            "fold_manifest_expanding_walk_forward.json is empty / missing "
            f"for training_package_id={training_package_id!r}; provide "
            "--folds explicitly."
        )
    return sorted(manifest.keys())


# ---------------------------------------------------------------------------
# Per-cell training loop
# ---------------------------------------------------------------------------


def _run_one_cell(
    spec: ConfounderSpec,
    seed: int,
    args: argparse.Namespace,
    *,
    fold_ids: list[str],
    metadata_by_date: dict[str, list[EventMetadata]],
) -> dict[str, Any]:
    """Train + evaluate one (cell, seed) cell across every fold."""

    from app.models.config import ModelConfig, RICH_FEATURE_SIZE
    from app.training.loaders import load_walk_forward_split
    from app.training.loop import train_model

    input_size = RICH_FEATURE_SIZE + spec.width
    config = ModelConfig(
        input_size=input_size,
        output_mode="classification",
        head_mode=str(args.head_mode),
        regression_alpha=float(args.regression_alpha),
        n_classes=3,
        hidden_size=int(args.hidden_size),
    )

    per_fold: list[dict[str, Any]] = []
    for fold_id in fold_ids:
        split = load_walk_forward_split(
            training_package_id=args.training_package_id,
            fold_id=fold_id,
            rich_features=True,
        )
        _attach_confounder_block(
            split.train,
            list(split.train_event_dates),
            metadata_by_date=metadata_by_date,
            spec=spec,
        )
        _attach_confounder_block(
            split.val,
            list(split.val_event_dates),
            metadata_by_date=metadata_by_date,
            spec=spec,
        )
        _attach_confounder_block(
            split.test,
            list(split.test_event_dates),
            metadata_by_date=metadata_by_date,
            spec=spec,
        )

        result = train_model(
            model_config=config,
            train_sequence_groups=split.train,
            val_sequence_groups=split.val,
            test_sequence_groups=split.test,
            fold_id=split.fold_id,
            protocol=split.protocol,
            epochs=int(args.epochs),
            seed=int(seed),
            save_checkpoint=False,
        )
        per_fold.append(
            {
                "fold_id": split.fold_id,
                "metrics": _trial_metrics(result.summary),
            }
        )

    return {
        "cell": spec.cell,
        "seed": seed,
        "confounder_width": spec.width,
        "folds": per_fold,
    }


def _group_metadata_by_date(
    events: Sequence[EventMetadata],
) -> dict[str, list[EventMetadata]]:
    """Bucket metadata by event_date, preserving text_hash sort order."""

    out: dict[str, list[EventMetadata]] = {}
    for meta in events:
        out.setdefault(meta.event_date, []).append(meta)
    for bucket in out.values():
        bucket.sort(key=lambda m: m.text_hash)
    return out


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _summarise_trials(
    cells: list[str],
    trials: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Per-cell summary stats + delta-vs-baseline. Pulled out of ``main``
    so the entrypoint stays under the C901 complexity ceiling."""

    summary: dict[str, dict[str, Any]] = {}
    baseline_means: dict[str, float | None] = {
        "regime_f1_macro": None,
        "regression_rmse_log_rv": None,
    }
    for cell in cells:
        f1_values: list[float] = []
        rmse_values: list[float] = []
        for trial in trials[cell]:
            for fold in trial["folds"]:
                metrics = fold.get("metrics", {}) or {}
                f1 = metrics.get("regime_f1_macro")
                rmse = metrics.get("regression_rmse_log_rv")
                if f1 is not None:
                    f1_values.append(float(f1))
                if rmse is not None:
                    rmse_values.append(float(rmse))
        f1_stats = _summary_stats(f1_values)
        rmse_stats = _summary_stats(rmse_values)
        summary[cell] = {
            "regime_f1_macro": f1_stats,
            "regression_rmse_log_rv": rmse_stats,
        }
        if cell == "baseline":
            baseline_means["regime_f1_macro"] = f1_stats["mean"] if f1_stats else None
            baseline_means["regression_rmse_log_rv"] = (
                rmse_stats["mean"] if rmse_stats else None
            )
    for cell in cells:
        cell_summary = summary[cell]
        deltas: dict[str, float | None] = {
            "regime_f1_macro": None,
            "regression_rmse_log_rv": None,
        }
        for metric_key in ("regime_f1_macro", "regression_rmse_log_rv"):
            baseline_value = baseline_means[metric_key]
            cell_stats = cell_summary.get(metric_key)
            if (
                cell != "baseline"
                and baseline_value is not None
                and cell_stats is not None
            ):
                deltas[metric_key] = float(cell_stats["mean"]) - float(baseline_value)
        cell_summary["delta_vs_baseline"] = deltas
    return summary


def main() -> int:
    ensure_compile_safe()
    args = _parse_args()
    output_path = _resolve_output_path(args.output)
    print(f"[confounder_ablation] writing -> {output_path}")

    fold_ids = _resolve_fold_ids(args.training_package_id, args.folds)
    print(f"[confounder_ablation] folds={fold_ids}")

    events = _load_event_metadata(args.training_package_id)
    if not events:
        raise RuntimeError(
            "events.parquet returned no usable rows for "
            f"training_package_id={args.training_package_id!r}"
        )
    year_range = _resolve_year_range(events)
    metadata_by_date = _group_metadata_by_date(events)
    print(
        f"[confounder_ablation] year_range={year_range} "
        f"meeting_kinds={CANONICAL_MEETING_KINDS} n_events={len(events)}"
    )

    cells = list(args.cells)
    print(f"[confounder_ablation] cells={cells}")

    trials: dict[str, list[dict[str, Any]]] = {cell: [] for cell in cells}
    for cell in cells:
        spec = _resolve_spec(cell, year_range)
        print(
            f"[confounder_ablation] >>> cell={cell} width={spec.width}",
            flush=True,
        )
        for seed in args.seeds:
            print(
                f"[confounder_ablation] {cell} seed={seed} epochs={args.epochs}",
                flush=True,
            )
            trials[cell].append(
                _run_one_cell(
                    spec,
                    seed,
                    args,
                    fold_ids=fold_ids,
                    metadata_by_date=metadata_by_date,
                )
            )

    summary = _summarise_trials(cells, trials)

    payload: dict[str, Any] = {
        "cells": cells,
        "seeds": list(args.seeds),
        "fold_ids": fold_ids,
        "training_package_id": args.training_package_id,
        "epochs": int(args.epochs),
        "hidden_size": int(args.hidden_size),
        "head_mode": str(args.head_mode),
        "regression_alpha": float(args.regression_alpha),
        "year_range": list(year_range),
        "meeting_kinds": list(CANONICAL_MEETING_KINDS),
        "trials": trials,
        "summary": summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[confounder_ablation] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
