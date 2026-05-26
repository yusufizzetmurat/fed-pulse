"""Surface the freshest ``classification_breakdown`` block from the
regime sweep artifacts at ``data/artifacts/regime_*``.

Files are emitted by ``app/evaluation/classification_breakdown.py``
during training; their well-known shape is
``best_trial.summary.metrics.classification_breakdown``. The loader
walks every matching JSON, picks the most-recently-modified file that
actually carries the breakdown, and returns the block plus enough
provenance for the UI to deep-link back to the source. Missing artifact
returns a sentinel ``available=False`` payload so the dashboard renders
its fallback aggregation instead of erroring."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# Directories under ``data/artifacts/`` that the regime training scripts
# write breakdown JSON into. New sweep names should be added here when
# they ship; missing directories are tolerated.
_SCAN_PREFIXES: tuple[str, ...] = (
    "regime_baseline_tiers",
    "regime_arch_sweep",
    "regime_arch_sweep_chunk1",
    "regime_arch_sweep_macro_aug",
    "regime_capacity_push",
    "regime_pretrain_sweep",
    "regime_baseline",
)


@dataclass(frozen=True)
class BreakdownPayload:
    confusion_matrix: list[list[int]]
    per_class: list[dict[str, Any]]
    macro_f1: float | None
    macro_precision: float | None
    macro_recall: float | None
    macro_roc_auc: float | None
    macro_pr_auc: float | None
    weighted_f1: float | None
    n_classes: int | None
    class_labels: list[str] | None
    source_relative: str
    training_package_id: str | None
    checkpoint_path: str | None
    modified_at: str


def _iter_candidate_files(artifacts_root: Path) -> Iterable[Path]:
    for prefix in _SCAN_PREFIXES:
        root = artifacts_root / prefix
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            if path.is_file():
                yield path


def _extract_breakdown(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    best_trial = payload.get("best_trial")
    if not isinstance(best_trial, dict):
        return None
    summary = best_trial.get("summary")
    if not isinstance(summary, dict):
        return None
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return None
    breakdown = metrics.get("classification_breakdown")
    if not isinstance(breakdown, dict):
        return None
    if not isinstance(breakdown.get("confusion_matrix"), list):
        return None
    if not isinstance(breakdown.get("per_class"), list):
        return None
    return breakdown


# Skip very large sweep JSONs (>10MB). The breakdown-bearing artifacts
# we care about are small per-trial summaries; the multi-megabyte files
# are full hyperparameter sweep dumps that don't carry the breakdown
# block at all. Walking them costs hundreds of MB of memory on hosts
# that have accumulated several sweep runs.
_MAX_CANDIDATE_BYTES = 10 * 1024 * 1024


def _try_load(path: Path) -> dict[str, Any] | None:
    try:
        if path.stat().st_size > _MAX_CANDIDATE_BYTES:
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.debug("classification_breakdown skip %s: %s", path, exc)
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def load_latest(artifacts_root: Path) -> BreakdownPayload | None:
    """Walk the regime sweep directories under ``artifacts_root`` and
    return the freshest breakdown payload, or None when no artifact
    carries the expected shape."""

    best: tuple[float, Path, dict[str, Any], dict[str, Any]] | None = None
    for path in _iter_candidate_files(artifacts_root):
        payload = _try_load(path)
        if payload is None:
            continue
        breakdown = _extract_breakdown(payload)
        if breakdown is None:
            continue
        mtime = path.stat().st_mtime
        if best is None or mtime > best[0]:
            best = (mtime, path, payload, breakdown)
    if best is None:
        return None

    mtime, path, payload, breakdown = best
    relative = str(path.relative_to(artifacts_root))
    modified_iso = (
        datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    )

    per_class_raw = breakdown.get("per_class") or []
    per_class: list[dict[str, Any]] = []
    for entry in per_class_raw:
        if not isinstance(entry, dict):
            continue
        per_class.append(
            {
                "class_id": int(entry.get("class_id", 0)),
                "precision": float(entry.get("precision", 0.0)),
                "recall": float(entry.get("recall", 0.0)),
                "f1": float(entry.get("f1", 0.0)),
                "support": int(entry.get("support", 0)),
                "roc_auc": _maybe_float(entry.get("roc_auc")),
                "pr_auc": _maybe_float(entry.get("pr_auc")),
            }
        )

    labels = breakdown.get("class_labels")
    if not (isinstance(labels, list) and all(isinstance(x, str) for x in labels)):
        labels = None

    return BreakdownPayload(
        confusion_matrix=[
            [int(v) for v in row]
            for row in breakdown.get("confusion_matrix", [])
            if isinstance(row, list)
        ],
        per_class=per_class,
        macro_f1=_maybe_float(breakdown.get("macro_f1")),
        macro_precision=_maybe_float(breakdown.get("macro_precision")),
        macro_recall=_maybe_float(breakdown.get("macro_recall")),
        macro_roc_auc=_maybe_float(breakdown.get("macro_roc_auc")),
        macro_pr_auc=_maybe_float(breakdown.get("macro_pr_auc")),
        weighted_f1=_maybe_float(breakdown.get("weighted_f1")),
        n_classes=_maybe_int(breakdown.get("n_classes")),
        class_labels=labels,
        source_relative=relative,
        training_package_id=_maybe_str(payload.get("training_package_id")),
        checkpoint_path=_maybe_str(payload.get("checkpoint_path")),
        modified_at=modified_iso,
    )


def _maybe_float(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _maybe_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _maybe_str(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    return None
