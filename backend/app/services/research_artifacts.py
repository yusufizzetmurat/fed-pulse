"""Read-only index over ``data/artifacts/`` for the /research dashboard.

The Phase 8 bake-off + cross-bank scripts write JSON under three
directories (``phase3/``, ``cross_bank/``, ``cross_asset/``). This
module walks those directories, captures lightweight per-file
metadata, and parses the well-known shapes the dashboard renders:

* ``phase3/**/aggregate.json`` -- per-encoder macro-F1 (the bake-off
  aggregator emits one per seed batch).
* ``cross_bank/**/transfer_matrix.json`` -- source->target heatmap
  cells. We accept both ``cells`` arrays and dense matrix dicts.

Missing files are not errors -- the response simply marks the section
unavailable so the UI can render an empty state.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

SECTIONS: tuple[str, ...] = ("phase3", "cross_bank", "cross_asset", "next_fomc")


@dataclass(frozen=True)
class ArtifactFileInfo:
    relative_path: str
    size_bytes: int
    modified_at: str
    suffix: str


def list_section_files(artifacts_root: Path, section: str) -> list[ArtifactFileInfo]:
    """List every file under ``<root>/<section>/`` with basic metadata.

    Returned paths are relative to ``artifacts_root``. Hidden files and
    empty directories are skipped. A missing section directory returns
    an empty list -- this is the empty-state contract.
    """

    section_dir = artifacts_root / section
    if not section_dir.is_dir():
        return []
    out: list[ArtifactFileInfo] = []
    for path in sorted(section_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        out.append(
            ArtifactFileInfo(
                relative_path=str(path.relative_to(artifacts_root)),
                size_bytes=int(stat.st_size),
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                suffix=path.suffix.lower(),
            )
        )
    return out


def _iter_aggregate_files(artifacts_root: Path) -> Iterable[Path]:
    section_dir = artifacts_root / "phase3"
    if not section_dir.is_dir():
        return
    yield from sorted(section_dir.rglob("aggregate.json"))


def load_encoder_bakeoff(artifacts_root: Path) -> dict[str, Any]:
    """Aggregate per-encoder macro-F1 across phase3/**/aggregate.json files.

    Returns a dict with ``available``, ``coverage``, ``rows``, and
    ``source_files``. ``rows`` is a list of dicts shaped like
    :class:`EncoderBakeoffRow` (see ``app.schemas``). When no aggregate
    files exist, ``available`` is False and ``rows`` is empty.
    """

    files = list(_iter_aggregate_files(artifacts_root))
    if not files:
        return {
            "available": False,
            "coverage": None,
            "rows": [],
            "source_files": [],
        }

    by_encoder: dict[str, dict[str, Any]] = {}
    coverage: float | None = None
    source_files: list[str] = []
    for path in files:
        source_files.append(str(path.relative_to(artifacts_root)))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("research_artifacts: failed to parse %s: %s", path, exc)
            continue
        if coverage is None and "coverage" in payload:
            try:
                coverage = float(payload["coverage"])
            except (TypeError, ValueError):
                coverage = None
        for encoder_key, encoder_payload in (payload.get("by_encoder") or {}).items():
            bucket = by_encoder.setdefault(
                encoder_key,
                {
                    "encoder_key": encoder_key,
                    "checkpoint": str(encoder_payload.get("checkpoint", "")),
                    "seeds": [],
                    "macro_f1_values": [],
                    "weighted_f1_values": [],
                    "accuracy_values": [],
                    "cohen_kappa_values": [],
                },
            )
            for seed_str, per_seed in (encoder_payload.get("per_seed") or {}).items():
                try:
                    seed_int = int(seed_str)
                except (TypeError, ValueError):
                    continue
                if seed_int in bucket["seeds"]:
                    continue
                bucket["seeds"].append(seed_int)
                bucket["macro_f1_values"].append(_safe_float(per_seed.get("macro_f1")))
                bucket["weighted_f1_values"].append(_safe_float(per_seed.get("weighted_f1")))
                bucket["accuracy_values"].append(_safe_float(per_seed.get("accuracy")))
                kappa = per_seed.get("cohen_kappa")
                if kappa is not None:
                    bucket["cohen_kappa_values"].append(_safe_float(kappa))
            # Some aggregates also expose a confidence interval.
            ci = encoder_payload.get("macro_f1_ci") or {}
            if ci and "ci_low" not in bucket:
                bucket["ci_low"] = _safe_float_or_none(ci.get("low"))
                bucket["ci_high"] = _safe_float_or_none(ci.get("high"))

    rows: list[dict[str, Any]] = []
    for encoder_key in sorted(by_encoder):
        bucket = by_encoder[encoder_key]
        macro_vals = bucket["macro_f1_values"]
        rows.append(
            {
                "encoder_key": encoder_key,
                "checkpoint": bucket["checkpoint"],
                "seeds": list(bucket["seeds"]),
                "macro_f1_values": list(macro_vals),
                "macro_f1_mean": _mean_or_zero(macro_vals),
                "macro_f1_ci_low": bucket.get("ci_low"),
                "macro_f1_ci_high": bucket.get("ci_high"),
                "weighted_f1_mean": _mean_or_zero(bucket["weighted_f1_values"]),
                "accuracy_mean": _mean_or_zero(bucket["accuracy_values"]),
                "cohen_kappa": _mean_or_none(bucket["cohen_kappa_values"]),
            }
        )

    return {
        "available": True,
        "coverage": coverage if coverage is not None else 0.95,
        "rows": rows,
        "source_files": source_files,
    }


def load_cross_bank_transfer(artifacts_root: Path) -> dict[str, Any]:
    """Read transfer-matrix JSON files under ``cross_bank/``.

    Accepts two shapes:

    * ``{"sources": [...], "targets": [...], "cells": [{"source", "target", "metric"}, ...]}``
    * ``{"matrix": {"source": {"target": 0.42, ...}, ...}, "metric_name": "macro_f1"}``

    Returns the cell-list normalised form.
    """

    section_dir = artifacts_root / "cross_bank"
    if not section_dir.is_dir():
        return {
            "available": False,
            "metric_name": "macro_f1",
            "sources": [],
            "targets": [],
            "cells": [],
            "source_files": [],
        }
    files = sorted(section_dir.rglob("transfer_matrix.json"))
    if not files:
        return {
            "available": False,
            "metric_name": "macro_f1",
            "sources": [],
            "targets": [],
            "cells": [],
            "source_files": [],
        }

    sources: list[str] = []
    targets: list[str] = []
    cells: list[dict[str, Any]] = []
    metric_name = "macro_f1"
    source_files: list[str] = []
    for path in files:
        source_files.append(str(path.relative_to(artifacts_root)))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("research_artifacts: failed to parse %s: %s", path, exc)
            continue
        metric_name = str(payload.get("metric_name", metric_name))
        if "cells" in payload:
            for cell in payload["cells"]:
                try:
                    cells.append(
                        {
                            "source": str(cell["source"]),
                            "target": str(cell["target"]),
                            "metric": _safe_float(cell.get("metric")),
                        }
                    )
                except (KeyError, TypeError):
                    continue
            for src in payload.get("sources", []):
                if str(src) not in sources:
                    sources.append(str(src))
            for tgt in payload.get("targets", []):
                if str(tgt) not in targets:
                    targets.append(str(tgt))
        elif "matrix" in payload:
            matrix = payload["matrix"]
            if not isinstance(matrix, dict):
                continue
            for src, row in matrix.items():
                if not isinstance(row, dict):
                    continue
                if str(src) not in sources:
                    sources.append(str(src))
                for tgt, value in row.items():
                    if str(tgt) not in targets:
                        targets.append(str(tgt))
                    cells.append(
                        {
                            "source": str(src),
                            "target": str(tgt),
                            "metric": _safe_float(value),
                        }
                    )

    return {
        "available": bool(cells),
        "metric_name": metric_name,
        "sources": sources,
        "targets": targets,
        "cells": cells,
        "source_files": source_files,
    }


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))
