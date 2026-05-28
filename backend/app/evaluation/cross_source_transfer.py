"""Cross-source transfer evaluation.

Given a stance-classification checkpoint trained on FOMC text and a training
package whose `registry_normalized.jsonl` carries labelled rows from multiple
``source_type`` strata (FOMC statements / minutes / meeting transcripts,
chair / governor speeches, congressional testimony, press conferences, Beige
Book, regional research, Op-Fed external corpus, etc.), compute per-source
macro-F1, accuracy, and per-class precision / recall / F1.

Inference-only. The trained checkpoint is held fixed; rows are filtered by
``source_type`` and scored against the model's existing weights. No
re-training, no leak — the eval simply asks "does the FOMC-trained model
generalise to source X?" for each source the registry carries labelled rows
for.

Output schema mirrors `cross_bank_transfer` so downstream aggregation can
share code paths: ``matrix.csv`` with one row per ``(encoder, source)`` pair
plus per-source ``support`` so under-populated cells are visible.

Continuous-target arm
---------------------
A second dispatch arm handles ``source_type`` strata that ship continuous
factor columns instead of categorical stance labels (today: GSS
target/path factor decomposition, ``gss_factor_decomposition``). The arm
runs the same stance checkpoint, derives a signed stance score
``P(hawkish) - P(dovish)`` per row, and reports Pearson + Spearman rank
correlation against the GSS target and path factors. Reported alongside is
a z-scored RMSE so both factor columns sit on a comparable scale; the raw
factor is in basis points and the stance score is in [-1, 1] so an
un-scaled RMSE would be meaningless.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.data.finetune_pilot import (
    ID2LABEL,
    LABELS,
    _compute_classification_metrics,
    _latency_summary,
)


# Canonical FOMC-side source_type strata that ship through the registry
# today. Extend by adding the source_type string here; the harness filters
# rows by exact match on the registry's ``source_type`` column.
CROSS_SOURCE_TYPES: tuple[str, ...] = (
    "fomc_statement",
    "fomc_minutes",
    "fomc_meeting_transcript",
    "fomc_press_conference",
    "chair_speech",
    "governor_speech",
    "congressional_testimony",
    "beige_book",
    "regional_research",
    "ny_fed_liberty_street",
    "gss_factor_decomposition",
)

# Continuous-target strata: rows carry no categorical stance label, they
# carry numeric factor columns lifted off ``multi_axis_extras``. The
# harness dispatches these through ``evaluate_continuous_source`` instead
# of the stance-classification path.
CROSS_SOURCE_CONTINUOUS_TYPES: tuple[str, ...] = (
    "gss_factor_decomposition",
)

# Per continuous source_type, the ``multi_axis_extras`` keys to score
# the model's signed stance score against. Order matters only for the
# CSV column layout downstream.
CONTINUOUS_TARGETS: dict[str, tuple[str, ...]] = {
    "gss_factor_decomposition": ("gss_target_factor", "gss_path_factor"),
}


@dataclass(frozen=True)
class CrossSourceRow:
    """A registry row read for the cross-source eval.

    ``label`` is empty string for rows from a continuous-target source_type
    (see ``CROSS_SOURCE_CONTINUOUS_TYPES``); the factor columns live on
    ``multi_axis_extras`` and the continuous-arm evaluator reads them
    directly.
    """

    record_id: str
    text: str
    label: str
    event_date: str
    source: str
    source_type: str
    provenance: str
    multi_axis_extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CrossSourceResult:
    source_type: str
    encoder_alias: str
    checkpoint: str
    support: int
    macro_f1: float
    weighted_f1: float
    accuracy: float
    per_class: dict[str, dict[str, float]]
    label_support: dict[str, int]
    latency_ms_p50: float
    latency_ms_p95: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "encoder_alias": self.encoder_alias,
            "checkpoint": self.checkpoint,
            "support": self.support,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "accuracy": self.accuracy,
            "per_class": self.per_class,
            "label_support": self.label_support,
            "latency_ms": {"p50": self.latency_ms_p50, "p95": self.latency_ms_p95},
        }


def load_cross_source_rows(
    package_dir: Path,
    *,
    include_zero_weight: bool = False,
) -> list[CrossSourceRow]:
    """Read every labelled row from the package registry that carries a
    canonical ``source_type``.

    ``include_zero_weight=False`` drops rows with ``sample_weight==0`` so
    cross-bank corpora (peer-reviewed-cross-bank provenance) and unlabelled
    archive rows (scraped, label="") do not enter the eval set. The base
    eval is FOMC-side cross-source only — cross-bank rides on its own
    harness (``app.evaluation.cross_bank_transfer``).
    """

    path = package_dir / "registry_normalized.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing registry: {path}")

    rows: list[CrossSourceRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        source_type = str(payload.get("source_type", "")).strip()
        if source_type not in CROSS_SOURCE_TYPES:
            continue
        text = str(payload.get("text", "")).strip()
        if not text:
            continue
        label = str(payload.get("mapped_label", "")).strip().lower()
        is_continuous = source_type in CROSS_SOURCE_CONTINUOUS_TYPES
        if not is_continuous and label not in LABELS:
            continue
        try:
            sample_weight = float(payload.get("sample_weight", 1.0))
        except (TypeError, ValueError):
            sample_weight = 1.0
        # Continuous-arm rows carry sample_weight=0 by design (they sit
        # in the registry for evaluation only, never enter training).
        # Don't gate them on the zero-weight flag.
        if not include_zero_weight and sample_weight == 0.0 and not is_continuous:
            continue
        extras_raw = payload.get("multi_axis_extras") or {}
        extras = extras_raw if isinstance(extras_raw, dict) else {}
        rows.append(
            CrossSourceRow(
                record_id=str(payload.get("record_id", "")).strip(),
                text=text,
                label=label if not is_continuous else "",
                event_date=str(payload.get("event_date", "")).strip(),
                source=str(payload.get("source", "")).strip(),
                source_type=source_type,
                provenance=str(payload.get("provenance", "")).strip(),
                multi_axis_extras=dict(extras),
            )
        )
    return rows


def group_by_source_type(
    rows: Iterable[CrossSourceRow],
) -> dict[str, list[CrossSourceRow]]:
    """Bucket rows by ``source_type``."""

    buckets: dict[str, list[CrossSourceRow]] = {}
    for row in rows:
        buckets.setdefault(row.source_type, []).append(row)
    return buckets


def _predict_with_model(
    rows: list[CrossSourceRow],
    *,
    checkpoint: str,
    max_length: int = 256,
    batch_size: int = 32,
) -> tuple[list[str], list[str], list[float]]:
    """Run inference using a HuggingFace classification checkpoint."""

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Honour the canonical TDW label order (dovish, hawkish, neutral) per the
    # cross_bank_transfer note — some legacy checkpoints carry an inverted
    # id2label that the patch script normalised in place.
    id2label = ID2LABEL

    y_true: list[str] = []
    y_pred: list[str] = []
    latencies: list[float] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch_texts = [r.text for r in batch_rows]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            t0 = time.perf_counter()
            logits = model(**enc).logits
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            per_item_ms = elapsed_ms / max(len(batch_rows), 1)
            latencies.extend([per_item_ms] * len(batch_rows))
            preds = logits.argmax(dim=-1).tolist()
            for row, pred_idx in zip(batch_rows, preds):
                y_true.append(row.label)
                y_pred.append(id2label[int(pred_idx)])
    return y_true, y_pred, latencies


def evaluate_source(
    rows: list[CrossSourceRow],
    *,
    source_type: str,
    encoder_alias: str,
    checkpoint: str,
    max_length: int = 256,
    batch_size: int = 32,
    predict_fn: Any = None,
) -> CrossSourceResult:
    """Score a single ``source_type`` slice end-to-end.

    ``predict_fn`` is injected by tests with a deterministic stub; production
    code passes ``None`` to fall through to the HF inference path.
    """

    if not rows:
        raise ValueError(
            f"No labelled rows for source_type={source_type!r} in registry."
        )
    record_ids = [r.record_id for r in rows if r.record_id]
    if record_ids and len(record_ids) != len(set(record_ids)):
        raise ValueError(
            f"Duplicate record_ids for source_type={source_type!r} — "
            "re-run ingest_sources to regenerate the registry."
        )

    if predict_fn is None:
        y_true, y_pred, latencies = _predict_with_model(
            rows, checkpoint=checkpoint, max_length=max_length, batch_size=batch_size
        )
    else:
        y_true, y_pred, latencies = predict_fn(rows)

    cls = _compute_classification_metrics(y_true, y_pred)
    latency = _latency_summary(latencies)
    label_support = dict(Counter(r.label for r in rows))
    return CrossSourceResult(
        source_type=source_type,
        encoder_alias=encoder_alias,
        checkpoint=checkpoint,
        support=len(rows),
        macro_f1=cls["macro_f1"],
        weighted_f1=cls["weighted_f1"],
        accuracy=cls["accuracy"],
        per_class=cls["per_class"],
        label_support=label_support,
        latency_ms_p50=latency["p50_ms"],
        latency_ms_p95=latency["p95_ms"],
    )


@dataclass(frozen=True)
class CrossSourceContinuousResult:
    """Continuous-arm result for a single (encoder, continuous source) cell.

    The stance checkpoint emits a signed score ``P(hawkish) - P(dovish)``
    per row; the harness scores that against each continuous target column
    on ``multi_axis_extras`` via Pearson + Spearman correlation and a
    z-scored RMSE (raw RMSE is meaningless cross-scale).
    """

    source_type: str
    encoder_alias: str
    checkpoint: str
    support: int
    # ``targets[target_key]`` carries ``support`` (int) plus ``pearson_r``,
    # ``spearman_r``, ``zscore_rmse`` which are ``float | None`` — None for
    # degenerate slices (zero variance, sub-2 pairs).
    targets: dict[str, dict[str, float | int | None]]
    latency_ms_p50: float
    latency_ms_p95: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "encoder_alias": self.encoder_alias,
            "checkpoint": self.checkpoint,
            "support": self.support,
            "kind": "continuous",
            "targets": self.targets,
            "latency_ms": {"p50": self.latency_ms_p50, "p95": self.latency_ms_p95},
        }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson r over paired finite samples. Returns ``None`` if degenerate."""

    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx2 = sum((x - mx) ** 2 for x in xs)
    dy2 = sum((y - my) ** 2 for y in ys)
    if dx2 <= 0 or dy2 <= 0:
        return None
    return num / math.sqrt(dx2 * dy2)


def _rank(values: list[float]) -> list[float]:
    """Average-rank assignment, ties get the mean of their tied positions."""

    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-indexed average
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation via Pearson on rank-transformed inputs."""

    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_rank(xs), _rank(ys))


def _zscore_rmse(xs: list[float], ys: list[float]) -> float | None:
    """RMSE after z-scoring both vectors. Returns ``None`` if degenerate."""

    n = len(xs)
    if n < 2 or len(ys) != n:
        return None

    def _z(vs: list[float]) -> list[float] | None:
        mean = sum(vs) / len(vs)
        var = sum((v - mean) ** 2 for v in vs) / len(vs)
        if var <= 0:
            return None
        sd = math.sqrt(var)
        return [(v - mean) / sd for v in vs]

    zx = _z(xs)
    zy = _z(ys)
    if zx is None or zy is None:
        return None
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(zx, zy)) / n)


def _predict_continuous_scores(
    rows: list[CrossSourceRow],
    *,
    checkpoint: str,
    max_length: int = 256,
    batch_size: int = 32,
) -> tuple[list[float], list[float]]:
    """Return ``(signed_scores, per_row_latency_ms)`` for the continuous arm.

    Signed score = ``softmax(logits)[hawkish_id] - softmax(logits)[dovish_id]``
    per row — a single bipolar dim in [-1, 1] that's the natural projection
    of a 3-class stance head onto the GSS factor axis.
    """

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    label2id = {v: k for k, v in ID2LABEL.items()}
    hawk = label2id["hawkish"]
    dove = label2id["dovish"]

    scores: list[float] = []
    latencies: list[float] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch_texts = [r.text for r in batch_rows]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            t0 = time.perf_counter()
            logits = model(**enc).logits
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            per_item_ms = elapsed_ms / max(len(batch_rows), 1)
            latencies.extend([per_item_ms] * len(batch_rows))
            probs = torch.softmax(logits, dim=-1)
            for prob in probs:
                scores.append(float(prob[hawk] - prob[dove]))
    return scores, latencies


def evaluate_continuous_source(
    rows: list[CrossSourceRow],
    *,
    source_type: str,
    encoder_alias: str,
    checkpoint: str,
    max_length: int = 256,
    batch_size: int = 32,
    predict_fn: Any = None,
) -> CrossSourceContinuousResult:
    """Score a continuous-target source slice.

    ``predict_fn(rows) -> (signed_scores, latencies_ms)`` is the test seam;
    production calls fall through to ``_predict_continuous_scores``.
    """

    if not rows:
        raise ValueError(
            f"No continuous rows for source_type={source_type!r} in registry."
        )
    target_keys = CONTINUOUS_TARGETS.get(source_type, ())
    if not target_keys:
        raise ValueError(
            f"No continuous targets configured for source_type={source_type!r}; "
            "extend CONTINUOUS_TARGETS to register the factor columns."
        )

    if predict_fn is None:
        scores, latencies = _predict_continuous_scores(
            rows, checkpoint=checkpoint, max_length=max_length, batch_size=batch_size
        )
    else:
        scores, latencies = predict_fn(rows)

    if len(scores) != len(rows):
        raise ValueError(
            f"predict_fn returned {len(scores)} scores for {len(rows)} rows."
        )

    targets: dict[str, dict[str, float | int | None]] = {}
    for key in target_keys:
        paired_x: list[float] = []
        paired_y: list[float] = []
        for row, score in zip(rows, scores):
            raw = row.multi_axis_extras.get(key)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isnan(value) or math.isinf(value):
                continue
            paired_x.append(score)
            paired_y.append(value)
        targets[key] = {
            "support": len(paired_x),
            "pearson_r": _pearson(paired_x, paired_y) if paired_x else None,
            "spearman_r": _spearman(paired_x, paired_y) if paired_x else None,
            "zscore_rmse": _zscore_rmse(paired_x, paired_y) if paired_x else None,
        }

    latency = _latency_summary(latencies)
    return CrossSourceContinuousResult(
        source_type=source_type,
        encoder_alias=encoder_alias,
        checkpoint=checkpoint,
        support=len(rows),
        targets=targets,
        latency_ms_p50=latency["p50_ms"],
        latency_ms_p95=latency["p95_ms"],
    )


def build_matrix(
    *,
    package_dir: Path,
    encoder_checkpoints: dict[str, str],
    source_types: list[str] | None = None,
    max_length: int = 256,
    batch_size: int = 32,
    predict_fn: Any = None,
    continuous_predict_fn: Any = None,
) -> dict[str, Any]:
    """Build the per-(encoder, source_type) cross-source transfer payload.

    ``encoder_checkpoints`` is ``{alias: checkpoint_path}``. The harness runs
    inference once per (alias, source_type) cell. Cells with zero rows are
    emitted as ``support=0`` with empty metrics so the under-populated
    sources stay visible in the CSV.

    Continuous-target source_types (see ``CROSS_SOURCE_CONTINUOUS_TYPES``)
    dispatch through ``evaluate_continuous_source`` and emit cells tagged
    ``kind="continuous"``. ``continuous_predict_fn`` is the corresponding
    test seam; production code passes ``None`` and lets the harness call
    the HF inference path.
    """

    rows = load_cross_source_rows(package_dir)
    buckets = group_by_source_type(rows)
    targets = source_types or list(CROSS_SOURCE_TYPES)

    matrix: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "training_package_id": package_dir.name,
        "encoders": list(encoder_checkpoints.keys()),
        "source_types": list(targets),
        "per_source_counts": {st: len(buckets.get(st, [])) for st in targets},
        "cells": [],
        "failures": [],
    }

    for encoder_alias, checkpoint in encoder_checkpoints.items():
        for source_type in targets:
            slice_rows = buckets.get(source_type, [])
            is_continuous = source_type in CROSS_SOURCE_CONTINUOUS_TYPES
            if not slice_rows:
                matrix["cells"].append(
                    {
                        "encoder_alias": encoder_alias,
                        "checkpoint": checkpoint,
                        "source_type": source_type,
                        "support": 0,
                        "status": "no_rows",
                        "kind": "continuous" if is_continuous else "stance",
                    }
                )
                continue
            try:
                if is_continuous:
                    cont = evaluate_continuous_source(
                        slice_rows,
                        source_type=source_type,
                        encoder_alias=encoder_alias,
                        checkpoint=checkpoint,
                        max_length=max_length,
                        batch_size=batch_size,
                        predict_fn=continuous_predict_fn,
                    )
                    cell = cont.to_dict()
                else:
                    result = evaluate_source(
                        slice_rows,
                        source_type=source_type,
                        encoder_alias=encoder_alias,
                        checkpoint=checkpoint,
                        max_length=max_length,
                        batch_size=batch_size,
                        predict_fn=predict_fn,
                    )
                    cell = result.to_dict()
                    cell["kind"] = "stance"
            except Exception as exc:  # noqa: BLE001 — surface per-cell failure
                matrix["failures"].append(
                    {
                        "encoder_alias": encoder_alias,
                        "checkpoint": checkpoint,
                        "source_type": source_type,
                        "error": str(exc),
                    }
                )
                continue
            cell["status"] = "ok"
            matrix["cells"].append(cell)

    return matrix


def render_csv(matrix: dict[str, Any]) -> str:
    """Render the stance-arm matrix as a CSV.

    Continuous-arm cells are skipped here so the stance CSV stays
    well-typed (the per-class columns don't apply). Continuous cells are
    emitted by ``render_continuous_csv``; both share ``matrix.json``.
    """

    fieldnames = [
        "encoder_alias",
        "checkpoint",
        "source_type",
        "status",
        "support",
        "dovish_n",
        "hawkish_n",
        "neutral_n",
        "macro_f1",
        "weighted_f1",
        "accuracy",
        "latency_ms_p50",
        "latency_ms_p95",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for cell in matrix.get("cells", []):
        if cell.get("kind") == "continuous":
            continue
        per_label = cell.get("label_support") or {}
        latency = cell.get("latency_ms") or {}
        writer.writerow(
            {
                "encoder_alias": cell.get("encoder_alias", ""),
                "checkpoint": cell.get("checkpoint", ""),
                "source_type": cell.get("source_type", ""),
                "status": cell.get("status", ""),
                "support": cell.get("support", 0),
                "dovish_n": per_label.get("dovish", 0),
                "hawkish_n": per_label.get("hawkish", 0),
                "neutral_n": per_label.get("neutral", 0),
                "macro_f1": _fmt(cell.get("macro_f1")),
                "weighted_f1": _fmt(cell.get("weighted_f1")),
                "accuracy": _fmt(cell.get("accuracy")),
                "latency_ms_p50": _fmt(latency.get("p50")),
                "latency_ms_p95": _fmt(latency.get("p95")),
            }
        )
    return buffer.getvalue()


def render_continuous_csv(matrix: dict[str, Any]) -> str:
    """Render the continuous-arm cells as a long-form CSV.

    One row per ``(encoder, source_type, target_key)`` tuple, so the
    Pearson / Spearman / z-RMSE numbers stay one-target-per-row regardless
    of how many factor columns a source ships. Returns an empty string if
    no continuous cells made it into the matrix (lets the caller skip the
    artefact instead of writing a header-only file).
    """

    rows: list[dict[str, Any]] = []
    for cell in matrix.get("cells", []):
        if cell.get("kind") != "continuous":
            continue
        latency = cell.get("latency_ms") or {}
        targets = cell.get("targets") or {}
        if not targets:
            rows.append(
                {
                    "encoder_alias": cell.get("encoder_alias", ""),
                    "checkpoint": cell.get("checkpoint", ""),
                    "source_type": cell.get("source_type", ""),
                    "status": cell.get("status", ""),
                    "support": cell.get("support", 0),
                    "target_key": "",
                    "paired_support": 0,
                    "pearson_r": "",
                    "spearman_r": "",
                    "zscore_rmse": "",
                    "latency_ms_p50": _fmt(latency.get("p50")),
                    "latency_ms_p95": _fmt(latency.get("p95")),
                }
            )
            continue
        for target_key, stats in targets.items():
            rows.append(
                {
                    "encoder_alias": cell.get("encoder_alias", ""),
                    "checkpoint": cell.get("checkpoint", ""),
                    "source_type": cell.get("source_type", ""),
                    "status": cell.get("status", ""),
                    "support": cell.get("support", 0),
                    "target_key": target_key,
                    "paired_support": stats.get("support", 0),
                    "pearson_r": _fmt(stats.get("pearson_r")),
                    "spearman_r": _fmt(stats.get("spearman_r")),
                    "zscore_rmse": _fmt(stats.get("zscore_rmse")),
                    "latency_ms_p50": _fmt(latency.get("p50")),
                    "latency_ms_p95": _fmt(latency.get("p95")),
                }
            )
    if not rows:
        return ""
    fieldnames = list(rows[0].keys())
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""


def _parse_encoder_spec(spec: str) -> dict[str, str]:
    """Parse ``alias=checkpoint[,alias=checkpoint]`` into a dict.

    Unlike the cross-bank transfer-matrix CLI we deliberately accept exactly
    one checkpoint per alias here — the cross-source eval reports a point
    estimate per (encoder, source) cell. Multiple checkpoints are a
    follow-up; if you need per-seed CIs, run this harness once per seed and
    aggregate downstream.
    """

    out: dict[str, str] = {}
    if not spec:
        return out
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"--encoder-checkpoints entry {piece!r} missing 'alias=path'")
        alias, path = piece.split("=", 1)
        alias = alias.strip()
        path = path.strip()
        if not alias or not path:
            raise ValueError(f"--encoder-checkpoints entry {piece!r} has empty alias or path")
        if alias in out:
            raise ValueError(
                f"--encoder-checkpoints alias {alias!r} duplicated; pass exactly one path per alias."
            )
        out[alias] = path
    return out


def _parse_source_types(spec: str) -> list[str] | None:
    if not spec:
        return None
    out: list[str] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token not in CROSS_SOURCE_TYPES:
            raise ValueError(
                f"unknown source_type {token!r}; allowed: {CROSS_SOURCE_TYPES}"
            )
        out.append(token)
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the cross-source transfer matrix.")
    parser.add_argument("--training-package-id", required=True)
    parser.add_argument(
        "--encoder-checkpoints",
        required=True,
        help=(
            "Comma-separated alias=path pairs (exactly one path per alias). "
            "Example: finbert_fed_adjacent=/path/to/ckpt"
        ),
    )
    parser.add_argument(
        "--source-types",
        default="",
        help=(
            "Comma-separated source_type strata to score. Defaults to the "
            "full canonical set; use this to restrict the matrix to a subset."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    from app.config import DATA_DIR

    args = _parse_args()
    package_dir = DATA_DIR / "processed" / args.training_package_id
    if not package_dir.exists():
        raise SystemExit(f"Training package not found: {package_dir}")
    encoder_checkpoints = _parse_encoder_spec(args.encoder_checkpoints)
    if not encoder_checkpoints:
        raise SystemExit("No encoder checkpoints provided.")
    source_types = _parse_source_types(args.source_types)

    matrix = build_matrix(
        package_dir=package_dir,
        encoder_checkpoints=encoder_checkpoints,
        source_types=source_types,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = DATA_DIR / "artifacts" / "v2_cross_source" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "matrix.json").write_text(
        json.dumps(matrix, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "matrix.csv").write_text(render_csv(matrix), encoding="utf-8")
    continuous_csv = render_continuous_csv(matrix)
    if continuous_csv:
        (output_dir / "matrix_continuous.csv").write_text(continuous_csv, encoding="utf-8")
    print(f"[cross_source_transfer] wrote artefacts to {output_dir}")
    return 0


__all__ = [
    "CONTINUOUS_TARGETS",
    "CROSS_SOURCE_CONTINUOUS_TYPES",
    "CROSS_SOURCE_TYPES",
    "CrossSourceContinuousResult",
    "CrossSourceResult",
    "CrossSourceRow",
    "build_matrix",
    "evaluate_continuous_source",
    "evaluate_source",
    "group_by_source_type",
    "load_cross_source_rows",
    "render_continuous_csv",
    "render_csv",
]


if __name__ == "__main__":
    raise SystemExit(main())
