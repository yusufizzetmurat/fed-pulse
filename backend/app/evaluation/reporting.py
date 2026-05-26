"""Honest headline reporting variants for the vol-regime classifier (#323).

The pooled headline number (e.g. `0.4538`) sits on top of three
stratification choices that the cell does not announce: which event
types make it into the pool, whether `wf_fold_4` (zero-`calm`-class
test slice, R-17) is included, and whether the macro-release augmented
rows from §6.7 Chunk-3 are pooled with the FOMC test rows. The same
underlying predictions can produce four meaningfully different
headlines depending on these choices; the published cell must say
which one it cites.

This module supplies the four reporting variants the issue calls for.
Each helper consumes a sequence of per-row records — the row-level
test-partition surface PR #226 wired onto :class:`EvaluationMetrics`
(`predictions` / `targets` / per-row metadata) — and emits a single
report dict carrying:

- `macro_f1` — the aggregate macro-F1 on the variant's row subset
- `per_class` — per-class precision / recall / F1 / support
- `ci` — block-bootstrap confidence interval (1k resamples,
  block_size=20 by default, matching :mod:`app.evaluation.bootstrap`)
- `support` — total row count contributing to the variant
- `pooling` — `mean-of-fold-means` and `row-pooled` cells per the
  §Aggregation rule in `docs/benchmark-policy.md`

The helpers stay pure-Python (no numpy / sklearn dependency) so they
share the CI surface with :mod:`app.evaluation.classification_breakdown`
and :mod:`app.evaluation.regime_pooled_aggregator`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.evaluation.classification_breakdown import (
    ClassificationBreakdown,
    compute_classification_breakdown,
)


# ---------------------------------------------------------------------------
# Input record contract
# ---------------------------------------------------------------------------

# A single row record carries the stratification metadata each variant
# filters on. Records are dict-like so the helpers stay compatible with
# the JSONL surface emitted by `app.data.finetune_batch` and the
# row-level test predictions PR #226 wired onto `EvaluationMetrics`.
# Required keys: `prediction` (int class index), `target` (int class
# index). Optional keys consumed by individual helpers below:
# - `fold_id`             — e.g. `"wf_fold_4"`; consumed by `with_without_fold`
# - `source_type`         — e.g. `"fomc_statement"`; consumed by
#                           `fomc_only_macro_f1`
# - `is_macro_release`    — bool; consumed by `with_without_macro_release`
Record = Mapping[str, Any]


_FOMC_SOURCE_TYPES = {"fomc_statement", "fomc_minutes"}


# ---------------------------------------------------------------------------
# Public helpers — the four variants the issue requires
# ---------------------------------------------------------------------------


def mixed_pool_macro_f1(
    records: Sequence[Record],
    *,
    n_classes: int = 3,
    block_size: int = 20,
    n_resamples: int = 1000,
    coverage: float = 0.95,
    bootstrap_seed: int = 11,
    fold_field: str = "fold_id",
) -> dict[str, Any]:
    """Mixed-pool (current behaviour): every record contributes once.

    Reports both pooling conventions side-by-side per the §Aggregation
    rule — `mean-of-fold-means` (the canonical published cell) and
    `row-pooled` (the secondary cell every honest report must cite).
    """

    return _report_cell(
        records,
        n_classes=n_classes,
        block_size=block_size,
        n_resamples=n_resamples,
        coverage=coverage,
        bootstrap_seed=bootstrap_seed,
        fold_field=fold_field,
        label="mixed_pool",
    )


def fomc_only_macro_f1(
    records: Sequence[Record],
    *,
    source_type_field: str = "source_type",
    fomc_source_types: Iterable[str] = _FOMC_SOURCE_TYPES,
    n_classes: int = 3,
    block_size: int = 20,
    n_resamples: int = 1000,
    coverage: float = 0.95,
    bootstrap_seed: int = 11,
    fold_field: str = "fold_id",
) -> dict[str, Any]:
    """FOMC-only stratified headline (`fomc_statement` + `fomc_minutes`).

    The primary thesis number under the §6.7 honest-headline programme:
    every cross-bank or macro-release row drops out so the pool reflects
    the FOMC corpus the thesis claim is over.
    """

    allowed = {str(s).lower() for s in fomc_source_types}
    subset = [
        r
        for r in records
        if str(r.get(source_type_field, "")).lower() in allowed
    ]
    return _report_cell(
        subset,
        n_classes=n_classes,
        block_size=block_size,
        n_resamples=n_resamples,
        coverage=coverage,
        bootstrap_seed=bootstrap_seed,
        fold_field=fold_field,
        label="fomc_only",
    )


def with_without_fold(
    records: Sequence[Record],
    *,
    drop_fold_id: str,
    fold_field: str = "fold_id",
    n_classes: int = 3,
    block_size: int = 20,
    n_resamples: int = 1000,
    coverage: float = 0.95,
    bootstrap_seed: int = 11,
) -> dict[str, dict[str, Any]]:
    """Headline with-and-without a named fold (R-17 fold-4 contribution).

    Returns a `{"with": ..., "without": ..., "delta": ...}` dict so
    every published macro-F1 cell can quote both readings and the
    magnitude of the fold's contribution. `drop_fold_id` is matched
    case-sensitively against the record's `fold_field`.
    """

    target_fold = str(drop_fold_id)
    with_records = list(records)
    without_records = [
        r for r in records if str(r.get(fold_field, "")) != target_fold
    ]
    with_cell = _report_cell(
        with_records,
        n_classes=n_classes,
        block_size=block_size,
        n_resamples=n_resamples,
        coverage=coverage,
        bootstrap_seed=bootstrap_seed,
        fold_field=fold_field,
        label="with_fold",
    )
    without_cell = _report_cell(
        without_records,
        n_classes=n_classes,
        block_size=block_size,
        n_resamples=n_resamples,
        coverage=coverage,
        bootstrap_seed=bootstrap_seed,
        fold_field=fold_field,
        label="without_fold",
    )
    delta = with_cell["macro_f1"]["row_pooled"] - without_cell["macro_f1"][
        "row_pooled"
    ]
    return {
        "with": with_cell,
        "without": without_cell,
        "dropped_fold_id": target_fold,
        "delta_macro_f1_row_pooled": delta,
    }


def with_without_macro_release(
    records: Sequence[Record],
    *,
    is_macro_release_field: str = "is_macro_release",
    fold_field: str = "fold_id",
    n_classes: int = 3,
    block_size: int = 20,
    n_resamples: int = 1000,
    coverage: float = 0.95,
    bootstrap_seed: int = 11,
) -> dict[str, dict[str, Any]]:
    """Headline with-and-without macro-release augmented rows.

    The §6.7 Chunk-3 augmentation pulls CPI / NFP release-date rows
    into the supervised pool. Reporting both cells side-by-side keeps
    the lift attribution auditable — the published macro-F1 must say
    whether the augmentation rows were inside the headline or not.
    """

    with_cell = _report_cell(
        records,
        n_classes=n_classes,
        block_size=block_size,
        n_resamples=n_resamples,
        coverage=coverage,
        bootstrap_seed=bootstrap_seed,
        fold_field=fold_field,
        label="with_macro_release",
    )
    without_records = [
        r for r in records if not bool(r.get(is_macro_release_field, False))
    ]
    without_cell = _report_cell(
        without_records,
        n_classes=n_classes,
        block_size=block_size,
        n_resamples=n_resamples,
        coverage=coverage,
        bootstrap_seed=bootstrap_seed,
        fold_field=fold_field,
        label="without_macro_release",
    )
    delta = with_cell["macro_f1"]["row_pooled"] - without_cell["macro_f1"][
        "row_pooled"
    ]
    return {
        "with": with_cell,
        "without": without_cell,
        "delta_macro_f1_row_pooled": delta,
    }


# ---------------------------------------------------------------------------
# Combined surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HonestHeadlineReport:
    """All four reporting variants for a single canonical cell.

    The §6.7 honest-headline protocol publishes every macro-F1 cell
    with these four variants attached so a reader can see the FOMC-only
    thesis number, the mixed-pool comparator, the fold-4 with/without
    pair, and the macro-release with/without pair without re-running
    the sweep.
    """

    mixed_pool: dict[str, Any]
    fomc_only: dict[str, Any]
    fold_4: dict[str, dict[str, Any]]
    macro_release: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mixed_pool": self.mixed_pool,
            "fomc_only": self.fomc_only,
            "fold_4_with_without": self.fold_4,
            "macro_release_with_without": self.macro_release,
        }


def four_variant_report(
    records: Sequence[Record],
    *,
    drop_fold_id: str = "wf_fold_4",
    source_type_field: str = "source_type",
    fomc_source_types: Iterable[str] = _FOMC_SOURCE_TYPES,
    is_macro_release_field: str = "is_macro_release",
    fold_field: str = "fold_id",
    n_classes: int = 3,
    block_size: int = 20,
    n_resamples: int = 1000,
    coverage: float = 0.95,
    bootstrap_seed: int = 11,
) -> HonestHeadlineReport:
    """Compute all four reporting variants for one canonical cell."""

    return HonestHeadlineReport(
        mixed_pool=mixed_pool_macro_f1(
            records,
            n_classes=n_classes,
            block_size=block_size,
            n_resamples=n_resamples,
            coverage=coverage,
            bootstrap_seed=bootstrap_seed,
            fold_field=fold_field,
        ),
        fomc_only=fomc_only_macro_f1(
            records,
            source_type_field=source_type_field,
            fomc_source_types=fomc_source_types,
            n_classes=n_classes,
            block_size=block_size,
            n_resamples=n_resamples,
            coverage=coverage,
            bootstrap_seed=bootstrap_seed,
            fold_field=fold_field,
        ),
        fold_4=with_without_fold(
            records,
            drop_fold_id=drop_fold_id,
            fold_field=fold_field,
            n_classes=n_classes,
            block_size=block_size,
            n_resamples=n_resamples,
            coverage=coverage,
            bootstrap_seed=bootstrap_seed,
        ),
        macro_release=with_without_macro_release(
            records,
            is_macro_release_field=is_macro_release_field,
            fold_field=fold_field,
            n_classes=n_classes,
            block_size=block_size,
            n_resamples=n_resamples,
            coverage=coverage,
            bootstrap_seed=bootstrap_seed,
        ),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _report_cell(  # noqa: PLR0913
    records: Sequence[Record],
    *,
    n_classes: int,
    block_size: int,
    n_resamples: int,
    coverage: float,
    bootstrap_seed: int,
    fold_field: str,
    label: str,
) -> dict[str, Any]:
    """Build one variant's report dict: macro-F1 (both poolings),
    per-class P/R/F1, and a block-bootstrap CI on the row-pooled cell.
    """

    preds, targets, fold_ids = _extract_arrays(records, fold_field=fold_field)
    breakdown = compute_classification_breakdown(
        predictions=preds,
        targets=targets,
        n_classes=n_classes,
    )
    mean_of_fold_means = _mean_of_fold_means(
        preds, targets, fold_ids, n_classes=n_classes
    )
    ci = _bootstrap_macro_f1_ci(
        preds,
        targets,
        n_classes=n_classes,
        block_size=block_size,
        n_resamples=n_resamples,
        coverage=coverage,
        seed=bootstrap_seed,
    )

    return {
        "label": label,
        "support": len(preds),
        "n_classes": n_classes,
        "macro_f1": {
            "row_pooled": float(breakdown.macro_f1),
            "mean_of_fold_means": mean_of_fold_means,
            "canonical": "mean_of_fold_means",
        },
        "per_class": [m.to_dict() for m in breakdown.per_class],
        "confusion_matrix": [list(row) for row in breakdown.confusion_matrix],
        "ci": {
            "point": ci["point"],
            "lo": ci["lo"],
            "hi": ci["hi"],
            "coverage": coverage,
            "n_resamples": n_resamples,
            "block_size": block_size,
            "statistic": "macro_f1_row_pooled",
        },
        "folds_present": sorted(set(fold_ids)) if fold_ids else [],
    }


def _extract_arrays(
    records: Sequence[Record],
    *,
    fold_field: str,
) -> tuple[list[int], list[int], list[str]]:
    preds: list[int] = []
    targets: list[int] = []
    folds: list[str] = []
    for r in records:
        if "prediction" not in r or "target" not in r:
            continue
        preds.append(int(r["prediction"]))
        targets.append(int(r["target"]))
        folds.append(str(r.get(fold_field, "")))
    return preds, targets, folds


def _mean_of_fold_means(
    preds: Sequence[int],
    targets: Sequence[int],
    folds: Sequence[str],
    *,
    n_classes: int,
) -> float | None:
    """Per-fold macro-F1, then unweighted mean across folds.

    Returns ``None`` if no fold label is populated on any record — the
    statistic is undefined without a fold tag, and the caller should
    fall back to the row-pooled cell.
    """

    if not folds or all(not f for f in folds):
        return None
    groups: dict[str, tuple[list[int], list[int]]] = {}
    for p, t, f in zip(preds, targets, folds):
        if not f:
            continue
        bucket = groups.setdefault(f, ([], []))
        bucket[0].append(p)
        bucket[1].append(t)
    if not groups:
        return None
    per_fold: list[float] = []
    for _f, (gp, gt) in groups.items():
        breakdown = compute_classification_breakdown(
            predictions=gp,
            targets=gt,
            n_classes=n_classes,
        )
        per_fold.append(float(breakdown.macro_f1))
    if not per_fold:
        return None
    return sum(per_fold) / len(per_fold)


def _bootstrap_macro_f1_ci(
    preds: Sequence[int],
    targets: Sequence[int],
    *,
    n_classes: int,
    block_size: int,
    n_resamples: int,
    coverage: float,
    seed: int,
) -> dict[str, float]:
    """1k-block-bootstrap CI on the row-pooled macro-F1.

    Mirrors the resampling convention in
    :mod:`app.evaluation.regime_pooled_aggregator` — moving blocks of
    length ``block_size``, ⌈n/block_size⌉ blocks per resample, trimmed
    to ``n`` rows. Macro-F1 is recomputed on each resample.
    """

    n = len(preds)
    if n == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}
    point = float(
        compute_classification_breakdown(
            predictions=preds,
            targets=targets,
            n_classes=n_classes,
        ).macro_f1
    )
    rng = random.Random(seed)
    block = max(1, min(block_size, n))
    n_blocks = max(1, math.ceil(n / block))
    samples: list[float] = []
    for _ in range(n_resamples):
        idx: list[int] = []
        for _ in range(n_blocks):
            start = rng.randint(0, max(0, n - block))
            idx.extend(range(start, min(n, start + block)))
        idx = idx[:n]
        rp = [preds[i] for i in idx]
        rt = [targets[i] for i in idx]
        samples.append(
            float(
                compute_classification_breakdown(
                    predictions=rp,
                    targets=rt,
                    n_classes=n_classes,
                ).macro_f1
            )
        )
    samples.sort()
    alpha = (1.0 - coverage) / 2.0
    lo_idx = max(0, min(n_resamples - 1, int(alpha * n_resamples)))
    hi_idx = max(0, min(n_resamples - 1, int((1.0 - alpha) * n_resamples) - 1))
    return {
        "point": point,
        "lo": samples[lo_idx],
        "hi": samples[hi_idx],
    }


__all__ = [
    "HonestHeadlineReport",
    "Record",
    "four_variant_report",
    "fomc_only_macro_f1",
    "mixed_pool_macro_f1",
    "with_without_fold",
    "with_without_macro_release",
]


def _breakdown_to_per_class_summary(  # noqa: D401
    breakdown: ClassificationBreakdown,
) -> list[dict[str, float | int | None]]:
    """Convenience accessor for downstream serialisers (e.g. the wiki
    table renderer) that need just the per-class P/R/F1/support block.
    """

    return [m.to_dict() for m in breakdown.per_class]
