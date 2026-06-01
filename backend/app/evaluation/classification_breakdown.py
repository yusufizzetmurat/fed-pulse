"""Per-class breakdown for the vol-regime classifier (#199).

The training-loop classification evaluator at
``app.training.loop._evaluate_model`` reports a single accuracy, macro-F1
and cross-entropy loss per partition. That headline number does not say
*which* class the model handles well, which one it confuses, or whether
the class probabilities are calibrated.

This module supplies the per-class breakdown the appendix + UI need:

- 3x3 confusion matrix
- per-class precision / recall / F1 / support
- one-vs-rest ROC-AUC + PR-AUC

The helpers are pure Python (no sklearn dependency) so they ship through
the same CI gates as the rest of the eval surface; the math is small
enough that an explicit implementation is easier to audit than the
sklearn equivalent.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any


@dataclasses.dataclass(frozen=True)
class PerClassMetrics:
    """Precision / recall / F1 / support for a single class index."""

    class_id: int
    precision: float
    recall: float
    f1: float
    support: int
    roc_auc: float | None = None
    pr_auc: float | None = None

    def to_dict(self) -> dict[str, float | int | None]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ClassificationBreakdown:
    """Full per-class breakdown over a (predictions, targets) pair.

    ``confusion_matrix`` is ``[n_classes][n_classes]`` with
    ``rows = true class, columns = predicted class``. ``per_class`` is a
    tuple of ``PerClassMetrics`` indexed by class id.

    Cell coverage:
    - macro_precision / macro_recall / macro_f1 are unweighted means over
      classes that have at least one ground-truth occurrence (support>0).
    - weighted_f1 weights per-class F1 by support.
    - macro_roc_auc / macro_pr_auc are unweighted means over the
      per-class one-vs-rest curves that could be computed (i.e. that had
      both a positive and a negative example in the partition).
    """

    n_classes: int
    confusion_matrix: tuple[tuple[int, ...], ...]
    per_class: tuple[PerClassMetrics, ...]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    macro_roc_auc: float | None = None
    macro_pr_auc: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_classes": self.n_classes,
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "per_class": [m.to_dict() for m in self.per_class],
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "macro_roc_auc": self.macro_roc_auc,
            "macro_pr_auc": self.macro_pr_auc,
        }


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _binary_roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """One-vs-rest ROC-AUC via the Mann-Whitney U identity.

    Returns ``None`` when the partition has no positive or no negative
    example (degenerate ROC curve); avoids the division-by-zero that a
    naive trapezoid implementation would hit.
    """

    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if len(scores) < 2:
        return None
    pos_count = sum(1 for y in labels if y == 1)
    neg_count = len(labels) - pos_count
    if pos_count == 0 or neg_count == 0:
        return None
    # Rank-sum via average ranks to handle ties deterministically.
    indexed = sorted(enumerate(scores), key=lambda kv: kv[1])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based ranks; ties averaged
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    rank_sum_pos = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    # AUC = (R_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    auc = (rank_sum_pos - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)
    return float(auc)


def _binary_pr_auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """One-vs-rest PR-AUC via the trapezoidal-rule integral of P(R).

    Returns ``None`` when no positive example exists (recall is undefined).
    """

    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    pos_count = sum(1 for y in labels if y == 1)
    if pos_count == 0:
        return None
    if len(scores) == 0:
        return None
    # Sort by score descending; tie-break by label ascending so the
    # negative wins (rank stays conservative on ties).
    indexed = sorted(
        zip(scores, labels),
        key=lambda sl: (-float(sl[0]), int(sl[1])),
    )
    tp = 0
    fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    last_recall = 0.0
    last_precision = 1.0
    area = 0.0
    for _score, label in indexed:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, pos_count)
        # Trapezoidal step between last and current point.
        area += (recall - last_recall) * (precision + last_precision) / 2.0
        last_recall = recall
        last_precision = precision
        precisions.append(precision)
        recalls.append(recall)
    # If the last cumulative recall is < 1.0, close to (recall=1, p=0).
    if last_recall < 1.0:
        area += (1.0 - last_recall) * last_precision / 2.0
    return float(area)


def _confusion_matrix(
    predictions: Sequence[int],
    targets: Sequence[int],
    n_classes: int,
) -> tuple[tuple[int, ...], ...]:
    if len(predictions) != len(targets):
        raise ValueError(
            "predictions and targets must have the same length; "
            f"got {len(predictions)} vs {len(targets)}"
        )
    matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for true_y, pred_y in zip(targets, predictions):
        if not (0 <= true_y < n_classes and 0 <= pred_y < n_classes):
            continue
        matrix[int(true_y)][int(pred_y)] += 1
    return tuple(tuple(row) for row in matrix)


def compute_classification_breakdown(
    predictions: Sequence[int],
    targets: Sequence[int],
    *,
    n_classes: int,
    class_scores: Sequence[Sequence[float]] | None = None,
) -> ClassificationBreakdown:
    """Compute the full per-class breakdown for a partition's predictions.

    ``predictions`` carries the argmax class index per row;
    ``targets`` carries the ground-truth class index per row;
    ``class_scores`` (optional) carries the per-class softmax probability
    so the ROC / PR helpers can compute one-vs-rest AUCs. When omitted
    the AUC fields are ``None``.
    """

    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2; got {n_classes}")
    cm = _confusion_matrix(predictions, targets, n_classes=n_classes)
    per_class: list[PerClassMetrics] = []
    for c in range(n_classes):
        tp = cm[c][c]
        fn = sum(cm[c][k] for k in range(n_classes) if k != c)
        fp = sum(cm[r][c] for r in range(n_classes) if r != c)
        support = tp + fn
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = (
            _safe_div(2 * precision * recall, precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        roc_auc: float | None = None
        pr_auc: float | None = None
        if class_scores is not None:
            # One-vs-rest: positive label when target == c, score is the
            # class-c softmax probability.
            scores = [float(row[c]) for row in class_scores]
            ovr_labels = [1 if int(y) == c else 0 for y in targets]
            roc_auc = _binary_roc_auc(scores, ovr_labels)
            pr_auc = _binary_pr_auc(scores, ovr_labels)
        per_class.append(
            PerClassMetrics(
                class_id=c,
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                support=int(support),
                roc_auc=roc_auc,
                pr_auc=pr_auc,
            )
        )

    classes_with_support = [m for m in per_class if m.support > 0]
    if classes_with_support:
        macro_precision = sum(m.precision for m in classes_with_support) / len(classes_with_support)
        macro_recall = sum(m.recall for m in classes_with_support) / len(classes_with_support)
        macro_f1 = sum(m.f1 for m in classes_with_support) / len(classes_with_support)
        total_support = sum(m.support for m in classes_with_support)
        weighted_f1 = (
            sum(m.f1 * m.support for m in classes_with_support) / total_support
            if total_support > 0
            else 0.0
        )
    else:
        macro_precision = macro_recall = macro_f1 = weighted_f1 = 0.0

    roc_values = [m.roc_auc for m in per_class if m.roc_auc is not None]
    pr_values = [m.pr_auc for m in per_class if m.pr_auc is not None]
    macro_roc_auc = sum(roc_values) / len(roc_values) if roc_values else None
    macro_pr_auc = sum(pr_values) / len(pr_values) if pr_values else None

    return ClassificationBreakdown(
        n_classes=int(n_classes),
        confusion_matrix=cm,
        per_class=tuple(per_class),
        macro_precision=float(macro_precision),
        macro_recall=float(macro_recall),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        macro_roc_auc=macro_roc_auc,
        macro_pr_auc=macro_pr_auc,
    )


def render_confusion_matrix_text(
    breakdown: ClassificationBreakdown,
    *,
    class_labels: Sequence[str] | None = None,
) -> str:
    """Render the confusion matrix as a fixed-width text table for logs."""

    n = breakdown.n_classes
    labels = (
        list(class_labels)
        if class_labels is not None and len(class_labels) == n
        else [str(c) for c in range(n)]
    )
    column_width = max(6, *(len(label) for label in labels))
    header = (
        "true\\pred".ljust(column_width)
        + " "
        + " ".join(label.rjust(column_width) for label in labels)
    )
    rows = []
    for r, label in enumerate(labels):
        row_cells = " ".join(
            str(breakdown.confusion_matrix[r][c]).rjust(column_width) for c in range(n)
        )
        rows.append(label.ljust(column_width) + " " + row_cells)
    return "\n".join([header, *rows])


__all__ = [
    "PerClassMetrics",
    "ClassificationBreakdown",
    "compute_classification_breakdown",
    "render_confusion_matrix_text",
]
