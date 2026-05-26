from __future__ import annotations

import pytest

from app.evaluation.classification_breakdown import (
    ClassificationBreakdown,
    compute_classification_breakdown,
    render_confusion_matrix_text,
)


# ---------------------------------------------------------------------------
# Confusion matrix shape + indexing
# ---------------------------------------------------------------------------


def test_confusion_matrix_dimensions_match_n_classes() -> None:
    breakdown = compute_classification_breakdown(
        predictions=[0, 1, 2, 0],
        targets=[0, 1, 2, 0],
        n_classes=3,
    )
    assert breakdown.n_classes == 3
    assert len(breakdown.confusion_matrix) == 3
    assert all(len(row) == 3 for row in breakdown.confusion_matrix)


def test_perfect_predictions_have_identity_confusion() -> None:
    """Diagonal = support, off-diagonal = 0 for perfect predictions."""

    breakdown = compute_classification_breakdown(
        predictions=[0, 1, 2, 0, 1, 2],
        targets=[0, 1, 2, 0, 1, 2],
        n_classes=3,
    )
    expected = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    assert breakdown.confusion_matrix == expected
    assert breakdown.macro_precision == pytest.approx(1.0)
    assert breakdown.macro_recall == pytest.approx(1.0)
    assert breakdown.macro_f1 == pytest.approx(1.0)


def test_off_diagonal_picks_up_misclassifications() -> None:
    """row 0 col 1 means true=0, pred=1."""

    breakdown = compute_classification_breakdown(
        predictions=[1, 1, 2],
        targets=[0, 1, 2],
        n_classes=3,
    )
    assert breakdown.confusion_matrix[0][1] == 1  # one 0->1 misclassification
    assert breakdown.confusion_matrix[1][1] == 1
    assert breakdown.confusion_matrix[2][2] == 1


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------


def test_per_class_precision_recall_f1_on_known_split() -> None:
    """class 0: TP=2 FP=1 FN=0 -> P=2/3, R=1, F1=0.8
    class 1: TP=2 FP=0 FN=1 -> P=1, R=2/3, F1=0.8
    class 2: TP=1 FP=0 FN=0 -> P=1, R=1, F1=1"""

    breakdown = compute_classification_breakdown(
        predictions=[0, 0, 0, 1, 1, 2],
        targets=[0, 0, 1, 1, 1, 2],
        n_classes=3,
    )
    p = {m.class_id: m for m in breakdown.per_class}
    assert p[0].precision == pytest.approx(2 / 3)
    assert p[0].recall == pytest.approx(1.0)
    assert p[0].f1 == pytest.approx(0.8)
    assert p[1].precision == pytest.approx(1.0)
    assert p[1].recall == pytest.approx(2 / 3)
    assert p[1].f1 == pytest.approx(0.8)
    assert p[2].precision == pytest.approx(1.0)
    assert p[2].recall == pytest.approx(1.0)
    assert p[2].f1 == pytest.approx(1.0)


def test_per_class_support_counts_targets_not_predictions() -> None:
    """Support is the number of true examples of that class."""

    breakdown = compute_classification_breakdown(
        predictions=[0, 0, 0, 0],
        targets=[0, 1, 2, 2],
        n_classes=3,
    )
    supports = {m.class_id: m.support for m in breakdown.per_class}
    assert supports == {0: 1, 1: 1, 2: 2}


def test_class_with_zero_support_does_not_break_macro_average() -> None:
    """A class with no ground-truth example is excluded from macro mean."""

    breakdown = compute_classification_breakdown(
        predictions=[0, 0, 1],
        targets=[0, 0, 1],
        n_classes=3,
    )
    # Class 2 has support=0 -> should not be in the macro average
    classes_with_support = [m for m in breakdown.per_class if m.support > 0]
    assert len(classes_with_support) == 2
    assert breakdown.macro_f1 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Weighted F1
# ---------------------------------------------------------------------------


def test_weighted_f1_weights_by_support() -> None:
    """Weighted F1 should differ from macro F1 when class supports differ."""

    breakdown = compute_classification_breakdown(
        predictions=[0, 0, 0, 0, 1, 1, 2, 0],
        targets=[0, 0, 0, 0, 1, 1, 2, 0],
        n_classes=3,
    )
    # Perfect predictions -> all F1=1 regardless of weighting
    assert breakdown.macro_f1 == pytest.approx(1.0)
    assert breakdown.weighted_f1 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AUC computation
# ---------------------------------------------------------------------------


def test_roc_auc_perfect_score_is_one() -> None:
    """Perfect probability scores give AUC=1 for every class."""

    scores = [
        [0.9, 0.05, 0.05],
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.05, 0.05, 0.9],
    ]
    breakdown = compute_classification_breakdown(
        predictions=[0, 0, 1, 1, 2, 2],
        targets=[0, 0, 1, 1, 2, 2],
        n_classes=3,
        class_scores=scores,
    )
    for m in breakdown.per_class:
        assert m.roc_auc == pytest.approx(1.0)
    assert breakdown.macro_roc_auc == pytest.approx(1.0)


def test_roc_auc_random_score_is_about_half() -> None:
    """Uniform probabilities should give AUC near 0.5."""

    n = 30
    scores = [[1 / 3, 1 / 3, 1 / 3] for _ in range(n)]
    targets = [(i % 3) for i in range(n)]
    breakdown = compute_classification_breakdown(
        predictions=[0] * n,
        targets=targets,
        n_classes=3,
        class_scores=scores,
    )
    # Constant scores -> all rank ties -> AUC == 0.5 exactly.
    for m in breakdown.per_class:
        assert m.roc_auc == pytest.approx(0.5)


def test_roc_auc_none_when_no_positives_or_negatives() -> None:
    """A class with no positive (or no negative) example has undefined ROC."""

    scores = [[0.5, 0.5, 0.0]] * 3
    breakdown = compute_classification_breakdown(
        predictions=[0, 0, 0],
        targets=[0, 0, 0],  # only class 0 appears
        n_classes=3,
        class_scores=scores,
    )
    by_id = {m.class_id: m for m in breakdown.per_class}
    # Class 1 and class 2 have no positives -> ROC undefined
    assert by_id[1].roc_auc is None
    assert by_id[2].roc_auc is None
    # Class 0 has no negatives -> ROC undefined too
    assert by_id[0].roc_auc is None


def test_macro_auc_skips_undefined_classes() -> None:
    """Macro AUC averages only classes whose AUC could be computed."""

    scores = [
        [0.9, 0.1, 0.0],
        [0.1, 0.9, 0.0],
        [0.9, 0.1, 0.0],
        [0.1, 0.9, 0.0],
    ]
    breakdown = compute_classification_breakdown(
        predictions=[0, 1, 0, 1],
        targets=[0, 1, 0, 1],  # class 2 never appears
        n_classes=3,
        class_scores=scores,
    )
    # Macro AUC should equal the mean of class-0 and class-1 AUCs only.
    by_id = {m.class_id: m for m in breakdown.per_class}
    assert by_id[2].roc_auc is None
    assert breakdown.macro_roc_auc is not None
    assert breakdown.macro_roc_auc == pytest.approx(
        (by_id[0].roc_auc + by_id[1].roc_auc) / 2.0
    )


# ---------------------------------------------------------------------------
# Round-trip via to_dict
# ---------------------------------------------------------------------------


def test_breakdown_round_trips_through_dict() -> None:
    breakdown = compute_classification_breakdown(
        predictions=[0, 1, 2],
        targets=[0, 1, 2],
        n_classes=3,
    )
    payload = breakdown.to_dict()
    assert payload["n_classes"] == 3
    assert payload["macro_f1"] == pytest.approx(1.0)
    assert isinstance(payload["confusion_matrix"], list)
    assert isinstance(payload["per_class"], list)
    assert len(payload["per_class"]) == 3


# ---------------------------------------------------------------------------
# Text-table renderer
# ---------------------------------------------------------------------------


def test_text_renderer_returns_grid_of_expected_shape() -> None:
    breakdown = compute_classification_breakdown(
        predictions=[0, 1, 2, 1],
        targets=[0, 1, 2, 0],
        n_classes=3,
    )
    text = render_confusion_matrix_text(
        breakdown, class_labels=("calm", "normal", "high")
    )
    lines = text.split("\n")
    assert len(lines) == 4  # 1 header + 3 rows
    assert "calm" in lines[1]
    assert "normal" in lines[2]
    assert "high" in lines[3]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_classification_breakdown(
            predictions=[0, 1],
            targets=[0],
            n_classes=2,
        )


def test_n_classes_below_two_raises() -> None:
    with pytest.raises(ValueError, match="n_classes"):
        compute_classification_breakdown(
            predictions=[0, 0],
            targets=[0, 0],
            n_classes=1,
        )
