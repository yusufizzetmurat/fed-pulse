"""Unit tests for the honest-headline reporting variants (#323).

The four helpers in :mod:`app.evaluation.reporting` each filter the
input record set differently and must produce distinct outputs when
the stratification is non-trivial. The tests in this file exercise
that contract on a synthetic stratified pool with known per-stratum
accuracy so the assertions stay calibrated against expected values.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.evaluation.reporting import (
    HonestHeadlineReport,
    fomc_only_macro_f1,
    four_variant_report,
    mixed_pool_macro_f1,
    with_without_fold,
    with_without_macro_release,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _record(
    prediction: int,
    target: int,
    *,
    fold_id: str = "wf_fold_1",
    source_type: str = "fomc_statement",
    is_macro_release: bool = False,
) -> dict[str, Any]:
    return {
        "prediction": prediction,
        "target": target,
        "fold_id": fold_id,
        "source_type": source_type,
        "is_macro_release": is_macro_release,
    }


@pytest.fixture
def stratified_pool() -> list[dict[str, Any]]:
    """Build a synthetic 3-class pool whose strata disagree.

    Layout (24 rows total):
    - 9 FOMC rows on folds 1/2/3 — model is correct on 8 of 9 (rotates
      `(0,0)(1,1)(2,2)` three times with one (0->1) miss on fold 3).
    - 6 cross-bank rows on fold 1 — model is correct on 2 of 6
      (deliberately worse so the FOMC-only variant lifts the headline).
    - 6 macro-release rows on fold 2 — model is correct on 5 of 6
      (`is_macro_release=True`; FOMC-only filter must drop them).
    - 3 fold-4 rows — class targets are all `1`/`2` (the R-17 zero-`calm`
      fixture); model predicts each correctly so the with-fold
      variant lifts the headline relative to without-fold.
    """

    rows: list[dict[str, Any]] = []
    # FOMC folds 1..3 — 8/9 correct.
    for fold in ("wf_fold_1", "wf_fold_2", "wf_fold_3"):
        rows.append(_record(0, 0, fold_id=fold))
        rows.append(_record(1, 1, fold_id=fold))
        rows.append(_record(2, 2, fold_id=fold))
    # Replace the last (2,2) on fold 3 with a (0->1) miss so the FOMC
    # subset is 8/9 = ~89% accurate, distinct from the cross-bank stratum.
    rows[-1] = _record(1, 0, fold_id="wf_fold_3")
    # Cross-bank rows on fold 1 — 2/6 correct.
    cross_bank_rows = [
        _record(0, 0, fold_id="wf_fold_1", source_type="cross_bank_ecb"),
        _record(1, 0, fold_id="wf_fold_1", source_type="cross_bank_ecb"),
        _record(2, 1, fold_id="wf_fold_1", source_type="cross_bank_ecb"),
        _record(0, 1, fold_id="wf_fold_1", source_type="cross_bank_boe"),
        _record(0, 2, fold_id="wf_fold_1", source_type="cross_bank_boe"),
        _record(2, 2, fold_id="wf_fold_1", source_type="cross_bank_boe"),
    ]
    rows.extend(cross_bank_rows)
    # Macro-release rows on fold 2 — 5/6 correct.
    macro_release_rows = [
        _record(
            0, 0, fold_id="wf_fold_2", source_type="cpi_release",
            is_macro_release=True,
        ),
        _record(
            1, 1, fold_id="wf_fold_2", source_type="cpi_release",
            is_macro_release=True,
        ),
        _record(
            2, 2, fold_id="wf_fold_2", source_type="cpi_release",
            is_macro_release=True,
        ),
        _record(
            0, 0, fold_id="wf_fold_2", source_type="nfp_release",
            is_macro_release=True,
        ),
        _record(
            1, 1, fold_id="wf_fold_2", source_type="nfp_release",
            is_macro_release=True,
        ),
        _record(
            0, 1, fold_id="wf_fold_2", source_type="nfp_release",
            is_macro_release=True,
        ),
    ]
    rows.extend(macro_release_rows)
    # Fold-4 rows — only class 1 + class 2 (no `calm`). All correct so
    # including fold-4 lifts the row-pooled macro-F1.
    fold4_rows = [
        _record(1, 1, fold_id="wf_fold_4"),
        _record(2, 2, fold_id="wf_fold_4"),
        _record(1, 1, fold_id="wf_fold_4"),
    ]
    rows.extend(fold4_rows)
    return rows


# ---------------------------------------------------------------------------
# mixed_pool_macro_f1
# ---------------------------------------------------------------------------


def test_mixed_pool_reports_both_poolings(stratified_pool: list[dict[str, Any]]) -> None:
    report = mixed_pool_macro_f1(stratified_pool, n_resamples=200)
    assert report["label"] == "mixed_pool"
    assert report["support"] == len(stratified_pool)
    assert "row_pooled" in report["macro_f1"]
    assert "mean_of_fold_means" in report["macro_f1"]
    assert report["macro_f1"]["canonical"] == "mean_of_fold_means"


def test_mixed_pool_emits_per_class_footnote(
    stratified_pool: list[dict[str, Any]],
) -> None:
    report = mixed_pool_macro_f1(stratified_pool, n_resamples=200)
    assert len(report["per_class"]) == 3
    for entry in report["per_class"]:
        assert {"class_id", "precision", "recall", "f1", "support"} <= set(
            entry.keys()
        )


def test_mixed_pool_emits_bootstrap_ci(stratified_pool: list[dict[str, Any]]) -> None:
    report = mixed_pool_macro_f1(
        stratified_pool, n_resamples=200, block_size=4
    )
    ci = report["ci"]
    assert ci["point"] >= 0.0
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["n_resamples"] == 200
    assert ci["block_size"] == 4


# ---------------------------------------------------------------------------
# fomc_only_macro_f1
# ---------------------------------------------------------------------------


def test_fomc_only_drops_cross_bank_and_macro_release(
    stratified_pool: list[dict[str, Any]],
) -> None:
    report = fomc_only_macro_f1(stratified_pool, n_resamples=200)
    fomc_count = sum(
        1
        for r in stratified_pool
        if str(r["source_type"]) in {"fomc_statement", "fomc_minutes"}
    )
    assert report["support"] == fomc_count
    assert report["label"] == "fomc_only"


def test_fomc_only_lifts_above_mixed_pool(
    stratified_pool: list[dict[str, Any]],
) -> None:
    """FOMC stratum is more accurate than cross-bank; the variant lifts."""

    mixed = mixed_pool_macro_f1(stratified_pool, n_resamples=200)
    fomc = fomc_only_macro_f1(stratified_pool, n_resamples=200)
    assert (
        fomc["macro_f1"]["row_pooled"] > mixed["macro_f1"]["row_pooled"]
    )


# ---------------------------------------------------------------------------
# with_without_fold
# ---------------------------------------------------------------------------


def test_with_without_fold_drops_named_fold(
    stratified_pool: list[dict[str, Any]],
) -> None:
    pair = with_without_fold(
        stratified_pool, drop_fold_id="wf_fold_4", n_resamples=200
    )
    assert pair["dropped_fold_id"] == "wf_fold_4"
    assert pair["with"]["support"] == len(stratified_pool)
    assert pair["without"]["support"] == sum(
        1 for r in stratified_pool if r["fold_id"] != "wf_fold_4"
    )


def test_with_without_fold_emits_distinct_macro_f1(
    stratified_pool: list[dict[str, Any]],
) -> None:
    pair = with_without_fold(
        stratified_pool, drop_fold_id="wf_fold_4", n_resamples=200
    )
    with_macro = pair["with"]["macro_f1"]["row_pooled"]
    without_macro = pair["without"]["macro_f1"]["row_pooled"]
    assert with_macro != pytest.approx(without_macro)
    assert pair["delta_macro_f1_row_pooled"] == pytest.approx(
        with_macro - without_macro
    )


# ---------------------------------------------------------------------------
# with_without_macro_release
# ---------------------------------------------------------------------------


def test_macro_release_with_without_drops_macro_rows(
    stratified_pool: list[dict[str, Any]],
) -> None:
    pair = with_without_macro_release(stratified_pool, n_resamples=200)
    assert pair["with"]["support"] == len(stratified_pool)
    assert pair["without"]["support"] == sum(
        1 for r in stratified_pool if not r["is_macro_release"]
    )


def test_macro_release_lift_attribution_distinct(
    stratified_pool: list[dict[str, Any]],
) -> None:
    """Macro-release rows are 5/6 correct vs the cross-bank 2/6; with
    the augmentation the row-pooled cell shifts."""

    pair = with_without_macro_release(stratified_pool, n_resamples=200)
    assert pair["with"]["macro_f1"]["row_pooled"] != pytest.approx(
        pair["without"]["macro_f1"]["row_pooled"]
    )


# ---------------------------------------------------------------------------
# Combined four-variant cell
# ---------------------------------------------------------------------------


def test_four_variant_report_returns_all_four_cells(
    stratified_pool: list[dict[str, Any]],
) -> None:
    report = four_variant_report(stratified_pool, n_resamples=200)
    assert isinstance(report, HonestHeadlineReport)
    blob = report.to_dict()
    for key in (
        "mixed_pool",
        "fomc_only",
        "fold_4_with_without",
        "macro_release_with_without",
    ):
        assert key in blob


def test_four_variants_distinct_on_non_trivial_input(
    stratified_pool: list[dict[str, Any]],
) -> None:
    """All four variants produce distinct headline numbers when the
    underlying stratification is non-trivial."""

    report = four_variant_report(stratified_pool, n_resamples=200)
    headlines = {
        "mixed": report.mixed_pool["macro_f1"]["row_pooled"],
        "fomc": report.fomc_only["macro_f1"]["row_pooled"],
        "with_fold4": report.fold_4["with"]["macro_f1"]["row_pooled"],
        "without_fold4": report.fold_4["without"]["macro_f1"]["row_pooled"],
        "with_macro": report.macro_release["with"]["macro_f1"]["row_pooled"],
        "without_macro": report.macro_release["without"]["macro_f1"]["row_pooled"],
    }
    distinct = {round(v, 6) for v in headlines.values()}
    # At least four distinct headline numbers across the six variant cells.
    assert len(distinct) >= 4, headlines


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_input_returns_nan_ci() -> None:
    report = mixed_pool_macro_f1([], n_resamples=10)
    assert report["support"] == 0
    assert report["ci"]["point"] != report["ci"]["point"]  # NaN


def test_records_missing_fold_field_skip_mean_of_fold_means() -> None:
    rows = [
        {"prediction": 0, "target": 0},
        {"prediction": 1, "target": 1},
        {"prediction": 2, "target": 2},
    ]
    report = mixed_pool_macro_f1(rows, n_resamples=50)
    assert report["macro_f1"]["mean_of_fold_means"] is None
    assert report["macro_f1"]["row_pooled"] == pytest.approx(1.0)


def test_fomc_filter_is_case_insensitive() -> None:
    rows = [
        _record(0, 0, source_type="FOMC_Statement"),
        _record(1, 1, source_type="fomc_minutes"),
        _record(2, 2, source_type="cross_bank_ecb"),
    ]
    report = fomc_only_macro_f1(rows, n_resamples=50)
    assert report["support"] == 2
