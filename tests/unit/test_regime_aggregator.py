from __future__ import annotations

from app.evaluation.regime_aggregator import REGIME_WINDOWS, aggregate_by_regime


def _holdout(fold_id: str, test_start: str, test_end: str, mean: float) -> dict:
    return {
        "fold_id": fold_id,
        "test_start": test_start,
        "test_end": test_end,
        "variants": {
            "baseline": {
                "combined_rmse": {"mean": mean, "std": 0.001, "count": 5, "per_seed": {}}
            }
        },
    }


def test_holdout_in_pre_2020_regime_only() -> None:
    rows = aggregate_by_regime([_holdout("wf_fold_1", "2018-01-01", "2018-06-30", 0.001)])
    regimes = {r.regime for r in rows}
    assert regimes == {"pre_2020_calm"}


def test_holdout_spanning_covid_and_hike_emits_both_rows() -> None:
    rows = aggregate_by_regime([_holdout("wf_fold_3", "2020-06-01", "2022-12-31", 0.005)])
    regimes = {r.regime for r in rows}
    assert "covid_shock" in regimes
    assert "hike_cycle" in regimes


def test_empty_holdouts_returns_empty() -> None:
    assert aggregate_by_regime([]) == []


def test_metric_keys_are_filtered() -> None:
    rows = aggregate_by_regime(
        [_holdout("wf_fold_1", "2015-01-01", "2015-06-30", 0.003)],
        metric_keys=("combined_rmse",),
    )
    metrics = {r.metric for r in rows}
    assert metrics == {"combined_rmse"}


def test_regime_windows_cover_2010_to_2023() -> None:
    starts = {w[1] for w in REGIME_WINDOWS}
    assert "2010-01-01" in starts
    assert "2020-01-01" in starts
    assert "2022-01-01" in starts


def _holdout_with_per_seed(
    fold_id: str, test_start: str, test_end: str, values: dict[str, float]
) -> dict:
    return {
        "fold_id": fold_id,
        "test_start": test_start,
        "test_end": test_end,
        "variants": {
            "baseline": {
                "combined_rmse": {
                    "mean": sum(values.values()) / max(1, len(values)),
                    "std": 0.0,
                    "count": len(values),
                    "per_seed": dict(values),
                }
            }
        },
    }


def test_row_carries_samples_from_per_seed_block() -> None:
    rows = aggregate_by_regime(
        [
            _holdout_with_per_seed(
                "wf_fold_1",
                "2018-01-01",
                "2018-06-30",
                {"11": 0.10, "29": 0.12, "47": 0.11, "71": 0.13, "97": 0.09},
            )
        ],
        metric_keys=("combined_rmse",),
    )
    assert len(rows) == 1
    row = rows[0]
    assert sorted(row.samples) == [0.09, 0.10, 0.11, 0.12, 0.13]
    assert row.ci_lo is not None and row.ci_hi is not None
    assert row.ci_lo <= row.mean <= row.ci_hi


def test_single_seed_emits_none_ci() -> None:
    rows = aggregate_by_regime(
        [
            _holdout_with_per_seed(
                "wf_fold_1",
                "2018-01-01",
                "2018-06-30",
                {"11": 0.10},
            )
        ],
        metric_keys=("combined_rmse",),
    )
    assert rows[0].samples == (0.10,)
    assert rows[0].ci_lo is None
    assert rows[0].ci_hi is None


def test_legacy_holdout_without_per_seed_keeps_running() -> None:
    rows = aggregate_by_regime([_holdout("wf_fold_1", "2018-01-01", "2018-06-30", 0.001)])
    # Older fixtures use ``per_seed: {}``; the aggregator should still emit
    # rows, just with empty samples and no CI.
    assert rows, "expected at least one row from the legacy holdout"
    assert all(r.samples == () for r in rows)
    assert all(r.ci_lo is None and r.ci_hi is None for r in rows)
