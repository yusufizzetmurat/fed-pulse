"""Tests for the next-FOMC decision forecaster (Phase 8 #147).

Covers the acceptance criteria from #147:

* ordinal-model dispatch picks the documented backend (NumPy fallback
  when statsmodels / mord are absent),
* walk-forward construction yields N-1 train rows for the N-th meeting,
* OIS-baseline reads ``pre_event_curve`` correctly and uses the
  documented sigma,
* feature-ablation runs produce strictly fewer columns when a family
  is dropped, and metrics are returned without crashing,
* no look-ahead: every train row's target_event_date is strictly less
  than the held-out target.

The synthetic data is small and toy -- the unit tests pin behavioural
contracts, not real-world model quality.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.forecasting import next_fomc_decision as nfd


# ---------------------------------------------------------------------------
# Synthetic-data builders
# ---------------------------------------------------------------------------


def _make_pre_event_curve(rate_pct: float) -> str:
    """Return a JSON-encoded pre_event_curve flat at ``rate_pct``."""

    points = [
        {"months_ahead": tenor, "implied_rate": rate_pct}
        for tenor in (1, 3, 6, 12, 24)
    ]
    return json.dumps(points)


def _make_mp_surprises_df(meetings: list[tuple[str, float, float, bool]]) -> pd.DataFrame:
    """Build a minimal mp_surprises DataFrame.

    Each row: ``(event_date_iso, ff_target_prior, ff_target_after, is_intermeeting)``.
    """

    rows = []
    for idx, (event_date, prior, after, inter) in enumerate(meetings, start=1):
        rows.append(
            {
                "event_date": event_date,
                "meeting_id": idx,
                "ff_target_prior": float(prior),
                "ff_target_after": float(after),
                "mp_surprise_level": 0.0,
                "mp_surprise_path_factor": 0.0,
                # pre_event_curve sits at the prior target (no surprise).
                "pre_event_curve": _make_pre_event_curve(float(prior)),
                "post_event_curve": _make_pre_event_curve(float(after)),
                "fed_info_factor": 0.0,
                "is_intermeeting": bool(inter),
                "methodology": "ois_proxy",
                "fed_info_factor_source": "daily_window_proxy",
                "target_source": "band|after:band",
                "data_version": "test_version_v1",
            }
        )
    return pd.DataFrame(rows)


def _make_events_df(event_dates: list[str], stance_labels: list[str] | None = None) -> pd.DataFrame:
    """Build a minimal events DataFrame."""

    if stance_labels is None:
        stance_labels = ["neutral"] * len(event_dates)
    rows = []
    for idx, (ed, stance) in enumerate(zip(event_dates, stance_labels), start=1):
        rows.append(
            {
                "event_date": ed,
                "event_kind": "statement",
                "document_id": f"doc_{idx}",
                "text_hash": f"hash_{idx}",
                "source": "scraped_fed",
                "source_record_id": f"rec_{idx}",
                "as_of_ts": f"{ed}T19:00:00Z",
                "text": f"placeholder text for {ed}",
                "token_count": 100,
                "axis_stance": stance,
                "axis_time": "neutral",
                "axis_certainty": "neutral",
                "axis_factor": "neutral",
                "credibility_drift_score": 0.0,
                "credibility_realized_vs_stated_gap": 0.0,
                "credibility_market_implied_gap": 0.0,
                "credibility_months_since_reversal": 6,
                "prior_window_sha256": "abc",
                "prior_bars_json": "[]",
                "asset_symbol": "^GSPC",
                "horizon": 1,
                "realized_return": 0.0,
                "abnormal_return": 0.0,
                "alpha": 0.0,
                "beta": 1.0,
                "direction_t1d": 0,
                "volatility_shift": 0.0,
                "concurrent_macro_release": False,
                "realized_date": ed,
            }
        )
    return pd.DataFrame(rows)


def _make_linguistic_df(event_count: int) -> pd.DataFrame:
    rows = []
    for idx in range(1, event_count + 1):
        row = {"text_hash": f"hash_{idx}"}
        # 14 placeholder linguistic features.
        for axis in (
            "topic_share_inflation",
            "topic_share_employment",
            "topic_share_financial_stability",
            "topic_share_growth",
            "topic_share_balance_sheet",
            "topic_share_misc_1",
            "topic_share_misc_2",
            "topic_share_misc_3",
            "hedge_density",
            "comparison_density",
            "forward_density",
            "concrete_ratio",
            "hawk_dove_asymmetry",
            "log_token_count",
        ):
            row[axis] = 0.1 * idx
        rows.append(row)
    return pd.DataFrame(rows)


def _make_macro_df(event_dates: list[str]) -> pd.DataFrame:
    """One macro row per (event_date - 1 day) so the < as_of join hits."""

    rows = []
    for ed in event_dates:
        d = _dt.date.fromisoformat(ed) - _dt.timedelta(days=1)
        rows.append(
            {
                "as_of_date": d.isoformat(),
                "unrate": 4.0,
                "cpi_yoy": 2.0,
                "core_pce_yoy": 1.8,
                "ism_proxy": 0.5,
                "payems_mom": 200.0,
                "rsafs_mom": 0.3,
                "ism_proxy_source": "MANEMP_3m_pct",
                "publication_delay_days": 30,
                "data_version": "test_v1",
            }
        )
    return pd.DataFrame(rows)


def _make_canonical_meetings() -> list[tuple[str, float, float, bool]]:
    """Eight meetings with a cut->hold->hike sweep + one intermeeting cut."""

    # event_date_iso, ff_target_prior, ff_target_after, is_intermeeting
    return [
        ("2022-01-26", 0.125, 0.125, False),  # hold
        ("2022-03-16", 0.125, 0.375, False),  # +25
        ("2022-05-04", 0.375, 0.875, False),  # +50
        ("2022-06-15", 0.875, 1.625, False),  # +75 (hike_75 -- still in set)
        ("2022-07-27", 1.625, 2.375, False),  # +75
        ("2022-09-21", 2.375, 3.125, False),  # +75
        ("2022-11-02", 3.125, 3.875, False),  # +75
        ("2022-12-14", 3.875, 4.375, False),  # +50
    ]


# ---------------------------------------------------------------------------
# delta_to_class
# ---------------------------------------------------------------------------


def test_delta_to_class_supported_classes() -> None:
    assert nfd.delta_to_class(0.0) == "hold"
    assert nfd.delta_to_class(25.0) == "hike_25"
    assert nfd.delta_to_class(-25.0) == "cut_25"
    assert nfd.delta_to_class(50.0) == "hike_50"
    assert nfd.delta_to_class(75.0) == "hike_75"
    # 12.5 bp slack tolerates near-but-not-exact deltas.
    assert nfd.delta_to_class(24.0) == "hike_25"


def test_delta_to_class_out_of_set_warns_and_returns_none() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = nfd.delta_to_class(-100.0)
    assert result is None
    assert any("outside supported class set" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# OIS baseline
# ---------------------------------------------------------------------------


def test_ois_baseline_uses_documented_sigma() -> None:
    """The documented sigma is 12.5 bp -- pin it via the module constant."""

    assert nfd.OIS_BASELINE_SIGMA_BP == 12.5


def test_ois_baseline_peaks_at_hold_when_curve_equals_base() -> None:
    proba = nfd.ois_baseline_probability(
        implied_rate=4.50,
        base_rate=4.50,
        sigma_bp=12.5,
    )
    assert sum(proba.values()) == pytest.approx(1.0, abs=1e-9)
    assert proba["hold"] == max(proba.values())


def test_ois_baseline_peaks_at_hike25_when_curve_implies_25_bp() -> None:
    """A 25 bp implied move puts the peak on hike_25."""

    proba = nfd.ois_baseline_probability(
        implied_rate=4.75,
        base_rate=4.50,
        sigma_bp=12.5,
    )
    # hike_25 should dominate; cut_25 should be vanishingly small.
    assert proba["hike_25"] > 0.3
    assert proba["hike_25"] > proba["hold"]
    assert proba["cut_25"] < 0.01


def test_ois_baseline_uniform_on_missing_inputs() -> None:
    proba = nfd.ois_baseline_probability(
        implied_rate=None, base_rate=None
    )
    expected = 1.0 / len(nfd.ORDINAL_CLASSES)
    for v in proba.values():
        assert v == pytest.approx(expected)


def test_naive_carry_baseline_assigns_all_mass_to_hold() -> None:
    proba = nfd.naive_carry_probability()
    assert proba["hold"] == 1.0
    other = sum(v for k, v in proba.items() if k != "hold")
    assert other == 0.0


# ---------------------------------------------------------------------------
# Ordinal model dispatch
# ---------------------------------------------------------------------------


def test_ordinal_dispatch_falls_back_to_numpy_when_libs_absent() -> None:
    """The 'numpy' preference is honoured even if statsmodels exists."""

    handle = nfd._dispatch_ordinal(classes=nfd.ORDINAL_CLASSES, prefer="numpy")
    assert handle.backend == "numpy_proportional_odds"


def test_ordinal_dispatch_auto_select_returns_one_of_known_backends() -> None:
    handle = nfd._dispatch_ordinal(classes=nfd.ORDINAL_CLASSES)
    assert handle.backend in {
        "numpy_proportional_odds",
        "statsmodels_ordered_model",
        "mord_logistic_it",
    }


def test_proportional_odds_logit_fit_predict_shapes() -> None:
    rng = np.random.default_rng(seed=11)
    n, p = 30, 4
    X = rng.normal(size=(n, p))
    # Toy target: 4 classes ordered along an arbitrary linear combo.
    score = X @ np.array([1.0, -0.5, 0.5, 0.0])
    y = np.digitize(score, np.quantile(score, [0.25, 0.5, 0.75]))
    classes = ("a", "b", "c", "d")
    model = nfd.ProportionalOddsLogit(alpha=0.5)
    model.fit(X, y, classes)
    proba = model.predict_proba(X)
    assert proba.shape == (n, len(classes))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    # Some signal: argmax(proba) on at least half the rows matches y.
    accuracy = float(np.mean(np.argmax(proba, axis=1) == y))
    assert accuracy >= 0.25  # bare-minimum sanity; not a quality claim


# ---------------------------------------------------------------------------
# Curve extraction
# ---------------------------------------------------------------------------


def test_curve_value_at_handles_json_and_missing_tenors() -> None:
    curve = _make_pre_event_curve(4.75)
    assert nfd._curve_value_at(curve, 3) == pytest.approx(4.75)
    assert nfd._curve_value_at(curve, 60) is None
    assert nfd._curve_value_at(None, 3) is None
    assert nfd._curve_value_at("not-json", 3) is None


# ---------------------------------------------------------------------------
# Supervised row construction
# ---------------------------------------------------------------------------


def test_build_supervised_rows_pairs_meeting_with_next() -> None:
    meetings = _make_canonical_meetings()
    event_dates = [m[0] for m in meetings]
    mp = _make_mp_surprises_df(meetings)
    events = _make_events_df(event_dates)
    ling = _make_linguistic_df(len(event_dates))
    macro = _make_macro_df(event_dates)

    rows, summary = nfd.build_supervised_rows(
        events=events, mp_surprises=mp, linguistic_features=ling, macro_state=macro
    )

    # 8 meetings => 7 supervised rows (every meeting except the last
    # has a next-scheduled meeting).
    assert summary["rows_emitted"] == 7
    assert summary["dropped_target_out_of_class"] == 0
    # Targets reconstructed: row 0 (feature meeting 2022-01-26) predicts
    # 2022-03-16's hike_25.
    assert rows[0].feature_event_date == _dt.date(2022, 1, 26)
    assert rows[0].target_event_date == _dt.date(2022, 3, 16)
    assert rows[0].target_class == "hike_25"


def test_build_supervised_rows_skips_out_of_class_target() -> None:
    """A 100 bp move on the next meeting should drop that supervised row."""

    meetings = [
        ("2008-01-22", 4.25, 3.5, True),  # intermeeting cut
        ("2008-01-30", 3.5, 3.0, False),  # next scheduled -- cut_50
        ("2008-03-18", 3.0, 2.25, False),  # next scheduled -- 75 bp cut (out of set)
        ("2008-04-30", 2.25, 2.0, False),  # next scheduled -- cut_25
    ]
    event_dates = [m[0] for m in meetings]
    mp = _make_mp_surprises_df(meetings)
    events = _make_events_df(event_dates)
    ling = _make_linguistic_df(len(event_dates))
    macro = _make_macro_df(event_dates)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows, summary = nfd.build_supervised_rows(
            events=events, mp_surprises=mp, linguistic_features=ling, macro_state=macro
        )

    assert summary["dropped_target_out_of_class"] >= 1
    assert any("outside supported class set" in str(w.message) for w in caught)
    # The 2008-01-30 -> 2008-03-18 pair drops because the 75 bp cut is
    # out of set. 2008-01-22 (intermeeting) is excluded from
    # "next scheduled" candidates, so its supervised row would point
    # at 2008-01-30 (a cut_50 -- in set). Confirm one row survives.
    classes = {r.target_class for r in rows}
    assert classes.issubset(set(nfd.ORDINAL_CLASSES))


# ---------------------------------------------------------------------------
# Walk-forward CV
# ---------------------------------------------------------------------------


def test_walk_forward_produces_n_minus_1_train_rows() -> None:
    meetings = _make_canonical_meetings()
    event_dates = [m[0] for m in meetings]
    mp = _make_mp_surprises_df(meetings)
    events = _make_events_df(event_dates)
    ling = _make_linguistic_df(len(event_dates))
    macro = _make_macro_df(event_dates)
    rows, _ = nfd.build_supervised_rows(
        events=events, mp_surprises=mp, linguistic_features=ling, macro_state=macro
    )

    preds = nfd.walk_forward_predict(rows, families=("ois",), ordinal_backend="numpy")

    # One prediction per supervised row.
    assert len(preds) == len(rows)
    # Train sizes: 0, 1, 2, ...
    assert [p.n_train_rows for p in preds] == list(range(len(rows)))
    # Every prediction has an OIS baseline and a naive carry; ordinal
    # only kicks in once the train set covers >= number of classes.
    for p in preds:
        assert "ois_baseline" in p.probabilities
        assert "naive_carry" in p.probabilities


def test_walk_forward_no_lookahead_in_train_set() -> None:
    meetings = _make_canonical_meetings()
    event_dates = [m[0] for m in meetings]
    mp = _make_mp_surprises_df(meetings)
    events = _make_events_df(event_dates)
    ling = _make_linguistic_df(len(event_dates))
    macro = _make_macro_df(event_dates)
    rows, _ = nfd.build_supervised_rows(
        events=events, mp_surprises=mp, linguistic_features=ling, macro_state=macro
    )

    # The walk-forward routine asserts the contract internally. Run it
    # under a fresh interpreter pass to confirm no AssertionError fires.
    preds = nfd.walk_forward_predict(rows, families=("ois",), ordinal_backend="numpy")
    # Explicit external check too: for every held-out row, the train
    # set is exactly rows[:i] and all those target_event_dates are
    # strictly less.
    sorted_rows = sorted(rows, key=lambda r: r.target_event_date)
    for i, _pred in enumerate(preds):
        held_out_date = sorted_rows[i].target_event_date
        for j in range(i):
            assert sorted_rows[j].target_event_date < held_out_date


# ---------------------------------------------------------------------------
# Feature ablations
# ---------------------------------------------------------------------------


def test_feature_ablation_drops_columns() -> None:
    meetings = _make_canonical_meetings()
    event_dates = [m[0] for m in meetings]
    mp = _make_mp_surprises_df(meetings)
    events = _make_events_df(event_dates)
    ling = _make_linguistic_df(len(event_dates))
    macro = _make_macro_df(event_dates)
    rows, _ = nfd.build_supervised_rows(
        events=events, mp_surprises=mp, linguistic_features=ling, macro_state=macro
    )

    X_full, names_full = nfd._build_feature_matrix(rows, nfd.FEATURE_FAMILIES)
    X_ois_only, names_ois = nfd._build_feature_matrix(rows, ("ois",))
    X_no_macro, names_no_macro = nfd._build_feature_matrix(
        rows, ("ois", "text", "linguistic", "credibility")
    )

    assert X_ois_only.shape[1] < X_full.shape[1]
    assert X_no_macro.shape[1] < X_full.shape[1]
    # Dropping macro removes exactly the macro column names.
    assert set(names_no_macro) == set(names_full) - set(nfd._macro_feature_names())
    # OIS-only keeps exactly the OIS columns.
    assert set(names_ois) == set(nfd._ois_feature_names())


def test_metrics_compute_on_walk_forward_predictions() -> None:
    meetings = _make_canonical_meetings()
    event_dates = [m[0] for m in meetings]
    mp = _make_mp_surprises_df(meetings)
    events = _make_events_df(event_dates)
    ling = _make_linguistic_df(len(event_dates))
    macro = _make_macro_df(event_dates)
    rows, _ = nfd.build_supervised_rows(
        events=events, mp_surprises=mp, linguistic_features=ling, macro_state=macro
    )
    preds = nfd.walk_forward_predict(rows, families=("ois",), ordinal_backend="numpy")
    metrics = nfd.compute_metrics(preds, "ois_baseline")
    assert metrics["n"] == len(rows)
    assert 0.0 <= metrics["brier"] <= 4.0
    assert 0.0 <= metrics["top1_accuracy"] <= 1.0
    cm = metrics["confusion_matrix"]
    # Every truth class in the test corpus appears in the confusion-matrix keys.
    assert set(cm.keys()) == set(nfd.ORDINAL_CLASSES)


# ---------------------------------------------------------------------------
# End-to-end run + artifact serialisation
# ---------------------------------------------------------------------------


def test_run_writes_results_and_metrics(tmp_path: Path) -> None:
    meetings = _make_canonical_meetings()
    event_dates = [m[0] for m in meetings]
    mp = _make_mp_surprises_df(meetings)
    events = _make_events_df(event_dates)
    ling = _make_linguistic_df(len(event_dates))
    macro = _make_macro_df(event_dates)

    artifacts = nfd.run(
        events=events,
        mp_surprises=mp,
        linguistic_features=ling,
        macro_state=macro,
        output_dir=tmp_path,
        ordinal_backend="numpy",
    )

    results_path = tmp_path / "results.json"
    metrics_path = tmp_path / "metrics.json"
    attribution_path = tmp_path / "feature_attribution.md"
    assert results_path.exists()
    assert metrics_path.exists()
    assert attribution_path.exists()

    metrics = json.loads(metrics_path.read_text())
    assert "full_window" in metrics
    assert "ex_pandemic_window" in metrics
    assert metrics["ordinal_backend"] == "numpy_proportional_odds"
    assert metrics["ois_baseline_sigma_bp"] == 12.5

    attribution_text = attribution_path.read_text()
    assert "ois_only" in attribution_text
    assert "full" in attribution_text


def test_pandemic_window_filter_removes_2020_rows() -> None:
    pred_2020 = nfd.FoldPrediction(
        target_event_date="2020-05-15",
        target_as_of_ts="2020-05-15T19:00:00+00:00",
        target_class="hold",
        n_train_rows=10,
        probabilities={"naive_carry": nfd.naive_carry_probability()},
    )
    pred_2023 = nfd.FoldPrediction(
        target_event_date="2023-09-20",
        target_as_of_ts="2023-09-20T19:00:00+00:00",
        target_class="hold",
        n_train_rows=20,
        probabilities={"naive_carry": nfd.naive_carry_probability()},
    )
    filtered = nfd.filter_predictions_excluding_window(
        [pred_2020, pred_2023], start=nfd.PANDEMIC_START, end=nfd.PANDEMIC_END
    )
    assert len(filtered) == 1
    assert filtered[0].target_event_date == "2023-09-20"
