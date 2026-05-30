"""Tests for the cross-asset response head (Phase 8 #148).

Covers the acceptance criteria from #148:

* walk-forward construction yields ``N-1`` train rows for the N-th
  meeting *within a single cell*, with strict no-look-ahead asserted,
* per-cell ridge recovers a synthetic linear target,
* feature-family ablation produces strictly fewer columns when a
  family is dropped,
* target reconstruction reads ``abnormal_return`` from
  ``events.parquet`` and silently ignores rows where the asset's
  target is missing,
* the pandemic-window filter excludes the documented date range.

Synthetic data is intentionally small; the tests pin behavioural
contracts, not real-world model quality.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.forecasting import cross_asset_response as car


# ---------------------------------------------------------------------------
# Synthetic-data builders
# ---------------------------------------------------------------------------


def _make_pre_event_curve(rate_pct: float) -> str:
    points = [
        {"months_ahead": tenor, "implied_rate": rate_pct}
        for tenor in (1, 3, 6, 12, 24)
    ]
    return json.dumps(points)


def _make_mp_surprises_df(
    meetings: list[tuple[str, float, float]],
) -> pd.DataFrame:
    """``(event_date_iso, ff_target_prior, ff_target_after)`` rows."""

    rows = []
    for idx, (event_date, prior, after) in enumerate(meetings, start=1):
        rows.append(
            {
                "event_date": event_date,
                "meeting_id": idx,
                "ff_target_prior": float(prior),
                "ff_target_after": float(after),
                "mp_surprise_level": 0.0,
                "mp_surprise_path_factor": 0.0,
                "pre_event_curve": _make_pre_event_curve(float(prior)),
                "post_event_curve": _make_pre_event_curve(float(after)),
                "fed_info_factor": 0.0,
                "is_intermeeting": False,
                "methodology": "ois_proxy",
                "fed_info_factor_source": "daily_window_proxy",
                "target_source": "band|after:band",
                "data_version": "test_version_v1",
            }
        )
    return pd.DataFrame(rows)


def _make_events_df(
    *,
    event_dates: list[str],
    assets: list[str],
    horizons: list[int],
    abnormal_overrides: dict[tuple[str, str, int], float] | None = None,
) -> pd.DataFrame:
    """Build a minimal events frame with one row per ``(date, asset, horizon)``."""

    overrides = abnormal_overrides or {}
    rows: list[dict] = []
    for idx, ed in enumerate(event_dates, start=1):
        for asset in assets:
            for h in horizons:
                key = (ed, asset, h)
                rows.append(
                    {
                        "event_date": ed,
                        "event_kind": "statement",
                        "document_id": f"doc_{idx}_{asset}_{h}",
                        "text_hash": f"hash_{idx}",
                        "source": "scraped_fed",
                        "source_record_id": f"rec_{idx}_{asset}_{h}",
                        "as_of_ts": f"{ed}T19:00:00Z",
                        "text": f"placeholder text for {ed}",
                        "token_count": 100,
                        "axis_stance": "neutral",
                        "axis_time": "neutral",
                        "axis_certainty": "neutral",
                        "axis_factor": "neutral",
                        "credibility_drift_score": 0.0,
                        "credibility_realized_vs_stated_gap": 0.0,
                        "credibility_market_implied_gap": 0.0,
                        "credibility_months_since_reversal": 6,
                        "prior_window_sha256": "abc",
                        "prior_bars_json": "[]",
                        "asset_symbol": asset,
                        "horizon": h,
                        "realized_return": 0.0,
                        "abnormal_return": overrides.get(key, 0.0),
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
        row: dict = {"text_hash": f"hash_{idx}"}
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


def _make_canonical_meetings() -> list[tuple[str, float, float]]:
    """Twelve meetings stepping through the 2022-2023 hiking cycle."""

    return [
        ("2022-01-26", 0.125, 0.125),
        ("2022-03-16", 0.125, 0.375),
        ("2022-05-04", 0.375, 0.875),
        ("2022-06-15", 0.875, 1.625),
        ("2022-07-27", 1.625, 2.375),
        ("2022-09-21", 2.375, 3.125),
        ("2022-11-02", 3.125, 3.875),
        ("2022-12-14", 3.875, 4.375),
        ("2023-02-01", 4.375, 4.625),
        ("2023-03-22", 4.625, 4.875),
        ("2023-05-03", 4.875, 5.125),
        ("2023-06-14", 5.125, 5.125),
    ]


# ---------------------------------------------------------------------------
# Supervised row construction
# ---------------------------------------------------------------------------


def test_build_supervised_rows_fans_out_per_asset_per_horizon() -> None:
    meetings = _make_canonical_meetings()
    event_dates = [m[0] for m in meetings]
    assets = ["^GSPC", "^TNX", "XLF"]
    horizons = [1, 5]
    mp = _make_mp_surprises_df(meetings)
    events = _make_events_df(
        event_dates=event_dates, assets=assets, horizons=horizons
    )
    ling = _make_linguistic_df(len(event_dates))
    macro = _make_macro_df(event_dates)

    rows, summary = car.build_supervised_rows(
        events=events,
        mp_surprises=mp,
        linguistic_features=ling,
        macro_state=macro,
    )

    # 12 meetings * 3 assets * 2 horizons = 72 supervised rows.
    expected = len(meetings) * len(assets) * len(horizons)
    assert summary["rows_emitted"] == expected
    assert summary["asset_universe"] == sorted(assets)
    assert summary["horizons"] == sorted(horizons)
    # Rows arrive sorted by (date, asset, horizon).
    keys = [(r.feature_event_date, r.asset_symbol, r.horizon) for r in rows]
    assert keys == sorted(keys)


def test_build_supervised_rows_drops_missing_abnormal_return() -> None:
    """A row whose abnormal_return is NaN must drop, leaving the rest intact."""

    meetings = _make_canonical_meetings()[:3]
    event_dates = [m[0] for m in meetings]
    assets = ["^GSPC", "GC=F"]
    horizons = [1]
    mp = _make_mp_surprises_df(meetings)
    events = _make_events_df(
        event_dates=event_dates, assets=assets, horizons=horizons
    )
    # Corrupt one cell's abnormal_return.
    mask = (
        (events["event_date"] == event_dates[1])
        & (events["asset_symbol"] == "GC=F")
    )
    events.loc[mask, "abnormal_return"] = float("nan")
    ling = _make_linguistic_df(len(event_dates))
    macro = _make_macro_df(event_dates)

    rows, summary = car.build_supervised_rows(
        events=events,
        mp_surprises=mp,
        linguistic_features=ling,
        macro_state=macro,
    )
    # 6 - 1 = 5 rows survive.
    assert summary["rows_emitted"] == 5
    assert summary["dropped_missing_target"] == 1
    # The dropped row's (date, asset) pair is absent.
    missing_keys = {(r.feature_event_date, r.asset_symbol) for r in rows}
    assert (_dt.date.fromisoformat(event_dates[1]), "GC=F") not in missing_keys


def test_build_supervised_rows_honours_asset_filter() -> None:
    meetings = _make_canonical_meetings()[:4]
    event_dates = [m[0] for m in meetings]
    assets = ["^GSPC", "^TNX", "GC=F", "CL=F"]
    horizons = [1, 5]
    mp = _make_mp_surprises_df(meetings)
    events = _make_events_df(
        event_dates=event_dates, assets=assets, horizons=horizons
    )
    ling = _make_linguistic_df(len(event_dates))
    macro = _make_macro_df(event_dates)

    rows, summary = car.build_supervised_rows(
        events=events,
        mp_surprises=mp,
        linguistic_features=ling,
        macro_state=macro,
        asset_universe=["^GSPC", "GC=F"],
        horizons=[1],
    )

    expected = len(meetings) * 2 * 1
    assert summary["rows_emitted"] == expected
    assert set(r.asset_symbol for r in rows) == {"^GSPC", "GC=F"}
    assert set(r.horizon for r in rows) == {1}
    # The filter logs dropped counts.
    assert summary["dropped_unknown_asset"] > 0
    assert summary["dropped_unknown_horizon"] > 0


def test_collapse_events_picks_statement_over_minutes() -> None:
    """When both kinds exist for a meeting, statement wins."""

    df = pd.DataFrame(
        [
            {
                "event_date": "2022-01-26",
                "event_kind": "minutes",
                "asset_symbol": "^GSPC",
                "horizon": 1,
                "axis_stance": "neutral",
                "abnormal_return": 0.1,
            },
            {
                "event_date": "2022-01-26",
                "event_kind": "statement",
                "asset_symbol": "^GSPC",
                "horizon": 1,
                "axis_stance": "hawkish",
                "abnormal_return": 0.2,
            },
        ]
    )
    out = car._collapse_events_to_meeting_axis(df)
    assert len(out) == 1
    assert out.iloc[0]["event_kind"] == "statement"
    assert out.iloc[0]["axis_stance"] == "hawkish"


# ---------------------------------------------------------------------------
# Feature ablation
# ---------------------------------------------------------------------------


def test_feature_ablation_drops_columns() -> None:
    meetings = _make_canonical_meetings()
    event_dates = [m[0] for m in meetings]
    assets = ["^GSPC"]
    horizons = [1]
    mp = _make_mp_surprises_df(meetings)
    events = _make_events_df(
        event_dates=event_dates, assets=assets, horizons=horizons
    )
    ling = _make_linguistic_df(len(event_dates))
    macro = _make_macro_df(event_dates)
    rows, _ = car.build_supervised_rows(
        events=events,
        mp_surprises=mp,
        linguistic_features=ling,
        macro_state=macro,
    )

    X_full, names_full = car._build_feature_matrix(rows, car.FEATURE_FAMILIES)
    X_ois, names_ois = car._build_feature_matrix(rows, ("ois",))
    X_no_macro, names_no_macro = car._build_feature_matrix(
        rows, ("ois", "text", "linguistic", "credibility")
    )

    # Each ablation removes columns; nothing is silently retained.
    assert X_ois.shape[1] < X_full.shape[1]
    assert X_no_macro.shape[1] < X_full.shape[1]
    assert set(names_ois).issubset(set(names_full))
    assert set(names_no_macro).issubset(set(names_full))
    # Dropping macro removes exactly the macro feature columns -- the
    # set difference matches the documented macro family schema.
    macro_cols = set(rows[0].feature_names["macro"])
    assert set(names_full) - set(names_no_macro) == macro_cols
    # OIS-only keeps exactly the OIS family.
    assert set(names_ois) == set(rows[0].feature_names["ois"])


# ---------------------------------------------------------------------------
# Walk-forward CV
# ---------------------------------------------------------------------------


def test_walk_forward_cell_produces_n_minus_1_train_rows() -> None:
    meetings = _make_canonical_meetings()
    event_dates = [m[0] for m in meetings]
    assets = ["^GSPC"]
    horizons = [1]
    mp = _make_mp_surprises_df(meetings)
    events = _make_events_df(
        event_dates=event_dates, assets=assets, horizons=horizons
    )
    ling = _make_linguistic_df(len(event_dates))
    macro = _make_macro_df(event_dates)
    rows, _ = car.build_supervised_rows(
        events=events,
        mp_surprises=mp,
        linguistic_features=ling,
        macro_state=macro,
    )

    cell_rows = car._rows_for_cell(rows, asset="^GSPC", horizon=1)
    preds = car.walk_forward_predict_cell(
        cell_rows, families=("ois",), include_gbt=False
    )
    assert len(preds) == len(cell_rows)
    assert [p.n_train_rows for p in preds] == list(range(len(cell_rows)))
    # Every prediction has the two model-free baselines.
    for p in preds:
        assert "zero_baseline" in p.predictions
        assert "ois_bp_baseline" in p.predictions


def test_walk_forward_cell_no_look_ahead_assertion() -> None:
    """The walk-forward routine asserts no-look-ahead internally."""

    meetings = _make_canonical_meetings()
    event_dates = [m[0] for m in meetings]
    mp = _make_mp_surprises_df(meetings)
    events = _make_events_df(
        event_dates=event_dates, assets=["^GSPC"], horizons=[1]
    )
    ling = _make_linguistic_df(len(event_dates))
    macro = _make_macro_df(event_dates)
    rows, _ = car.build_supervised_rows(
        events=events,
        mp_surprises=mp,
        linguistic_features=ling,
        macro_state=macro,
    )
    cell_rows = car._rows_for_cell(rows, asset="^GSPC", horizon=1)
    # Confirm: the internal assertion does not fire on well-formed data.
    car.walk_forward_predict_cell(cell_rows, families=("ois",), include_gbt=False)

    # External cross-check: rebuild a deliberately mis-ordered cell and
    # verify the internal AssertionError fires.
    bad = list(cell_rows)
    bad[0], bad[-1] = bad[-1], bad[0]  # swap first and last
    with pytest.raises(AssertionError, match="walk-forward leak"):
        car.walk_forward_predict_cell(bad, families=("ois",), include_gbt=False)


def test_walk_forward_cell_ridge_fits_synthetic_linear_target() -> None:
    """A pure-linear target in the feature subspace lets ridge approximate it.

    We synthesise the abnormal return for ^GSPC as a linear function of
    the mp_surprise_level + path factor, then check that ridge's
    out-of-fold MAE is much lower than the zero baseline's MAE on the
    held-out tail.
    """

    rng = np.random.default_rng(seed=11)
    n_meetings = 40
    base = _dt.date(2018, 1, 1)
    meetings = []
    for i in range(n_meetings):
        d = base + _dt.timedelta(days=42 * i)
        prior = round(0.25 * (i // 5), 3)
        after = prior  # no rate move; surprises live in MP-surprise columns
        meetings.append((d.isoformat(), prior, after))

    mp_df = _make_mp_surprises_df(meetings)
    # Make mp_surprise_level vary so the linear target has something to
    # learn against.
    levels = rng.normal(size=n_meetings)
    mp_df["mp_surprise_level"] = levels

    event_dates = [m[0] for m in meetings]
    events = _make_events_df(
        event_dates=event_dates, assets=["^GSPC"], horizons=[1]
    )
    # Target: 0.7 * surprise_level + small noise.
    noise = rng.normal(scale=0.05, size=n_meetings)
    abnormals = 0.7 * levels + noise
    for i, ed in enumerate(event_dates):
        mask = events["event_date"] == ed
        events.loc[mask, "abnormal_return"] = float(abnormals[i])
    ling = _make_linguistic_df(len(event_dates))
    macro = _make_macro_df(event_dates)

    rows, _ = car.build_supervised_rows(
        events=events,
        mp_surprises=mp_df,
        linguistic_features=ling,
        macro_state=macro,
    )
    cell_rows = car._rows_for_cell(rows, asset="^GSPC", horizon=1)
    preds = car.walk_forward_predict_cell(
        cell_rows, families=("ois",), include_gbt=False
    )

    # Restrict to the tail half where ridge has enough train rows.
    tail = preds[len(preds) // 2 :]
    ridge_mae = np.mean(
        [
            abs(p.target - p.predictions["ridge"])
            for p in tail
            if "ridge" in p.predictions
        ]
    )
    zero_mae = np.mean([abs(p.target) for p in tail])
    # Ridge should comfortably beat the zero baseline on the tail half;
    # 0.7 of the target is in-subspace so a < 0.7 ratio is sufficient.
    assert ridge_mae < 0.7 * zero_mae


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_compute_cell_metrics_zero_baseline_against_zero_truth() -> None:
    """Zero predictions vs zero truth -> RMSE 0, R^2 None (zero var)."""

    preds = [
        car.CellPrediction(
            feature_event_date="2022-01-26",
            asset_symbol="^GSPC",
            horizon=1,
            target=0.0,
            n_train_rows=0,
            predictions={"zero_baseline": 0.0},
        )
        for _ in range(5)
    ]
    m = car.compute_cell_metrics(preds, "zero_baseline")
    assert m["n"] == 5
    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0
    assert m["r2"] is None  # zero-variance truth


def test_compute_cell_metrics_directional_hit_rate() -> None:
    """All-positive predictions vs alternating-sign truth -> 50% hit rate."""

    preds = []
    for i, truth in enumerate([0.1, -0.2, 0.3, -0.4]):
        preds.append(
            car.CellPrediction(
                feature_event_date=f"2022-01-{20 + i:02d}",
                asset_symbol="^GSPC",
                horizon=1,
                target=truth,
                n_train_rows=i,
                predictions={"ridge": 0.05},
            )
        )
    m = car.compute_cell_metrics(preds, "ridge")
    assert m["directional_hit_rate"] == 0.5


# ---------------------------------------------------------------------------
# Pandemic-window filter
# ---------------------------------------------------------------------------


def test_pandemic_window_filter_excludes_documented_range() -> None:
    in_window = car.CellPrediction(
        feature_event_date="2020-05-15",
        asset_symbol="^GSPC",
        horizon=1,
        target=0.0,
        n_train_rows=10,
        predictions={"zero_baseline": 0.0},
    )
    out_window = car.CellPrediction(
        feature_event_date="2023-09-20",
        asset_symbol="^GSPC",
        horizon=1,
        target=0.0,
        n_train_rows=20,
        predictions={"zero_baseline": 0.0},
    )
    # Right at the boundary: PANDEMIC_END (2021-06-30) inclusive.
    on_boundary = car.CellPrediction(
        feature_event_date=car.PANDEMIC_END.isoformat(),
        asset_symbol="^GSPC",
        horizon=1,
        target=0.0,
        n_train_rows=15,
        predictions={"zero_baseline": 0.0},
    )
    filtered = car.filter_predictions_excluding_window(
        [in_window, out_window, on_boundary],
        start=car.PANDEMIC_START,
        end=car.PANDEMIC_END,
    )
    assert len(filtered) == 1
    assert filtered[0].feature_event_date == "2023-09-20"


# ---------------------------------------------------------------------------
# End-to-end run + artifact serialisation
# ---------------------------------------------------------------------------


def test_run_writes_artifacts(tmp_path: Path) -> None:
    meetings = _make_canonical_meetings()
    event_dates = [m[0] for m in meetings]
    assets = ["^GSPC", "^TNX"]
    horizons = [1, 5]
    mp = _make_mp_surprises_df(meetings)
    events = _make_events_df(
        event_dates=event_dates, assets=assets, horizons=horizons
    )
    ling = _make_linguistic_df(len(event_dates))
    macro = _make_macro_df(event_dates)

    artifacts = car.run(
        events=events,
        mp_surprises=mp,
        linguistic_features=ling,
        macro_state=macro,
        output_dir=tmp_path,
        # The synthetic frame's targets are flat; disable pooled to keep
        # the run fast and deterministic.
        include_pooled=False,
    )

    predictions_path = tmp_path / "predictions.json"
    metrics_path = tmp_path / "metrics.json"
    attribution_path = tmp_path / "feature_attribution.md"
    assert predictions_path.exists()
    assert metrics_path.exists()
    assert attribution_path.exists()

    metrics = json.loads(metrics_path.read_text())
    assert "full_window" in metrics
    assert "ex_pandemic_window" in metrics
    assert metrics["asset_universe"] == sorted(assets)
    assert metrics["horizons"] == sorted(horizons)
    assert metrics["methodology_source"] == "cross_asset_event_response_v1"

    attribution_text = attribution_path.read_text()
    assert "ois_only" in attribution_text
    assert "full" in attribution_text
    # The headline cell appears in the markdown.
    assert "^GSPC|h1" in attribution_text
    assert artifacts.summary["rows_emitted"] > 0


def test_ois_baseline_bp_from_mp_row_handles_missing_curve() -> None:
    """No curve, no signal -- the baseline bp is ``None``."""

    s = pd.Series(
        {
            "post_event_curve": None,
            "ff_target_after": 5.0,
        }
    )
    assert car._ois_baseline_bp_from_mp_row(s) is None
    # Valid curve and base rate -> a numeric bp signal.
    s2 = pd.Series(
        {
            "post_event_curve": _make_pre_event_curve(5.25),
            "ff_target_after": 5.0,
        }
    )
    bp = car._ois_baseline_bp_from_mp_row(s2)
    assert bp is not None
    assert math.isclose(bp, 25.0, abs_tol=1e-6)
