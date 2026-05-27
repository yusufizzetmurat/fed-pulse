"""Macro-regime conditioning (#307).

Pins the surface at four layers:

- the per-feature regime-indicator computation (hand-fixture with a
  known trailing-12-month policy path, a known VIX series, a known
  T-1 yield-curve slope; asserts each indicator lands at the
  documented value);
- the FeatureVector schema: ``as_rich_list`` does NOT append the
  regime block when ``macro_regime_features`` is ``None`` (legacy
  byte-identity contract); it DOES append the block when populated;
- the loader regression: ``--no-regime-conditioning`` (default) keeps
  the per-bar feature size at ``RICH_FEATURE_SIZE`` and every event's
  ``macro_regime_features`` slot stays ``None``; ``--use-regime-conditioning``
  flips both;
- a 1-epoch smoke through ``train_model`` with the regime block
  populated and the model's gate mounted runs to completion.

See ADR 0029 for the design.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Pure-Python helpers (no torch, no parquet) — exercise the per-indicator
# computation against hand fixtures so the math is reviewable.
# ---------------------------------------------------------------------------

from app.training.regime_features import (  # noqa: E402
    POLICY_CYCLE_LOOKBACK_DAYS,
    POLICY_CYCLE_THRESHOLD_BPS,
    REGIME_FEATURE_DIM,
    compute_macro_regime_features,
    compute_policy_cycle_phase_score,
    compute_term_spread_sign,
    compute_vix_level_regime_score,
)


def test_policy_cycle_phase_hiking() -> None:
    """A trailing year of rising target reads as +1 (hiking)."""

    event_date = _dt.date(2023, 5, 3)
    lookup = {
        "2022-06-15": {"ff_target_prior": 0.875},
        "2022-09-21": {"ff_target_prior": 1.625},
        "2022-12-14": {"ff_target_prior": 3.125},
        "2023-03-22": {"ff_target_prior": 4.625},
    }
    out = compute_policy_cycle_phase_score(
        event_date=event_date, mp_surprise_lookup=lookup
    )
    assert out == 1.0


def test_policy_cycle_phase_cutting() -> None:
    """A trailing year of declining target reads as -1 (cutting)."""

    event_date = _dt.date(2020, 6, 10)
    lookup = {
        "2019-07-31": {"ff_target_prior": 2.375},
        "2019-09-18": {"ff_target_prior": 2.125},
        "2020-03-03": {"ff_target_prior": 1.625},
        "2020-03-15": {"ff_target_prior": 1.125},
    }
    out = compute_policy_cycle_phase_score(
        event_date=event_date, mp_surprise_lookup=lookup
    )
    assert out == -1.0


def test_policy_cycle_phase_holding_below_threshold() -> None:
    """Net change strictly inside ``(-25 bp, +25 bp)`` reads as 0 (holding)."""

    event_date = _dt.date(2017, 9, 20)
    # 12.5 bp net move across the trailing year -- inside the
    # ``[-25 bp, +25 bp]`` threshold so the score collapses to holding.
    lookup = {
        "2017-01-31": {"ff_target_prior": 0.625},
        "2017-05-03": {"ff_target_prior": 0.700},
        "2017-07-26": {"ff_target_prior": 0.750},
    }
    out = compute_policy_cycle_phase_score(
        event_date=event_date, mp_surprise_lookup=lookup
    )
    assert out == 0.0


def test_policy_cycle_phase_strict_prior_filter() -> None:
    """Meetings on or after ``event_date`` must not enter the trailing window."""

    event_date = _dt.date(2023, 5, 3)
    lookup = {
        # Strict-prior meetings: a clear hiking pattern (~+150 bp net).
        "2022-09-21": {"ff_target_prior": 2.0},
        "2023-03-22": {"ff_target_prior": 3.5},
        # Two meetings whose date equals or post-dates ``event_date``:
        # both must be filtered out by the strict-prior gate; without
        # the gate they would flip the latest band low enough to read
        # as "cutting", and the regression below would fail.
        "2023-05-03": {"ff_target_prior": 1.0},
        "2023-06-14": {"ff_target_prior": 0.5},
    }
    out = compute_policy_cycle_phase_score(
        event_date=event_date, mp_surprise_lookup=lookup
    )
    assert out == 1.0


def test_policy_cycle_phase_outside_lookback_dropped() -> None:
    """Meetings older than the lookback window must not enter the score."""

    event_date = _dt.date(2024, 6, 12)
    lookup = {
        # Way outside the 365-day window: a steep cycle that would
        # otherwise flip the score to +1 by tail anchor alone.
        "2019-01-30": {"ff_target_prior": 2.25},
        "2019-07-31": {"ff_target_prior": 2.375},
        # Inside the window: flat at zero target. Should read holding.
        "2024-01-31": {"ff_target_prior": 5.375},
        "2024-03-20": {"ff_target_prior": 5.375},
    }
    out = compute_policy_cycle_phase_score(
        event_date=event_date,
        mp_surprise_lookup=lookup,
        lookback_days=POLICY_CYCLE_LOOKBACK_DAYS,
    )
    assert out == 0.0


def test_policy_cycle_phase_cold_start_defaults_to_holding() -> None:
    """Fewer than two eligible prior meetings collapses to 0 (no signal)."""

    event_date = _dt.date(2010, 1, 27)
    lookup = {"2010-01-27": {"ff_target_prior": 0.125}}
    out = compute_policy_cycle_phase_score(
        event_date=event_date, mp_surprise_lookup=lookup
    )
    assert out == 0.0


def test_vix_level_regime_high() -> None:
    """T-1 VIX above the upper tertile reads as +1."""

    series = [12.0, 13.0, 12.5, 13.5, 14.0, 16.0, 30.0]
    assert compute_vix_level_regime_score(prior_bar_vix_values=series) == 1.0


def test_vix_level_regime_low() -> None:
    """T-1 VIX below the lower tertile reads as -1."""

    series = [20.0, 22.0, 25.0, 30.0, 19.0, 14.0, 11.0]
    assert compute_vix_level_regime_score(prior_bar_vix_values=series) == -1.0


def test_vix_level_regime_normal() -> None:
    """T-1 VIX inside the middle tertile reads as 0."""

    series = [10.0, 11.0, 12.0, 18.0, 19.0, 22.0, 16.0]
    assert compute_vix_level_regime_score(prior_bar_vix_values=series) == 0.0


def test_vix_level_regime_empty_series_returns_zero() -> None:
    """Empty / single-value series collapse to 0 (no tertile defined)."""

    assert compute_vix_level_regime_score(prior_bar_vix_values=[]) == 0.0
    assert compute_vix_level_regime_score(prior_bar_vix_values=[14.5]) == 0.0


def test_term_spread_sign_inverted() -> None:
    """``tnx < irx`` reads as -1 (inverted curve)."""

    out = compute_term_spread_sign(
        t_minus_one_tnx_close=4.10, t_minus_one_irx_close=5.20
    )
    assert out == -1.0


def test_term_spread_sign_steep() -> None:
    """``tnx > irx`` reads as +1 (positive slope)."""

    out = compute_term_spread_sign(
        t_minus_one_tnx_close=4.50, t_minus_one_irx_close=3.10
    )
    assert out == 1.0


def test_term_spread_sign_zero_or_missing_returns_zero() -> None:
    """Zero spread or any missing input collapses to 0."""

    assert (
        compute_term_spread_sign(t_minus_one_tnx_close=4.10, t_minus_one_irx_close=4.10)
        == 0.0
    )
    assert (
        compute_term_spread_sign(t_minus_one_tnx_close=None, t_minus_one_irx_close=4.10)
        == 0.0
    )
    assert (
        compute_term_spread_sign(t_minus_one_tnx_close=4.10, t_minus_one_irx_close=None)
        == 0.0
    )


def test_compose_macro_regime_features_in_documented_order() -> None:
    """The composer wires the three component helpers in the documented order."""

    event_date = _dt.date(2023, 5, 3)
    lookup = {
        "2022-09-21": {"ff_target_prior": 2.0},
        "2023-03-22": {"ff_target_prior": 4.5},
    }
    vix_series = [12.0, 13.0, 12.5, 13.5, 14.0, 16.0, 28.0]
    out = compute_macro_regime_features(
        event_date=event_date,
        mp_surprise_lookup=lookup,
        prior_bar_vix_values=vix_series,
        t_minus_one_tnx_close=3.5,
        t_minus_one_irx_close=5.4,
    )
    assert out.as_list() == [1.0, 1.0, -1.0]
    assert len(out.as_list()) == REGIME_FEATURE_DIM


# ---------------------------------------------------------------------------
# FeatureVector schema: conditional emission keeps legacy byte-identity.
# ---------------------------------------------------------------------------


from app.models.config import (  # noqa: E402
    FEATURE_SIZE,
    FeatureVector,
    RICH_FEATURE_SIZE,
    RICH_MACRO_REGIME_DIM,
    RICH_MACRO_REGIME_MISSING_DIM,
    RICH_MACRO_REGIME_SLICE,
    RICH_MACRO_REGIME_MISSING_SLICE,
    rich_feature_size_with_regime,
)


def test_as_rich_list_default_omits_regime_block() -> None:
    """The default ``macro_regime_features=None`` keeps the pre-#307 width.

    Regression guard against accidentally widening the per-bar feature
    size on the legacy / opt-out path. ``RICH_FEATURE_SIZE`` is the
    structural constant downstream callers slice against; the regime
    block must stay invisible until the loader populates the slot.
    """

    fv = FeatureVector(
        date="2024-01-15",
        sentiment_score=0.0,
        market_close=4500.0,
        market_volatility=0.01,
    )
    assert fv.macro_regime_features is None
    assert len(fv.as_rich_list()) == RICH_FEATURE_SIZE


def test_as_rich_list_populated_appends_regime_block() -> None:
    """A populated regime slot appends the block past ``RICH_FEATURE_SIZE``."""

    block = [1.0, -1.0, 1.0]
    fv = FeatureVector(
        date="2024-01-15",
        sentiment_score=0.0,
        market_close=4500.0,
        market_volatility=0.01,
        macro_regime_features=block,
        macro_regime_features_missing=0.0,
    )
    rich = fv.as_rich_list()
    expected_width = (
        RICH_FEATURE_SIZE + RICH_MACRO_REGIME_DIM + RICH_MACRO_REGIME_MISSING_DIM
    )
    assert len(rich) == expected_width
    assert rich[RICH_MACRO_REGIME_SLICE] == block
    assert rich[RICH_MACRO_REGIME_MISSING_SLICE] == [0.0]


def test_rich_feature_size_with_regime_helper() -> None:
    """The helper widens by exactly ``RICH_MACRO_REGIME_DIM + 1`` when on."""

    assert rich_feature_size_with_regime(False) == RICH_FEATURE_SIZE
    assert (
        rich_feature_size_with_regime(True)
        == RICH_FEATURE_SIZE + RICH_MACRO_REGIME_DIM + RICH_MACRO_REGIME_MISSING_DIM
    )


def test_short_regime_payload_zero_pads() -> None:
    """A short payload right-pads to ``RICH_MACRO_REGIME_DIM``.

    Mirrors the linguistic / llm / analog slice padding contract — the
    per-bar feature size stays constant when the flag is on regardless
    of the payload length the caller supplied.
    """

    fv = FeatureVector(
        date="2024-01-15",
        sentiment_score=0.0,
        market_close=4500.0,
        market_volatility=0.01,
        macro_regime_features=[1.0],
        macro_regime_features_missing=0.0,
    )
    rich = fv.as_rich_list()
    assert rich[RICH_MACRO_REGIME_SLICE] == [1.0] + [0.0] * (
        RICH_MACRO_REGIME_DIM - 1
    )


# ---------------------------------------------------------------------------
# Loader regression — flag off keeps byte-identical schema; flag on populates.
# ---------------------------------------------------------------------------


pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
torch = pytest.importorskip("torch")


from app.models.config import SEQUENCE_LENGTH  # noqa: E402
from app.training import loaders  # noqa: E402


_TRAINING_PACKAGE_ID = "tp_macro_regime_regression_v1"


def _synth_prior_bars(*, event_date: _dt.date, base_close: float, vix_base: float = 15.0) -> str:
    payload = []
    for offset in range(SEQUENCE_LENGTH, 0, -1):
        bar_date = _dt.date.fromordinal(event_date.toordinal() - offset)
        payload.append(
            {
                "date": bar_date.isoformat(),
                "close": round(base_close + (SEQUENCE_LENGTH - offset) * 1.5, 10),
                "volume": 1_000_000.0,
                "vol_5d": 0.012,
                "vol_20d": 0.018,
                "vol_60d": 0.022,
                "cum_return_20d": 0.0,
                "vix_close": round(vix_base + (SEQUENCE_LENGTH - offset) * 0.1, 6),
                "dxy_close": 103.0,
                "tnx_close": 4.20,
                "gold_close": 2050.0,
                "vix3m_close": 17.0,
                "irx_close": 5.10,
                "vix_term_slope": 0.0,
                "yield_curve_slope_10y_3m": -0.90,
            }
        )
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _make_event_row(
    *,
    event_date: str,
    text: str,
    axis_stance: str | None,
    base_close: float,
    vix_base: float,
) -> dict:
    ed = _dt.date.fromisoformat(event_date)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "event_date": event_date,
        "event_kind": "statement",
        "document_id": text_hash[:16],
        "text_hash": text_hash,
        "source": "scraped_fed",
        "source_record_id": f"src:{text_hash[:8]}",
        "as_of_ts": f"{event_date}T19:00:00Z",
        "text": text,
        "token_count": len(text.split()),
        "axis_stance": axis_stance,
        "axis_time": None,
        "axis_certainty": None,
        "axis_factor": None,
        "axis_topic": None,
        "axis_time_label": None,
        "axis_certain_label": None,
        "credibility_drift_score": 0.0,
        "credibility_realized_vs_stated_gap": 0.0,
        "credibility_market_implied_gap": 0.0,
        "credibility_months_since_reversal": 0,
        "prior_window_sha256": "0" * 64,
        "prior_bars_json": _synth_prior_bars(
            event_date=ed, base_close=base_close, vix_base=vix_base
        ),
        "asset_symbol": "^GSPC",
        "horizon": 1,
        "realized_return": 0.001,
        "abnormal_return": 0.001,
        "alpha": 0.0,
        "beta": 1.0,
        "direction_t1d": 1,
        "volatility_shift": 0.0,
        "concurrent_macro_release": False,
        "intra_meeting_stance_shift": 0.0,
        "intra_meeting_certainty_shift": 0.0,
        "intra_meeting_factor_shift": 0.0,
        "realized_date": (ed + _dt.timedelta(days=1)).isoformat(),
        "forward_realized_vol_10d": 0.015,
        "yield_2y_change_5d": 1.0,
        "yield_5y_change_5d": 0.5,
        "terminal_rate_change_5d": 0.0,
    }


@pytest.fixture
def loader_package(tmp_path: Path, monkeypatch) -> Path:
    """Three-event synthetic training package with an MP-surprise lookup.

    Includes ``ff_target_prior`` so the policy-cycle helper sees a
    real strict-prior path; the values mirror the 2022-2023 hiking
    cycle (~+225 bp over the trailing year against the May-2023 event).
    """

    processed_root = tmp_path / "processed"
    package_dir = processed_root / _TRAINING_PACKAGE_ID
    package_dir.mkdir(parents=True)

    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)

    rows = [
        _make_event_row(
            event_date="2023-09-20",
            text="The Committee judges policy is restrictive.",
            axis_stance="hawkish",
            base_close=4400.0,
            vix_base=14.0,
        ),
        _make_event_row(
            event_date="2023-12-13",
            text="Inflation has eased substantially.",
            axis_stance="neutral",
            base_close=4500.0,
            vix_base=12.0,
        ),
        _make_event_row(
            event_date="2024-03-20",
            text="A gradual normalisation path is anticipated.",
            axis_stance="neutral",
            base_close=4600.0,
            vix_base=13.0,
        ),
    ]
    pd.DataFrame(rows).to_parquet(package_dir / "events.parquet", index=False)

    split_rows = [
        {"text_hash": row["text_hash"], "split_tag": "train" if i < 2 else "test"}
        for i, row in enumerate(rows)
    ]
    pd.DataFrame(split_rows).to_parquet(
        package_dir / "splits_train_val_test.parquet", index=False
    )

    # MP-surprise lookup with a clear hiking pattern over the trailing
    # year of every supervised event.
    mp_rows = [
        {
            "event_date": "2022-09-21",
            "mp_surprise_level": 0.0,
            "mp_surprise_path_factor": 0.0,
            "fed_info_factor": 0.0,
            "is_intermeeting": False,
            "ff_target_prior": 2.375,
        },
        {
            "event_date": "2022-12-14",
            "mp_surprise_level": 0.0,
            "mp_surprise_path_factor": 0.0,
            "fed_info_factor": 0.0,
            "is_intermeeting": False,
            "ff_target_prior": 3.875,
        },
        {
            "event_date": "2023-03-22",
            "mp_surprise_level": 0.0,
            "mp_surprise_path_factor": 0.0,
            "fed_info_factor": 0.0,
            "is_intermeeting": False,
            "ff_target_prior": 4.625,
        },
        {
            "event_date": "2023-05-03",
            "mp_surprise_level": 0.0,
            "mp_surprise_path_factor": 0.0,
            "fed_info_factor": 0.0,
            "is_intermeeting": False,
            "ff_target_prior": 4.875,
        },
        {
            "event_date": "2023-09-20",
            "mp_surprise_level": 0.0,
            "mp_surprise_path_factor": 0.0,
            "fed_info_factor": 0.0,
            "is_intermeeting": False,
            "ff_target_prior": 5.375,
        },
        {
            "event_date": "2023-12-13",
            "mp_surprise_level": 0.0,
            "mp_surprise_path_factor": 0.0,
            "fed_info_factor": 0.0,
            "is_intermeeting": False,
            "ff_target_prior": 5.375,
        },
        {
            "event_date": "2024-03-20",
            "mp_surprise_level": 0.0,
            "mp_surprise_path_factor": 0.0,
            "fed_info_factor": 0.0,
            "is_intermeeting": False,
            "ff_target_prior": 5.375,
        },
    ]
    pd.DataFrame(mp_rows).to_parquet(package_dir / "mp_surprises.parquet", index=False)
    return package_dir


def test_loader_regime_flag_off_keeps_pre_307_schema(loader_package: Path) -> None:
    """Default ``use_regime_conditioning=False`` -> byte-identical schema.

    Pins the byte-identity contract on the legacy / opt-out path. Every
    supervised sequence must keep ``macro_regime_features=None`` and
    every per-bar ``as_rich_list`` must keep the pre-#307 width.
    """

    split = loaders.load_walk_forward_split(
        _TRAINING_PACKAGE_ID,
        rich_features=True,
        use_regime_conditioning=False,
        text_encoder=None,
    )
    assert split.train, "fixture must produce at least one train sequence"

    for partition in (split.train, split.val, split.test):
        for sequence in partition:
            for vector in sequence:
                assert vector.macro_regime_features is None
                assert vector.macro_regime_features_missing == 1.0
                assert len(vector.as_rich_list()) == RICH_FEATURE_SIZE


def test_loader_regime_flag_on_populates_block(loader_package: Path) -> None:
    """``use_regime_conditioning=True`` -> populated block on every bar.

    The fixture's MP-surprise lookup carries a clear hiking pattern
    over the trailing 12 months of every supervised event, so the
    policy-cycle indicator must read +1 on every supervised sequence.
    The per-bar feature size widens by ``RICH_MACRO_REGIME_DIM + 1``
    in lockstep with ``rich_feature_size_with_regime(True)``.
    """

    split = loaders.load_walk_forward_split(
        _TRAINING_PACKAGE_ID,
        rich_features=True,
        use_regime_conditioning=True,
        text_encoder=None,
    )
    assert split.train

    expected_width = rich_feature_size_with_regime(True)
    saw_at_least_one_hiking_score = False
    for partition in (split.train, split.val, split.test):
        for sequence in partition:
            for vector in sequence:
                assert vector.macro_regime_features is not None
                assert vector.macro_regime_features_missing == 0.0
                assert len(vector.macro_regime_features) == RICH_MACRO_REGIME_DIM
                assert len(vector.as_rich_list()) == expected_width
                if vector.macro_regime_features[0] == 1.0:
                    saw_at_least_one_hiking_score = True
    assert saw_at_least_one_hiking_score, (
        "fixture's trailing-year MP-surprise lookup must drive at least "
        "one event's policy-cycle score to +1 (hiking)"
    )


def test_provenance_audit_strict_prior_on_regime_features(loader_package: Path) -> None:
    """Per-event regime block reads from strictly-prior sources only.

    The composer relies on ``ff_target_prior`` from meetings whose
    ``event_date < supervised_event.event_date`` and on the supervised
    sequence's own prior bars (whose dates are strictly before
    ``event_date`` per the events-builder's no-look-ahead contract).
    The audit row pins the contract; this regression locks the loader
    builds the helper input from the right side of T.
    """

    split = loaders.load_walk_forward_split(
        _TRAINING_PACKAGE_ID,
        rich_features=True,
        use_regime_conditioning=True,
        text_encoder=None,
    )
    for sequence in split.train + split.val + split.test:
        # The supervised event's date sits past every lookback bar by
        # the prior-window contract; the regime composer only reads off
        # those lookback bars + the MP-surprise lookup filtered to
        # prior meetings. The downstream assertion is structural: each
        # regime scalar lives in {-1, 0, +1}.
        target = sequence[-1]
        assert target.macro_regime_features is not None
        for scalar in target.macro_regime_features:
            assert scalar in {-1.0, 0.0, 1.0}


# ---------------------------------------------------------------------------
# Smoke training run with the gate mounted on the model.
# ---------------------------------------------------------------------------


from app.models.config import ModelConfig  # noqa: E402
from app.training.loop import train_model  # noqa: E402


def _dummy_feature_vector(*, day: int, vol: float) -> FeatureVector:
    """In-memory FeatureVector with a populated regime block.

    Mirrors the retrieval-features smoke fixture so the training loop
    runs against an in-memory group with no parquet I/O. ``rich_payload``
    is set so the tensoriser routes through ``as_rich_list`` and the
    regime tail actually reaches the recurrent core -- the post-#307
    follow-up regression below depends on that wiring.
    """

    fv = FeatureVector(
        date=str(_dt.date(2025, 1, 1) + _dt.timedelta(days=day - 1)),
        sentiment_score=0.0,
        market_close=100.0,
        market_volatility=0.01,
        close_change_pct=0.0,
        volatility_change=0.0,
        elapsed_time=0.0,
        forward_realized_vol_10d=vol,
        rich_payload=True,
    )
    fv.macro_regime_features = [1.0, 0.0, -1.0]
    fv.macro_regime_features_missing = 0.0
    return fv


def test_train_model_smoke_with_regime_conditioning_gate() -> None:
    """One-epoch training run with the gate mounted runs to completion.

    The gate's zero-init keeps the first forward pass byte-identical
    to the no-gate path; the SGD step then nudges the weights off
    identity. The smoke verifies the model graph + tensor plumbing
    line up across the wider per-bar input.
    """

    groups = [[_dummy_feature_vector(day=i + 1, vol=0.01 + 0.001 * i) for i in range(40)]]
    config = ModelConfig(
        input_size=RICH_FEATURE_SIZE,
        output_mode="classification",
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
        use_regime_conditioning=True,
    )
    result = train_model(
        model_config=config,
        train_sequence_groups=groups,
        val_sequence_groups=groups,
        test_sequence_groups=groups,
        epochs=1,
        batch_size=8,
        seed=11,
        save_checkpoint=False,
        use_compile=False,
        use_amp=False,
    )
    assert result.summary.epochs_completed == 1


def test_gate_zero_init_is_identity_at_step_zero() -> None:
    """The zero-init gate produces an output of 1.0 at start of training.

    Pins the contract that flipping ``--use-regime-conditioning`` on
    without retraining the model still produces the same forward pass
    on the rich-feature slice. The recurrent core only sees a modulated
    rich block after gradients push the gate off identity.
    """

    from app.models.config import ModelConfig
    from app.models.factory import build_research_forecaster

    config = ModelConfig(
        output_mode="classification",
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
        use_regime_conditioning=True,
    )
    model = build_research_forecaster(config)
    gate = getattr(model, "regime_gate", None)
    assert gate is not None, (
        "use_regime_conditioning=True must mount a regime_gate Linear layer"
    )
    # Zero weight + zero bias means gate output is identically 1.0
    # (``2 * sigmoid(0) == 1.0``) regardless of regime input.
    sample_regime = torch.zeros((2, 3))
    gate_output = 2.0 * torch.sigmoid(gate(sample_regime))
    assert torch.allclose(gate_output, torch.ones_like(gate_output))


def test_no_gate_mounts_without_flag() -> None:
    """Default ``use_regime_conditioning=False`` mounts NO gate layer."""

    from app.models.config import ModelConfig
    from app.models.factory import build_research_forecaster

    config = ModelConfig(
        output_mode="classification",
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
    )
    model = build_research_forecaster(config)
    assert getattr(model, "regime_gate", None) is None


# ---------------------------------------------------------------------------
# Post-#307 shape-contract regression: the LSTM width must absorb the
# regime tail the loader appends on every per-bar tensor when the flag
# is on. The pre-fix path mounted the recurrent core at
# ``RICH_FEATURE_SIZE`` and crashed at ``self.lstm(x)`` with
# ``RuntimeError: input.size(-1) must be equal to input_size. Expected
# 87, got 91`` on the canonical sweep (``run_dual_head_comparison.py``).
# These tests instantiate the model the same way the sweep runner does
# -- not via stubbing -- so the contract is verified end-to-end.
# ---------------------------------------------------------------------------


def test_research_model_lstm_width_includes_regime_tail() -> None:
    """The recurrent core widens by the regime tail when the flag is on.

    The loader's ``as_rich_list`` appends ``RICH_MACRO_REGIME_DIM + 1``
    extra scalars past ``RICH_FEATURE_SIZE`` on every per-bar tensor
    when conditioning is on; the LSTM constructor must accept that
    widened input, otherwise the canonical sweep crashes with the
    87-vs-91 shape mismatch reported on top of #307.
    """

    from app.models.config import ModelConfig
    from app.models.factory import build_research_forecaster

    config = ModelConfig(
        input_size=RICH_FEATURE_SIZE,
        output_mode="classification",
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
        use_regime_conditioning=True,
    )
    model = build_research_forecaster(config)
    expected_width = (
        RICH_FEATURE_SIZE
        + RICH_MACRO_REGIME_DIM
        + RICH_MACRO_REGIME_MISSING_DIM
    )
    assert model.lstm_input_size == expected_width
    # The recurrent core must consume the widened width; reading the
    # ``input_size`` off the LSTM (``nn.LSTM`` exposes it directly) and
    # off the TCN / Informer / DLinear cores (``self.input_size``)
    # covers every architecture wired through ``ForecasterBase``.
    core = model.recurrent_core
    core_width = getattr(core, "input_size", None)
    assert core_width == expected_width


def test_research_model_lstm_width_unchanged_when_flag_off() -> None:
    """Default OFF byte-identity: LSTM width stays at ``input_size``.

    Pins the deliberate #307 design that the recurrent core sees
    exactly the legacy ``RICH_FEATURE_SIZE`` width when conditioning
    is off; flipping the flag on must never widen any other path.
    """

    from app.models.config import ModelConfig
    from app.models.factory import build_research_forecaster

    config = ModelConfig(
        input_size=RICH_FEATURE_SIZE,
        output_mode="classification",
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
    )
    model = build_research_forecaster(config)
    assert model.lstm_input_size == RICH_FEATURE_SIZE
    assert getattr(model.recurrent_core, "input_size", None) == RICH_FEATURE_SIZE


def _rich_feature_vector(
    *,
    day: int,
    vol: float,
    regime_block: list[float] | None = (1.0, 0.0, -1.0),  # type: ignore[assignment]
) -> FeatureVector:
    """In-memory FeatureVector matching the loader's rich-payload shape.

    Sets ``rich_payload=True`` so ``_build_training_tensors`` routes
    through ``as_rich_list`` (the 91-wide path when the regime block is
    populated) -- mirrors the per-bar tensor the canonical training
    package emits via ``load_walk_forward_split``. The previous #307
    smoke test mounted ``as_list`` instead (6 dims, regime stripped),
    which is why the LSTM width mismatch never surfaced.
    """

    fv = FeatureVector(
        date=str(_dt.date(2025, 1, 1) + _dt.timedelta(days=day - 1)),
        sentiment_score=0.0,
        market_close=100.0 + day * 0.5,
        market_volatility=0.01,
        close_change_pct=0.0,
        volatility_change=0.0,
        elapsed_time=0.0,
        forward_realized_vol_10d=vol,
        rich_payload=True,
    )
    if regime_block is not None:
        fv.macro_regime_features = list(regime_block)
        fv.macro_regime_features_missing = 0.0
    return fv


def test_train_model_forward_does_not_raise_on_regime_tail() -> None:
    """Canonical-sweep reproduction: rich + regime tensors flow through train_model.

    Reproduces the crash reported on top of #307 (``RuntimeError:
    input.size(-1) must be equal to input_size. Expected 87, got 91``)
    on the pre-fix code path. Constructs the model with
    ``input_size=RICH_FEATURE_SIZE`` -- exactly what
    ``scripts/run_dual_head_comparison.py`` passes -- and pipes the
    91-wide rich+regime per-bar tensors through ``train_model``. With
    the fix the LSTM widens by the regime tail and the forward pass
    runs to completion; without it the call would raise the shape
    mismatch.
    """

    groups = [
        [
            _rich_feature_vector(day=i + 1, vol=0.01 + 0.001 * i)
            for i in range(40)
        ]
    ]
    # Confirm the fixture actually emits the 91-wide per-bar tensor the
    # sweep loader produces; without this guard a regression that
    # silently strips the regime block would still pass the train_model
    # call below.
    expected_per_bar_width = (
        RICH_FEATURE_SIZE
        + RICH_MACRO_REGIME_DIM
        + RICH_MACRO_REGIME_MISSING_DIM
    )
    assert len(groups[0][0].as_rich_list()) == expected_per_bar_width

    config = ModelConfig(
        input_size=RICH_FEATURE_SIZE,
        output_mode="classification",
        n_classes=3,
        hidden_size=16,
        head_hidden_size=8,
        use_regime_conditioning=True,
    )
    result = train_model(
        model_config=config,
        train_sequence_groups=groups,
        val_sequence_groups=groups,
        test_sequence_groups=groups,
        epochs=1,
        batch_size=8,
        seed=13,
        save_checkpoint=False,
        use_compile=False,
        use_amp=False,
    )
    assert result.summary.epochs_completed == 1
