"""#214 FOMC press conference Q&A feature wiring.

Covers the three structural locks introduced under ADR 0037:

1. ``FeatureVector.as_rich_list`` is byte-identical to the pre-#214
   output when ``press_conf_features is None`` (the default-off path
   must not widen the per-bar feature size or change a single slot).
2. With the slot populated, the press-conf scalar lands at the
   documented offset past the regime / SEP tails — the conditional
   append order is part of the input-tensor contract every consumer
   relies on.
3. ``rich_feature_size_with_blocks`` returns ``RICH_FEATURE_SIZE + 1``
   when ``use_press_conf=True`` and stacks additively with the regime
   / SEP flags so checkpoints trained under multiple opt-ins
   rehydrate at the matching width.
"""

from __future__ import annotations

import pytest

from app.models.config import (
    FeatureVector,
    ModelConfig,
    RICH_FEATURE_SIZE,
    RICH_MACRO_REGIME_DIM,
    RICH_MACRO_REGIME_MISSING_DIM,
    RICH_PRESS_CONF_DIM,
    RICH_SEP_DIM,
    RICH_SEP_MISSING_DIM,
    rich_feature_size_with_blocks,
)


def _baseline_vector() -> FeatureVector:
    return FeatureVector(
        date="2024-01-31",
        sentiment_score=0.5,
        market_close=4000.0,
        market_volatility=0.012,
        close_change_pct=0.001,
        volatility_change=0.0,
        elapsed_time=-1.0,
    )


def test_default_press_conf_slot_keeps_rich_list_byte_identical() -> None:
    """Pre-#214 byte-identity: ``as_rich_list`` returns ``RICH_FEATURE_SIZE``
    floats when ``press_conf_features`` is left at the default ``None``."""

    fv = _baseline_vector()
    assert fv.press_conf_features is None
    rich = fv.as_rich_list()
    assert len(rich) == RICH_FEATURE_SIZE


def test_press_conf_slot_appends_one_scalar_at_documented_offset() -> None:
    """When the slot is populated, ``as_rich_list`` appends exactly one
    scalar past the legacy width and the value at that slot is the
    ``has_press_conf`` flag."""

    fv = _baseline_vector()
    fv.press_conf_features = [1.0]
    rich = fv.as_rich_list()
    assert len(rich) == RICH_FEATURE_SIZE + RICH_PRESS_CONF_DIM
    assert rich[RICH_FEATURE_SIZE] == 1.0


def test_press_conf_pre_2011_zero_imputed_flag_lands_at_zero() -> None:
    """Covariate-shift handling: pre-2011 events get a zero-imputed flag
    so the walk-forward fold protocol stays whole (route 1 per ADR 0037)."""

    fv = _baseline_vector()
    fv.press_conf_features = [0.0]
    rich = fv.as_rich_list()
    assert rich[RICH_FEATURE_SIZE] == 0.0
    assert len(rich) == RICH_FEATURE_SIZE + 1


def test_press_conf_stacks_with_regime_and_sep_blocks() -> None:
    """All three optional blocks land at the documented append order
    (regime, then SEP, then press_conf) so a checkpoint trained under
    multiple opt-ins rehydrates with the matching widened width."""

    fv = _baseline_vector()
    fv.macro_regime_features = [1.0, 0.0, -1.0]
    fv.macro_regime_features_missing = 0.0
    fv.sep_features = [5.25, 4.0, 3.0, 0.50, 1.0]
    fv.sep_features_missing = 0.0
    fv.press_conf_features = [1.0]

    rich = fv.as_rich_list()
    expected = (
        RICH_FEATURE_SIZE
        + RICH_MACRO_REGIME_DIM
        + RICH_MACRO_REGIME_MISSING_DIM
        + RICH_SEP_DIM
        + RICH_SEP_MISSING_DIM
        + RICH_PRESS_CONF_DIM
    )
    assert len(rich) == expected

    # Press-conf scalar lands at the very end under the append order.
    assert rich[-1] == 1.0


def test_rich_feature_size_with_blocks_respects_press_conf_flag() -> None:
    """The size helper must agree with ``as_rich_list`` for every
    combination of the three flags so the model factory's input
    projection widens in lockstep with the loader's emission."""

    assert rich_feature_size_with_blocks(
        use_regime=False, use_sep=False, use_press_conf=False
    ) == RICH_FEATURE_SIZE
    assert rich_feature_size_with_blocks(
        use_regime=False, use_sep=False, use_press_conf=True
    ) == RICH_FEATURE_SIZE + RICH_PRESS_CONF_DIM
    assert rich_feature_size_with_blocks(
        use_regime=True, use_sep=False, use_press_conf=True
    ) == (
        RICH_FEATURE_SIZE
        + RICH_MACRO_REGIME_DIM
        + RICH_MACRO_REGIME_MISSING_DIM
        + RICH_PRESS_CONF_DIM
    )
    assert rich_feature_size_with_blocks(
        use_regime=True, use_sep=True, use_press_conf=True
    ) == (
        RICH_FEATURE_SIZE
        + RICH_MACRO_REGIME_DIM
        + RICH_MACRO_REGIME_MISSING_DIM
        + RICH_SEP_DIM
        + RICH_SEP_MISSING_DIM
        + RICH_PRESS_CONF_DIM
    )


def test_model_config_carries_use_press_conf_default_off() -> None:
    """Default-off contract: the ModelConfig flag defaults to False so
    every existing checkpoint deserialises without re-encoding."""

    config = ModelConfig()
    assert config.use_press_conf is False


def test_model_config_use_press_conf_round_trips_through_coerce_payload() -> None:
    """A saved checkpoint payload must rehydrate the press-conf opt-in
    so the input projection widens at the right width on resume."""

    from app.training.checkpoint import _coerce_payload_config

    payload = {
        "model_config": {
            "input_size": 6,
            "use_press_conf": True,
        }
    }
    coerced = _coerce_payload_config(payload)
    assert coerced.use_press_conf is True

    # Default-off path: pre-#214 checkpoints without the key collapse
    # to the byte-identical no-press-conf configuration.
    coerced_legacy = _coerce_payload_config({"model_config": {"input_size": 6}})
    assert coerced_legacy.use_press_conf is False


def test_qa_lookup_reader_returns_empty_dict_on_missing_parquet(
    tmp_path, monkeypatch
) -> None:
    """The loader's #214 lookup reader is graceful-degrade: an absent
    parquet returns ``{}`` and downstream every event collapses to the
    pre-2011 zero-impute path."""

    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    from app.training import loaders

    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)
    lookup = loaders._read_press_conf_qa_lookup(tmp_path / "missing_package")
    assert lookup == {}


def test_qa_lookup_reader_keys_on_event_date(tmp_path, monkeypatch) -> None:
    """When the parquet is on disk, the reader keys on ISO event_date and
    propagates the ``has_press_conf`` flag with the Q&A text payload."""

    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    from app.training import loaders

    package_dir = tmp_path / "package"
    package_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "event_date": "2024-01-31",
                "qa_text": "Q. and A. content",
                "prepared_remarks_text": "Opening remarks",
                "has_press_conf": "1",
            },
            {
                "event_date": "2024-03-20",
                "qa_text": "More Q&A",
                "prepared_remarks_text": "More remarks",
                "has_press_conf": "1",
            },
        ]
    ).to_parquet(package_dir / "qa_lookup.parquet", index=False)

    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)
    lookup = loaders._read_press_conf_qa_lookup(package_dir)
    assert set(lookup.keys()) == {"2024-01-31", "2024-03-20"}
    assert lookup["2024-01-31"]["qa_text"].startswith("Q. and A.")
    assert lookup["2024-01-31"]["has_press_conf"] == 1.0


def test_loader_concats_qa_onto_statement_raw_text_under_lora(
    tmp_path, monkeypatch
) -> None:
    """Route 1 contract: when ``use_press_conf=True`` AND ``encoder_lora=True``,
    the LoRA path's per-event ``raw_text`` carries the statement text
    concatenated with the same-date Q&A so the encoder sees a single
    joint document."""

    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    from app.models.config import SEQUENCE_LENGTH
    from app.training import loaders

    package_id = "tp_press_conf_lora_concat_v1"
    processed_root = tmp_path / "processed"
    package_dir = processed_root / package_id
    package_dir.mkdir(parents=True)
    monkeypatch.setattr(loaders, "DATA_DIR", tmp_path)

    # Synth one FOMC statement event with a Q&A lookup hit on the same
    # event_date.
    import datetime as _dt
    import json as _json

    def _make_event(event_date: str, text_hash: str, base_close: float) -> dict:
        ed_local = _dt.date.fromisoformat(event_date)
        bars = [
            {
                "date": (ed_local - _dt.timedelta(days=offset)).isoformat(),
                "close": base_close + (SEQUENCE_LENGTH - offset) * 1.0,
                "volume": 1_000_000.0,
                "vol_5d": 0.012,
                "vol_20d": 0.018,
                "vol_60d": 0.022,
                "cum_return_20d": 0.0,
                "vix_close": 15.0,
                "dxy_close": 103.0,
                "tnx_close": 4.2,
                "gold_close": 2050.0,
                "vix3m_close": 17.0,
                "irx_close": 5.1,
                "vix_term_slope": 0.0,
                "yield_curve_slope_10y_3m": -0.9,
            }
            for offset in range(SEQUENCE_LENGTH, 0, -1)
        ]
        return {
            "event_date": event_date,
            "event_kind": "statement",
            "document_id": text_hash[:16],
            "text_hash": text_hash,
            "source": "scraped_fed",
            "source_record_id": f"src:{text_hash}",
            "as_of_ts": f"{event_date}T19:00:00Z",
            "text": f"STATEMENT BODY {text_hash}",
            "token_count": 2,
            "axis_stance": "hawkish",
            "axis_time": None,
            "axis_certainty": None,
            "axis_factor": None,
            "axis_time_label": None,
            "axis_certain_label": None,
            "credibility_drift_score": 0.0,
            "credibility_realized_vs_stated_gap": 0.0,
            "credibility_market_implied_gap": 0.0,
            "credibility_months_since_reversal": 0,
            "prior_window_sha256": "0" * 64,
            "prior_bars_json": _json.dumps(bars, separators=(",", ":"), sort_keys=True),
            "asset_symbol": "^GSPC",
            "horizon": 1,
            "realized_return": 0.01,
            "abnormal_return": 0.01,
            "alpha": 0.0,
            "beta": 1.0,
            "direction_t1d": 1,
            "volatility_shift": 0.0,
            "concurrent_macro_release": False,
            "intra_meeting_stance_shift": 0.0,
            "intra_meeting_certainty_shift": 0.0,
            "intra_meeting_factor_shift": 0.0,
            "realized_date": (
                _dt.date.fromisoformat(event_date) + _dt.timedelta(days=1)
            ).isoformat(),
            "forward_realized_vol_10d": 0.015,
            "yield_2y_change_5d": -3.2,
            "yield_5y_change_5d": -2.1,
            "terminal_rate_change_5d": -1.0,
        }

    event_date = "2024-01-31"
    ed = _dt.date.fromisoformat(event_date)
    events = [
        _make_event(event_date, "hash_a", 4400.0),
        _make_event("2024-03-20", "hash_b", 4500.0),
    ]
    pd.DataFrame(events).to_parquet(package_dir / "events.parquet", index=False)
    pd.DataFrame(
        [
            {"text_hash": "hash_a", "split_tag": "train"},
            {"text_hash": "hash_b", "split_tag": "test"},
        ]
    ).to_parquet(package_dir / "splits_train_val_test.parquet", index=False)
    pd.DataFrame(
        [
            {
                "event_date": event_date,
                "qa_text": "POWELL Q AND A",
                "prepared_remarks_text": "OPENING",
                "has_press_conf": "1",
            }
        ]
    ).to_parquet(package_dir / "qa_lookup.parquet", index=False)

    # Off path: raw_text == statement only, byte-identical pre-#214.
    split_off = loaders.load_walk_forward_split(
        package_id,
        text_encoder=None,
        encoder_lora=True,
        use_press_conf=False,
    )
    assert split_off.train
    raw_off = split_off.train[0][-1].raw_text
    assert "STATEMENT BODY" in raw_off
    assert "POWELL Q AND A" not in raw_off

    # On path: raw_text carries the joint statement + Q&A text.
    split_on = loaders.load_walk_forward_split(
        package_id,
        text_encoder=None,
        encoder_lora=True,
        use_press_conf=True,
    )
    raw_on = split_on.train[0][-1].raw_text
    assert "STATEMENT BODY" in raw_on
    assert "POWELL Q AND A" in raw_on
    # press_conf_features slot must also fire on the joint corpus rows.
    assert split_on.train[0][-1].press_conf_features == [1.0]


def test_press_conf_features_composer_emits_one_on_lookup_hit() -> None:
    """The per-event composer emits ``[1.0]`` on a lookup hit and
    ``[0.0]`` on a miss (the canonical zero-impute for pre-2011 and
    other no-press-conf rows)."""

    from app.training.loaders import _compute_press_conf_features_for_event

    lookup = {"2024-01-31": {"has_press_conf": 1.0, "qa_text": "x"}}
    assert _compute_press_conf_features_for_event(
        event_date_str="2024-01-31", press_conf_lookup=lookup
    ) == [1.0]
    assert _compute_press_conf_features_for_event(
        event_date_str="2009-12-16", press_conf_lookup=lookup
    ) == [0.0]
    assert _compute_press_conf_features_for_event(
        event_date_str="2024-01-31", press_conf_lookup={}
    ) == [0.0]
