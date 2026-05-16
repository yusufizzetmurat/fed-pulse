"""Unit tests for the pandera stage-boundary schemas.

Each schema gets three checks:

1. A synthetic happy-path frame validates without raising.
2. A targeted bad row triggers a ``SchemaError`` / ``SchemaErrors`` at
   the right column / value.
3. Lazy mode reports every violation in one exception, not just the
   first.

Three of the public emitters get a small write-seam integration test
that confirms the bad row's column name appears in the raised error.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pandera.errors as pa_errors
import pytest

from app.data import schemas as ps


# ---------------------------------------------------------------------------
# Synthetic-frame builders
# ---------------------------------------------------------------------------


def _hex64(payload: str = "x") -> str:
    return hashlib.sha256(payload.encode()).hexdigest()


def _hex16(payload: str = "x") -> str:
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _ingested_doc_row(**overrides) -> dict:
    base = {
        "record_id": _hex16("ing-1"),
        "source": "scraped_fed",
        "source_record_id": "fomc_statements.json:0",
        "document_type": "statement",
        "event_date": "2024-01-01",
        "text": "Sample FOMC statement text.",
        "label_origin": "human",
        "license_scope": "public_source_scrape_terms_required",
        "citation_ref": "federalreserve_primary_source",
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "text_hash": _hex64("ing-1"),
    }
    base.update(overrides)
    return base


def _normalized_doc_row(**overrides) -> dict:
    row = _ingested_doc_row()
    row.update(
        {
            "mapped_label": "neutral",
            "sample_weight": 1.0,
            "axes": {
                "stance": "neutral",
                "factor": None,
                "certainty": None,
                "topic": None,
            },
        }
    )
    row.update(overrides)
    return row


def _fold_row(split_tag: str = "train", **overrides) -> dict:
    row = _normalized_doc_row()
    row["split_tag"] = split_tag
    row.update(overrides)
    return row


def _event_row(**overrides) -> dict:
    base = {
        "event_date": "2024-01-15",
        "event_kind": "statement",
        "document_id": _hex16("evt-1"),
        "text_hash": _hex64("evt-1"),
        "source": "scraped_fed",
        "as_of_ts": "2024-01-15T19:00:00Z",
        "text": "Statement text body.",
        "token_count": 12,
        "axis_stance": "neutral",
        "axis_time": None,
        "axis_certainty": None,
        "axis_factor": None,
        "axis_topic": None,
        "credibility_drift_score": 0.0,
        "credibility_realized_vs_stated_gap": 0.0,
        "credibility_market_implied_gap": 0.0,
        "credibility_months_since_reversal": 0,
        "prior_window_sha256": _hex64("prior-1"),
        "prior_bars_json": json.dumps(
            [{"date": "2024-01-10", "close": 100.0, "volume": 1000.0}]
        ),
        "asset_symbol": "^GSPC",
        "horizon": 5,
        "realized_return": 0.01,
        "abnormal_return": 0.005,
        "alpha": 0.0,
        "beta": 1.0,
        "direction_t1d": 1,
        "volatility_shift": 0.02,
        "concurrent_macro_release": False,
        "intra_meeting_stance_shift": float("nan"),
        "intra_meeting_certainty_shift": float("nan"),
        "intra_meeting_factor_shift": float("nan"),
    }
    base.update(overrides)
    return base


def _linguistic_row(**overrides) -> dict:
    base = {
        "text_hash": _hex64("ling-1"),
        "topic_share_inflation": 0.1,
        "topic_share_employment": 0.1,
        "topic_share_financial_stability": 0.1,
        "topic_share_growth": 0.1,
        "topic_share_balance_sheet": 0.1,
        "topic_share_misc_1": 0.05,
        "topic_share_misc_2": 0.05,
        "topic_share_misc_3": 0.05,
        "hedge_density": 0.02,
        "comparison_density": 0.03,
        "forward_density": 0.04,
        "concrete_ratio": 0.5,
        "hawk_dove_asymmetry": 0.0,
        "log_token_count": 4.0,
        "pivot_distance": 0.3,
    }
    base.update(overrides)
    return base


def _mp_surprise_row(**overrides) -> dict:
    base = {
        "event_date": "2024-01-31",
        "meeting_id": 5,
        "ff_target_prior": 5.25,
        "ff_target_after": 5.25,
        "mp_surprise_level": 0.0,
        "mp_surprise_path_factor": 0.0,
        "pre_event_curve": json.dumps([{"months_ahead": 3, "implied_rate": 5.25}]),
        "post_event_curve": json.dumps([{"months_ahead": 3, "implied_rate": 5.25}]),
        "fed_info_factor": None,
        "is_intermeeting": False,
        "methodology": "ois_proxy",
        "data_version": "v1",
    }
    base.update(overrides)
    return base


def _macro_state_row(**overrides) -> dict:
    base = {
        "as_of_date": "2024-01-01",
        "unrate": 3.7,
        "cpi_yoy": 3.4,
        "core_pce_yoy": 2.9,
        "ism_proxy": 0.1,
        "payems_mom": 200_000.0,
        "rsafs_mom": 0.005,
        "ism_proxy_source": "MANEMP_3m_pct",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Happy-path validation
# ---------------------------------------------------------------------------


def test_ingested_doc_happy_path() -> None:
    frame = pd.DataFrame([_ingested_doc_row()])
    ps.validate_frame(ps.IngestedDocSchema, frame)


def test_normalized_doc_happy_path() -> None:
    frame = pd.DataFrame([_normalized_doc_row()])
    ps.validate_frame(ps.NormalizedDocSchema, frame)


def test_quality_passed_happy_path() -> None:
    rows = [
        _normalized_doc_row(record_id=_hex16(f"q-{i}"), text_hash=_hex64(f"q-{i}"))
        for i in range(3)
    ]
    frame = pd.DataFrame(rows)
    ps.validate_frame(ps.QualityPassedRowSchema, frame)


def test_fold_row_happy_path() -> None:
    rows = [
        _fold_row(
            split_tag=tag,
            record_id=_hex16(f"fold-{i}"),
            text_hash=_hex64(f"fold-{i}"),
        )
        for i, tag in enumerate(["train", "val", "test", "excluded_from_training"])
    ]
    frame = pd.DataFrame(rows)
    ps.validate_frame(ps.FoldRowSchema, frame)


def test_event_row_happy_path() -> None:
    frame = pd.DataFrame([_event_row()])
    ps.validate_frame(ps.EventRowSchema, frame)


def test_linguistic_row_happy_path() -> None:
    rows = [
        _linguistic_row(text_hash=_hex64(f"l-{i}"))
        for i in range(2)
    ]
    frame = pd.DataFrame(rows)
    ps.validate_frame(ps.LinguisticFeatureRowSchema, frame)


def test_mp_surprise_row_happy_path() -> None:
    frame = pd.DataFrame([_mp_surprise_row()])
    ps.validate_frame(ps.MpSurpriseRowSchema, frame)


def test_macro_state_row_happy_path() -> None:
    frame = pd.DataFrame([_macro_state_row()])
    ps.validate_frame(ps.MacroStateRowSchema, frame)


# ---------------------------------------------------------------------------
# Single-violation rejection
# ---------------------------------------------------------------------------


def test_ingested_doc_rejects_empty_text() -> None:
    bad = _ingested_doc_row(text="   ")
    frame = pd.DataFrame([bad])
    with pytest.raises(pa_errors.SchemaErrors) as exc:
        ps.validate_frame(ps.IngestedDocSchema, frame)
    assert "text" in str(exc.value)


def test_ingested_doc_rejects_missing_event_date_column() -> None:
    bad = _ingested_doc_row()
    bad.pop("event_date")
    frame = pd.DataFrame([bad])
    with pytest.raises(pa_errors.SchemaErrors) as exc:
        ps.validate_frame(ps.IngestedDocSchema, frame)
    assert "event_date" in str(exc.value)


def test_ingested_doc_rejects_bad_event_date_format() -> None:
    bad = _ingested_doc_row(event_date="2024/01/01")
    frame = pd.DataFrame([bad])
    with pytest.raises(pa_errors.SchemaErrors) as exc:
        ps.validate_frame(ps.IngestedDocSchema, frame)
    assert "event_date" in str(exc.value)


def test_ingested_doc_rejects_bad_text_hash() -> None:
    bad = _ingested_doc_row(text_hash="nothex")
    frame = pd.DataFrame([bad])
    with pytest.raises(pa_errors.SchemaErrors) as exc:
        ps.validate_frame(ps.IngestedDocSchema, frame)
    assert "text_hash" in str(exc.value)


def test_normalized_doc_rejects_invalid_mapped_label() -> None:
    bad = _normalized_doc_row(mapped_label="bullish")
    frame = pd.DataFrame([bad])
    with pytest.raises(pa_errors.SchemaErrors) as exc:
        ps.validate_frame(ps.NormalizedDocSchema, frame)
    assert "mapped_label" in str(exc.value)


def test_normalized_doc_rejects_negative_sample_weight() -> None:
    bad = _normalized_doc_row(sample_weight=-0.5)
    frame = pd.DataFrame([bad])
    with pytest.raises(pa_errors.SchemaErrors) as exc:
        ps.validate_frame(ps.NormalizedDocSchema, frame)
    assert "sample_weight" in str(exc.value)


def test_quality_passed_rejects_duplicate_text_hash() -> None:
    rows = [
        _normalized_doc_row(record_id=_hex16("a"), text_hash=_hex64("dup")),
        _normalized_doc_row(record_id=_hex16("b"), text_hash=_hex64("dup")),
    ]
    frame = pd.DataFrame(rows)
    with pytest.raises(pa_errors.SchemaErrors) as exc:
        ps.validate_frame(ps.QualityPassedRowSchema, frame)
    assert "text_hash" in str(exc.value)


def test_fold_row_rejects_unknown_partition() -> None:
    bad = _fold_row(split_tag="holdout")
    frame = pd.DataFrame([bad])
    with pytest.raises(pa_errors.SchemaErrors) as exc:
        ps.validate_frame(ps.FoldRowSchema, frame)
    assert "split_tag" in str(exc.value)


def test_event_row_rejects_unknown_kind() -> None:
    bad = _event_row(event_kind="op_ed")
    frame = pd.DataFrame([bad])
    with pytest.raises(pa_errors.SchemaErrors) as exc:
        ps.validate_frame(ps.EventRowSchema, frame)
    assert "event_kind" in str(exc.value)


def test_event_row_rejects_unknown_horizon() -> None:
    bad = _event_row(horizon=7)
    frame = pd.DataFrame([bad])
    with pytest.raises(pa_errors.SchemaErrors) as exc:
        ps.validate_frame(ps.EventRowSchema, frame)
    assert "horizon" in str(exc.value)


def test_event_row_rejects_short_prior_window_sha() -> None:
    bad = _event_row(prior_window_sha256="abcd")
    frame = pd.DataFrame([bad])
    with pytest.raises(pa_errors.SchemaErrors) as exc:
        ps.validate_frame(ps.EventRowSchema, frame)
    assert "prior_window_sha256" in str(exc.value)


def test_event_row_rejects_invalid_axis_stance() -> None:
    # axis_stance is one of the most-commonly-mislabeled output columns;
    # the closed three-class set must reject anything outside it.
    bad = _event_row(axis_stance="bullish")
    frame = pd.DataFrame([bad])
    with pytest.raises(pa_errors.SchemaErrors) as exc:
        ps.validate_frame(ps.EventRowSchema, frame)
    assert "axis_stance" in str(exc.value)


def test_event_row_rejects_invalid_direction_t1d() -> None:
    # direction_t1d is sign-of-return; the closed {-1, 0, 1} set must
    # reject any other value the upstream emitter could mis-produce.
    bad = _event_row(direction_t1d=2)
    frame = pd.DataFrame([bad])
    with pytest.raises(pa_errors.SchemaErrors) as exc:
        ps.validate_frame(ps.EventRowSchema, frame)
    assert "direction_t1d" in str(exc.value)


def test_event_row_rejects_negative_credibility_months_since_reversal() -> None:
    # The column is months-since; negative values are nonsense.
    bad = _event_row(credibility_months_since_reversal=-3)
    frame = pd.DataFrame([bad])
    with pytest.raises(pa_errors.SchemaErrors) as exc:
        ps.validate_frame(ps.EventRowSchema, frame)
    assert "credibility_months_since_reversal" in str(exc.value)


def test_linguistic_row_rejects_nan_named_topic() -> None:
    bad = _linguistic_row(topic_share_inflation=float("nan"))
    frame = pd.DataFrame([bad])
    with pytest.raises(pa_errors.SchemaErrors) as exc:
        ps.validate_frame(ps.LinguisticFeatureRowSchema, frame)
    assert "topic_share_inflation" in str(exc.value)


def test_linguistic_row_allows_nan_pivot_distance() -> None:
    row = _linguistic_row(pivot_distance=float("nan"))
    frame = pd.DataFrame([row])
    ps.validate_frame(ps.LinguisticFeatureRowSchema, frame)


def test_mp_surprise_rejects_invalid_methodology() -> None:
    bad = _mp_surprise_row(methodology="fudge")
    frame = pd.DataFrame([bad])
    with pytest.raises(pa_errors.SchemaErrors) as exc:
        ps.validate_frame(ps.MpSurpriseRowSchema, frame)
    assert "methodology" in str(exc.value)


def test_macro_state_rejects_bad_date_format() -> None:
    bad = _macro_state_row(as_of_date="not-a-date")
    frame = pd.DataFrame([bad])
    with pytest.raises(pa_errors.SchemaErrors) as exc:
        ps.validate_frame(ps.MacroStateRowSchema, frame)
    assert "as_of_date" in str(exc.value)


# ---------------------------------------------------------------------------
# Lazy-mode multi-violation
# ---------------------------------------------------------------------------


def test_lazy_mode_reports_all_event_row_violations() -> None:
    bad_kind = _event_row(event_kind="op_ed")
    bad_horizon = _event_row(horizon=7, text_hash=_hex64("evt-2"), document_id=_hex16("evt-2"))
    bad_sha = _event_row(
        prior_window_sha256="zzz",
        text_hash=_hex64("evt-3"),
        document_id=_hex16("evt-3"),
    )
    frame = pd.DataFrame([bad_kind, bad_horizon, bad_sha])
    with pytest.raises(pa_errors.SchemaErrors) as exc:
        ps.validate_frame(ps.EventRowSchema, frame)
    failure_cases = exc.value.failure_cases
    cols = set(failure_cases["column"].dropna().tolist())
    assert "event_kind" in cols
    assert "horizon" in cols
    assert "prior_window_sha256" in cols


# ---------------------------------------------------------------------------
# Skip env var
# ---------------------------------------------------------------------------


def test_skip_env_var_bypasses_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = _ingested_doc_row(text="")
    frame = pd.DataFrame([bad])
    monkeypatch.setenv(ps.FED_PULSE_SKIP_SCHEMA_VALIDATION, "1")
    # No exception: validation skipped.
    ps.validate_frame(ps.IngestedDocSchema, frame)


# ---------------------------------------------------------------------------
# Write-seam integration: event_dataset_builder
# ---------------------------------------------------------------------------


def test_event_dataset_builder_write_rejects_bad_row(tmp_path: Path) -> None:
    from app.data import event_dataset_builder as edb

    good = _event_row()
    bad = _event_row(
        event_kind="op_ed",
        text_hash=_hex64("evt-bad"),
        document_id=_hex16("evt-bad"),
    )
    frame = pd.DataFrame([good, bad])
    # Reorder to the canonical column order so the writer's later
    # df[list(COLUMN_ORDER)] step doesn't drop the new ones.
    frame = frame[[c for c in edb.COLUMN_ORDER if c in frame.columns]]

    with pytest.raises(pa_errors.SchemaErrors) as exc:
        edb.write_events_parquet(frame, tmp_path / "events.parquet")
    assert "event_kind" in str(exc.value)


def test_event_dataset_builder_write_accepts_good_frame(tmp_path: Path) -> None:
    from app.data import event_dataset_builder as edb

    good = _event_row()
    # The writer reindexes by COLUMN_ORDER inside the builder, so we only
    # need to deliver a frame whose required columns are present.
    frame = pd.DataFrame([good])
    frame = frame[[c for c in edb.COLUMN_ORDER if c in frame.columns]]
    out = tmp_path / "events.parquet"
    edb.write_events_parquet(frame, out)
    assert out.exists()


# ---------------------------------------------------------------------------
# Write-seam integration: linguistic features
# ---------------------------------------------------------------------------


def test_linguistic_write_rejects_bad_row(tmp_path: Path) -> None:
    from app.features import linguistic

    rows = [_linguistic_row(text_hash=_hex64("ok"))]
    rows.append(_linguistic_row(text_hash="not_hex", topic_share_growth=0.0))
    frame = pd.DataFrame(rows)
    with pytest.raises(pa_errors.SchemaErrors) as exc:
        linguistic.write_linguistic_parquet(frame, tmp_path / "linguistic.parquet")
    assert "text_hash" in str(exc.value)


def test_linguistic_write_accepts_good_frame(tmp_path: Path) -> None:
    from app.features import linguistic

    rows = [_linguistic_row(text_hash=_hex64(f"l-{i}")) for i in range(3)]
    frame = pd.DataFrame(rows)
    out = tmp_path / "linguistic.parquet"
    linguistic.write_linguistic_parquet(frame, out)
    assert out.exists()


# ---------------------------------------------------------------------------
# Write-seam integration: macro_state
# ---------------------------------------------------------------------------


def test_macro_state_write_rejects_bad_row(tmp_path: Path) -> None:
    from app.data import macro_state

    good = _macro_state_row()
    bad = _macro_state_row(as_of_date="January 2024")
    frame = pd.DataFrame([good, bad])
    with pytest.raises(pa_errors.SchemaErrors) as exc:
        macro_state.write_macro_state_parquet(frame, tmp_path / "macro_state.parquet")
    assert "as_of_date" in str(exc.value)


def test_macro_state_write_accepts_good_frame(tmp_path: Path) -> None:
    from app.data import macro_state

    frame = pd.DataFrame([_macro_state_row()])
    out = tmp_path / "macro_state.parquet"
    macro_state.write_macro_state_parquet(frame, out)
    assert out.exists()
