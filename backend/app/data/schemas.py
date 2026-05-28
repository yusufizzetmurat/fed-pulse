"""Pandera DataFrame schemas at every pipeline stage boundary.

Each schema models the row contract for one parquet emitter in the data
pipeline. The matching emitter calls ``<schema>.validate(frame)`` right
before writing parquet so a row that violates the contract raises at the
write site instead of propagating downstream.

Conventions:

- Every schema runs ``validate`` in lazy mode by default (every violation
  surfaces in a single ``SchemaErrors``).
- ``FED_PULSE_SKIP_SCHEMA_VALIDATION=1`` in the environment turns every
  validator into a no-op. Reserved for diagnostic re-runs against
  intentionally malformed inputs; the default behaviour is validation on.
- The schemas accept extra columns. The pipeline often carries optional
  enrichment columns (``multi_axis_extras``, ``provenance``, source-
  specific extras) that are not part of the headline contract.

The schemas guard the canonical parquet outputs:

- ``IngestedDocSchema``       — the row shape emitted by source ingestion
  (validated when the ingest stage hands rows downstream).
- ``NormalizedDocSchema``     — ``registry_normalized.parquet`` after
  label normalization (multi-axis labels nested under ``axes`` per the
  current write path; flat ``axis_*`` columns accepted when present).
- ``QualityPassedRowSchema``  — same row shape with deduped ``text_hash``.
- ``FoldRowSchema``           — ``splits_train_val_test.parquet`` with a
  ``split_tag`` partition tag.
- ``EventRowSchema``          — ``events.parquet`` / ``events_full.parquet``
  with the flat ``axis_*`` columns, credibility 4-vector, prior-window
  hash and per-horizon targets.
- ``LinguisticFeatureRowSchema`` — ``linguistic_features.parquet``: one
  row per ``text_hash`` with the 15 numeric feature columns.
- ``MpSurpriseRowSchema``     — ``mp_surprises.parquet`` per-FOMC
  monetary-policy surprise row.
- ``MacroStateRowSchema``     — ``macro_state.parquet`` per as-of-date
  macro snapshot row.

Schema-write coupling lives in the matching emitter module; the schemas
themselves are pure data contracts.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level shared validators
# ---------------------------------------------------------------------------


_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX16_RE = re.compile(r"^[0-9a-f]{16}$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


_ALLOWED_STANCE = {"hawkish", "dovish", "neutral"}
_ALLOWED_EVENT_KIND = {
    "statement",
    "minutes",
    "speech",
    "testimony",
    "press_conference",
    # Macro-release augmentation (CPI, NFP) — non-FOMC supervised
    # events that share the same forward-realised-vol target so the
    # vol-regime classifier sees more macro-context training rows.
    "macro_release",
}
_ALLOWED_DIRECTION = {-1, 0, 1}
_ALLOWED_HORIZON = {1, 5, 10, 30}
_ALLOWED_PARTITION = {"train", "val", "test", "excluded_from_training"}
_ALLOWED_METHODOLOGY = {"ois_proxy", "ff_futures"}


_SKIP_ENV_VAR = "FED_PULSE_SKIP_SCHEMA_VALIDATION"


def _schema_validation_disabled() -> bool:
    """Return True when the env flag asks the pipeline to skip validation.

    The flag exists for diagnostic re-runs against intentionally malformed
    inputs. Default behaviour is validation on; the flag is opt-in.
    """

    return os.environ.get(_SKIP_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}


def validate_frame(schema: DataFrameSchema, frame: pd.DataFrame) -> pd.DataFrame:
    """Run a pandera schema in lazy mode unless the skip flag is set.

    Returns the (possibly coerced) frame. When the skip flag is set the
    frame passes through untouched and a warning is logged so an
    operator inspecting the run does not silently assume validation
    succeeded. Otherwise pandera collects every violation across rows
    and columns and raises a single ``pandera.errors.SchemaErrors`` with
    the full failure table.
    """

    if _schema_validation_disabled():
        schema_name = getattr(schema, "name", None) or schema.__class__.__name__
        LOGGER.warning(
            "schema validation bypassed via FED_PULSE_SKIP_SCHEMA_VALIDATION "
            "for schema=%s rows=%d cols=%d",
            schema_name,
            len(frame),
            len(frame.columns),
        )
        return frame
    return schema.validate(frame, lazy=True)


def _hex64(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX64_RE.match(value))


def _hex16(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX16_RE.match(value))


def _is_iso_date(value: Any) -> bool:
    return isinstance(value, str) and bool(_ISO_DATE_RE.match(value))


def _is_iso_ts(value: Any) -> bool:
    return isinstance(value, str) and bool(_ISO_TS_RE.match(value))


def _non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _nullable_in_set(allowed: set[Any]) -> Any:
    def _check(series: pd.Series) -> pd.Series:
        return series.isna() | series.isin(allowed)

    return _check


def _nullable_str_or_none(series: pd.Series) -> pd.Series:
    return series.map(lambda v: v is None or isinstance(v, str))


def _nullable_finite_in_range(low: float, high: float) -> Any:
    """Vectorised range check for regression-typed nullable columns.

    NaN passes through; pandera's ``nullable=True`` flag handles None /
    NaN rows before this fires, but the check stays NaN-tolerant in case
    a non-nullable column reuses the same helper. Non-numeric values
    fail. The check runs through ``pd.to_numeric(..., errors="coerce")``
    so the whole column is one vectorised pass instead of a per-row
    Python callback.
    """

    def _check(series: pd.Series) -> pd.Series:
        coerced = pd.to_numeric(series, errors="coerce")
        non_numeric_now = coerced.isna() & series.notna()
        in_range = coerced.between(low, high)
        # Accept: NaN-on-input (nullable contract) OR in-range numeric.
        # Reject: non-numeric strings (non_numeric_now == True).
        return (series.isna() | in_range) & ~non_numeric_now

    return _check


# ---------------------------------------------------------------------------
# Stage 1 — Ingested document
# ---------------------------------------------------------------------------


_INGESTED_DOC_COLUMNS: dict[str, Column] = {
    "record_id": Column(
        str,
        Check(lambda s: s.map(_non_empty_str), element_wise=False),
        nullable=False,
        required=True,
        description="Deterministic 16-char hex id derived from source|source_record_id|event_date.",
    ),
    "source": Column(
        str,
        Check(lambda s: s.map(_non_empty_str), element_wise=False),
        nullable=False,
        required=True,
    ),
    "source_record_id": Column(
        str,
        Check(lambda s: s.map(_non_empty_str), element_wise=False),
        nullable=False,
        required=True,
    ),
    "document_type": Column(str, nullable=False, required=True),
    "event_date": Column(
        str,
        Check(
            lambda s: s.map(_is_iso_date),
            element_wise=False,
            error="event_date must be ISO YYYY-MM-DD",
        ),
        nullable=False,
        required=True,
    ),
    "text": Column(
        str,
        Check(
            lambda s: s.map(_non_empty_str),
            element_wise=False,
            error="text must be a non-empty string",
        ),
        nullable=False,
        required=True,
    ),
    "label_origin": Column(str, nullable=True, required=True),
    "license_scope": Column(str, nullable=False, required=True),
    "citation_ref": Column(str, nullable=False, required=True),
    "ingested_at_utc": Column(
        str,
        Check(lambda s: s.map(_is_iso_ts), element_wise=False),
        nullable=False,
        required=True,
    ),
    "text_hash": Column(
        str,
        Check(
            lambda s: s.map(_hex64),
            element_wise=False,
            error="text_hash must be 64-char lower-hex sha256",
        ),
        nullable=False,
        required=True,
    ),
}


IngestedDocSchema = DataFrameSchema(
    _INGESTED_DOC_COLUMNS,
    name="IngestedDocSchema",
    strict=False,
    coerce=False,
    description="Row contract emitted by app.data.ingest_sources before downstream stages.",
)


# ---------------------------------------------------------------------------
# Stage 2 — Normalized document with multi-axis labels
# ---------------------------------------------------------------------------


def _axes_stance_ok(value: Any) -> bool:
    return value is None or value in _ALLOWED_STANCE


def _axes_factor_ok(value: Any) -> bool:
    """``factor`` stays numeric (GSS factor decomposition)."""
    if value is None:
        return True
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _axes_certainty_ok(value: Any) -> bool:
    """Audit Tier 1.8: ``certainty`` may carry either a regression
    value (per data/schema/labels.yaml) or a string label
    (gtfintechlab's ``certain`` / ``uncertain`` category)."""
    if value is None:
        return True
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return isinstance(value, str)


def _axes_time_ok(value: Any) -> bool:
    """``time`` is the gtfintechlab-derived horizon label
    (``forward looking`` / ``not forward looking``); string-only."""
    return value is None or isinstance(value, str)


def _axes_topic_ok(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _axes_dict_ok(series: pd.Series) -> pd.Series:
    def _ok(value: Any) -> bool:
        if value is None:
            return True
        if not isinstance(value, dict):
            return False
        return (
            _axes_stance_ok(value.get("stance"))
            and _axes_factor_ok(value.get("factor"))
            and _axes_certainty_ok(value.get("certainty"))
            and _axes_time_ok(value.get("time"))
            and _axes_topic_ok(value.get("topic"))
        )

    return series.map(_ok)


_NORMALIZED_DOC_COLUMNS: dict[str, Column] = {
    **_INGESTED_DOC_COLUMNS,
    "mapped_label": Column(
        str,
        Check(_nullable_in_set(_ALLOWED_STANCE)),
        nullable=True,
        required=True,
        description="Normalized hawkish/dovish/neutral label; None for unlabeled rows.",
    ),
    "sample_weight": Column(
        float,
        Check.greater_than_or_equal_to(0.0),
        nullable=False,
        required=True,
        coerce=True,
    ),
    "axes": Column(
        checks=Check(_axes_dict_ok, element_wise=False),
        nullable=True,
        required=True,
        description="Multi-axis label payload {stance, factor, certainty, topic}.",
    ),
    # Optional flat axis columns surface on emitters that flatten the axes
    # dict (e.g. event_dataset_builder writes flat columns). Marked
    # required=False so this schema validates both representations.
    "axis_stance": Column(
        checks=Check(_nullable_in_set(_ALLOWED_STANCE)),
        nullable=True,
        required=False,
    ),
    "axis_time": Column(
        checks=Check(_nullable_finite_in_range(-10.0, 10.0)),
        nullable=True,
        required=False,
    ),
    "axis_certainty": Column(
        checks=Check(_nullable_finite_in_range(0.0, 1.0)),
        nullable=True,
        required=False,
    ),
    "axis_factor": Column(
        checks=Check(_nullable_finite_in_range(-1.0, 1.0)),
        nullable=True,
        required=False,
    ),
    "axis_topic": Column(
        checks=Check(_nullable_str_or_none, element_wise=False),
        nullable=True,
        required=False,
    ),
}


NormalizedDocSchema = DataFrameSchema(
    _NORMALIZED_DOC_COLUMNS,
    name="NormalizedDocSchema",
    strict=False,
    coerce=False,
    description="Row contract after label normalization (registry_normalized.parquet).",
)


# ---------------------------------------------------------------------------
# Stage 3 — Quality-passed row (deduped on text_hash)
# ---------------------------------------------------------------------------


QualityPassedRowSchema = DataFrameSchema(
    _NORMALIZED_DOC_COLUMNS,
    name="QualityPassedRowSchema",
    strict=False,
    coerce=False,
    unique=["text_hash"],
    description=(
        "Quality-passed registry row. Inherits NormalizedDocSchema and "
        "asserts text_hash uniqueness."
    ),
)


# ---------------------------------------------------------------------------
# Stage 4 — Walk-forward fold row (splits_train_val_test.parquet)
# ---------------------------------------------------------------------------


_FOLD_ROW_COLUMNS: dict[str, Column] = {
    **_NORMALIZED_DOC_COLUMNS,
    "split_tag": Column(
        str,
        Check.isin(_ALLOWED_PARTITION),
        nullable=False,
        required=True,
        description="Walk-forward partition tag: train | val | test | excluded_from_training.",
    ),
}


FoldRowSchema = DataFrameSchema(
    _FOLD_ROW_COLUMNS,
    name="FoldRowSchema",
    strict=False,
    coerce=False,
    description="One row per (registry row, fold partition).",
)


# ---------------------------------------------------------------------------
# Stage 5 — Event-row dataset (events.parquet / events_full.parquet)
# ---------------------------------------------------------------------------


def _prior_bars_json_ok(series: pd.Series) -> pd.Series:
    def _ok(value: Any) -> bool:
        if value is None:
            return False
        return isinstance(value, str) and value.startswith(("[", "{"))

    return series.map(_ok)


_EVENT_ROW_COLUMNS: dict[str, Column] = {
    "event_date": Column(
        str,
        Check(lambda s: s.map(_is_iso_date), element_wise=False),
        nullable=False,
        required=True,
    ),
    "event_kind": Column(
        str,
        Check.isin(_ALLOWED_EVENT_KIND),
        nullable=False,
        required=True,
    ),
    "document_id": Column(
        str,
        Check(lambda s: s.map(_hex16), element_wise=False),
        nullable=False,
        required=True,
    ),
    "text_hash": Column(
        str,
        Check(lambda s: s.map(_hex64), element_wise=False),
        nullable=False,
        required=True,
    ),
    "source": Column(str, nullable=False, required=True),
    "as_of_ts": Column(
        str,
        Check(lambda s: s.map(_is_iso_ts), element_wise=False),
        nullable=False,
        required=True,
    ),
    "text": Column(
        str,
        Check(lambda s: s.map(_non_empty_str), element_wise=False),
        nullable=False,
        required=True,
    ),
    "token_count": Column(
        int,
        Check.greater_than_or_equal_to(0),
        nullable=False,
        required=True,
        coerce=True,
    ),
    "axis_stance": Column(
        checks=Check(_nullable_in_set(_ALLOWED_STANCE)),
        nullable=True,
        required=True,
    ),
    "axis_time": Column(
        checks=Check(_nullable_finite_in_range(-10.0, 10.0)),
        nullable=True,
        required=True,
    ),
    "axis_certainty": Column(
        checks=Check(_nullable_finite_in_range(0.0, 1.0)),
        nullable=True,
        required=True,
    ),
    "axis_factor": Column(
        checks=Check(_nullable_finite_in_range(-1.0, 1.0)),
        nullable=True,
        required=True,
    ),
    "axis_topic": Column(
        checks=Check(_nullable_str_or_none, element_wise=False),
        nullable=True,
        required=True,
    ),
    # String indicators lifted off ``multi_axis_extras`` (only the
    # gtfintechlab cross-bank corpora ship them today). Required=False
    # so older events.parquet files validate; the loader treats absent
    # columns as ``None`` everywhere.
    "axis_time_label": Column(
        checks=Check(_nullable_in_set({"forward looking", "not forward looking"})),
        nullable=True,
        required=False,
    ),
    "axis_certain_label": Column(
        checks=Check(_nullable_in_set({"certain", "uncertain"})),
        nullable=True,
        required=False,
    ),
    "credibility_drift_score": Column(float, nullable=False, required=True, coerce=True),
    "credibility_realized_vs_stated_gap": Column(
        float, nullable=False, required=True, coerce=True
    ),
    "credibility_market_implied_gap": Column(
        float, nullable=False, required=True, coerce=True
    ),
    "credibility_months_since_reversal": Column(
        int,
        Check.greater_than_or_equal_to(0),
        nullable=False,
        required=True,
        coerce=True,
    ),
    "prior_window_sha256": Column(
        str,
        Check(lambda s: s.map(_hex64), element_wise=False),
        nullable=False,
        required=True,
    ),
    "prior_bars_json": Column(
        str,
        Check(_prior_bars_json_ok, element_wise=False),
        nullable=False,
        required=True,
    ),
    "asset_symbol": Column(
        str,
        Check(lambda s: s.map(_non_empty_str), element_wise=False),
        nullable=False,
        required=True,
    ),
    "horizon": Column(
        int,
        Check.isin(_ALLOWED_HORIZON),
        nullable=False,
        required=True,
        coerce=True,
    ),
    "realized_return": Column(float, nullable=True, required=True, coerce=True),
    "abnormal_return": Column(float, nullable=True, required=True, coerce=True),
    "alpha": Column(float, nullable=True, required=True, coerce=True),
    "beta": Column(float, nullable=True, required=True, coerce=True),
    "direction_t1d": Column(
        checks=Check(_nullable_in_set(_ALLOWED_DIRECTION)),
        nullable=True,
        required=True,
    ),
    "volatility_shift": Column(float, nullable=True, required=True, coerce=True),
    # Phase 9 V2 (#195) target: forward 10-trading-day realised vol of
    # log returns. Nullable for events too close to the end of the
    # price series; required=False so older events.parquet files (pre
    # Phase 9 V2) validate without the column present.
    "forward_realized_vol_10d": Column(
        float, nullable=True, required=False, coerce=True
    ),
    # #236 GARCH(1,1)-residual decomposition. Nullable for events whose
    # strict-prior window is shorter than ``MIN_FIT_RETURNS`` (~252 td)
    # or the QMLE fit failed to converge. required=False so older
    # events.parquet files (pre #236) validate without the columns.
    "forward_realized_vol_10d_garch_baseline": Column(
        float, nullable=True, required=False, coerce=True
    ),
    "forward_realized_vol_10d_garch_residual": Column(
        float, nullable=True, required=False, coerce=True
    ),
    "concurrent_macro_release": Column(
        bool, nullable=False, required=True, coerce=True
    ),
    "intra_meeting_stance_shift": Column(float, nullable=True, required=True, coerce=True),
    "intra_meeting_certainty_shift": Column(
        float, nullable=True, required=True, coerce=True
    ),
    "intra_meeting_factor_shift": Column(float, nullable=True, required=True, coerce=True),
    # ----- #291 rates-complex forward targets (raw bps) -----
    # required=False so older events.parquet files (pre #291) validate
    # without the columns present. The pipeline emits None when the
    # rates panel is unavailable so the schema stays nullable.
    "yield_2y_change_5d": Column(float, nullable=True, required=False, coerce=True),
    "yield_5y_change_5d": Column(float, nullable=True, required=False, coerce=True),
    "terminal_rate_change_5d": Column(float, nullable=True, required=False, coerce=True),
    # ----- #291 pre-meeting expectation features (strict-backward at t-1) -----
    "pre_meeting_yield_1y": Column(float, nullable=True, required=False, coerce=True),
    "pre_meeting_yield_2y": Column(float, nullable=True, required=False, coerce=True),
    "pre_meeting_yield_5y": Column(float, nullable=True, required=False, coerce=True),
    "pre_meeting_yield_10y": Column(float, nullable=True, required=False, coerce=True),
    "pre_meeting_slope_10y_2y": Column(float, nullable=True, required=False, coerce=True),
    "pre_meeting_slope_10y_3m": Column(float, nullable=True, required=False, coerce=True),
    "pre_meeting_trailing_2y_yield_change_5d_bps": Column(
        float, nullable=True, required=False, coerce=True
    ),
    "pre_meeting_implied_next_move_bps": Column(
        float, nullable=True, required=False, coerce=True
    ),
    "pre_meeting_implied_hike_prob": Column(
        float, nullable=True, required=False, coerce=True
    ),
    "pre_meeting_implied_cut_prob": Column(
        float, nullable=True, required=False, coerce=True
    ),
    "pre_meeting_implied_pause_prob": Column(
        float, nullable=True, required=False, coerce=True
    ),
    "pre_meeting_days_since_last_rate_change": Column(
        float,
        nullable=True,
        required=False,
        coerce=True,
        description=(
            "Calendar-day gap since the last DFEDTARU step change at t-1; "
            "stored as float so pandas keeps the nullable semantics across "
            "pyarrow round-trips (int columns cannot hold NaN)."
        ),
    ),
    # ---- #443 statement-delta (redline) text spans + embedding ----
    # required=False so pre-#443 events.parquet files validate clean.
    "statement_delta_inserted": Column(
        str, nullable=True, required=False
    ),
    "statement_delta_deleted": Column(
        str, nullable=True, required=False
    ),
    "statement_delta_substituted_pairs": Column(
        str, nullable=True, required=False
    ),
    "statement_delta_embedding": Column(
        object, nullable=True, required=False
    ),
    # ---- #444 vote tally + dissent ----
    "votes_for": Column(
        float, nullable=True, required=False, coerce=True
    ),
    "votes_against": Column(
        float, nullable=True, required=False, coerce=True
    ),
    "dissent_count": Column(
        float, nullable=True, required=False, coerce=True
    ),
    "is_unanimous": Column(
        checks=Check(_nullable_in_set({True, False})),
        nullable=True,
        required=False,
    ),
    "dissent_direction": Column(
        checks=Check(_nullable_in_set({"hawkish_dissent", "dovish_dissent"})),
        nullable=True,
        required=False,
    ),
}


EventRowSchema = DataFrameSchema(
    _EVENT_ROW_COLUMNS,
    name="EventRowSchema",
    strict=False,
    coerce=False,
    description="One row per (event_date, event_kind, source, asset_symbol, horizon).",
)


# ---------------------------------------------------------------------------
# Stage 6 — Linguistic feature row (linguistic_features.parquet)
# ---------------------------------------------------------------------------


_LING_NUMERIC_FIELDS: tuple[str, ...] = (
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
)


_LINGUISTIC_COLUMNS: dict[str, Column] = {
    "text_hash": Column(
        str,
        Check(lambda s: s.map(_hex64), element_wise=False),
        nullable=False,
        required=True,
        unique=True,
    ),
}
for _field in _LING_NUMERIC_FIELDS:
    _LINGUISTIC_COLUMNS[_field] = Column(
        float,
        Check(lambda s: s.map(lambda v: v is not None and float(v) == float(v))),
        nullable=False,
        required=True,
        coerce=True,
    )
_LINGUISTIC_COLUMNS["pivot_distance"] = Column(
    float,
    Check(_nullable_finite_in_range(0.0, 1.0)),
    nullable=True,
    required=True,
    coerce=True,
)


LinguisticFeatureRowSchema = DataFrameSchema(
    _LINGUISTIC_COLUMNS,
    name="LinguisticFeatureRowSchema",
    strict=False,
    coerce=False,
    description="One row per text_hash with 14 finite required features + nullable pivot_distance (15 columns total).",
)


# ---------------------------------------------------------------------------
# Stage 7 — Monetary-policy surprise row (mp_surprises.parquet)
# ---------------------------------------------------------------------------


_MP_SURPRISE_COLUMNS: dict[str, Column] = {
    "event_date": Column(
        str,
        Check(lambda s: s.map(_is_iso_date), element_wise=False),
        nullable=False,
        required=True,
    ),
    "meeting_id": Column(
        int,
        Check.greater_than_or_equal_to(0),
        nullable=False,
        required=True,
        coerce=True,
    ),
    "ff_target_prior": Column(float, nullable=True, required=True, coerce=True),
    "ff_target_after": Column(float, nullable=True, required=True, coerce=True),
    "mp_surprise_level": Column(float, nullable=True, required=True, coerce=True),
    "mp_surprise_path_factor": Column(
        float, nullable=True, required=True, coerce=True
    ),
    "pre_event_curve": Column(
        str,
        Check(_prior_bars_json_ok, element_wise=False),
        nullable=False,
        required=True,
    ),
    "post_event_curve": Column(
        str,
        Check(_prior_bars_json_ok, element_wise=False),
        nullable=False,
        required=True,
    ),
    "fed_info_factor": Column(float, nullable=True, required=True, coerce=True),
    "is_intermeeting": Column(bool, nullable=False, required=True, coerce=True),
    "methodology": Column(
        str,
        Check.isin(_ALLOWED_METHODOLOGY),
        nullable=False,
        required=True,
    ),
    "data_version": Column(
        str,
        Check(lambda s: s.map(_non_empty_str), element_wise=False),
        nullable=False,
        required=True,
    ),
}


MpSurpriseRowSchema = DataFrameSchema(
    _MP_SURPRISE_COLUMNS,
    name="MpSurpriseRowSchema",
    strict=False,
    coerce=False,
    description="One row per FOMC meeting with reconstructed surprise components.",
)


# ---------------------------------------------------------------------------
# Stage 8 — Macro state row (macro_state.parquet)
# ---------------------------------------------------------------------------


_MACRO_STATE_COLUMNS: dict[str, Column] = {
    "as_of_date": Column(
        str,
        Check(lambda s: s.map(_is_iso_date), element_wise=False),
        nullable=False,
        required=True,
    ),
    "unrate": Column(float, nullable=True, required=True, coerce=True),
    "cpi_yoy": Column(float, nullable=True, required=True, coerce=True),
    "core_pce_yoy": Column(float, nullable=True, required=True, coerce=True),
    "ism_proxy": Column(float, nullable=True, required=True, coerce=True),
    "payems_mom": Column(float, nullable=True, required=True, coerce=True),
    "rsafs_mom": Column(float, nullable=True, required=True, coerce=True),
    # Rates + financial-conditions panel. All level columns, nullable so
    # the contract degrades cleanly when an as-of date sits inside a
    # holiday cluster where no upstream observation is published.
    "treas_10y": Column(float, nullable=True, required=True, coerce=True),
    "slope_10y_2y": Column(float, nullable=True, required=True, coerce=True),
    "slope_10y_3m": Column(float, nullable=True, required=True, coerce=True),
    "hy_oas": Column(float, nullable=True, required=True, coerce=True),
    "nfci": Column(float, nullable=True, required=True, coerce=True),
    "tips_10y_real": Column(float, nullable=True, required=True, coerce=True),
    "ism_proxy_source": Column(
        str,
        Check(lambda s: s.map(_non_empty_str), element_wise=False),
        nullable=False,
        required=True,
    ),
}


MacroStateRowSchema = DataFrameSchema(
    _MACRO_STATE_COLUMNS,
    name="MacroStateRowSchema",
    strict=False,
    coerce=False,
    description="One row per as-of-date snapshot of the macro state vector.",
)


__all__ = (
    "FED_PULSE_SKIP_SCHEMA_VALIDATION",
    "EventRowSchema",
    "FoldRowSchema",
    "IngestedDocSchema",
    "LinguisticFeatureRowSchema",
    "MacroStateRowSchema",
    "MpSurpriseRowSchema",
    "NormalizedDocSchema",
    "QualityPassedRowSchema",
    "validate_frame",
)


FED_PULSE_SKIP_SCHEMA_VALIDATION = _SKIP_ENV_VAR
