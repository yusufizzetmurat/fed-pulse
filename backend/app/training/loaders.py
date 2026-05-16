from __future__ import annotations

import csv
import datetime
import json
import warnings
from pathlib import Path
from typing import Any, Literal, Sequence

import torch

from app.config import DATA_DIR
from app.evaluation.metrics import TrainingDataSourceSummary
from app.models.config import (
    DEFAULT_CLOSE_SCALE,
    DEFAULT_DATA_DIR,
    RICH_LINGUISTIC_DIM,
    SEQUENCE_LENGTH,
    FeatureVector,
)

# Ordered linguistic-feature columns expected on
# ``linguistic_features.parquet``. The order mirrors the
# :class:`app.features.linguistic.LinguisticVector` dataclass so the
# 15-dim slice the forecaster sees lines up byte-for-byte with the
# emitting module's documented layout.
_LINGUISTIC_FEATURE_COLUMNS: tuple[str, ...] = (
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
    "pivot_distance",
)

# Numeric encoding of ``axis_stance`` used as the sentiment proxy on the
# training-package path. Mirrors the encoding in
# ``app.data.event_dataset_builder._STANCE_ENCODING`` so the forecaster
# sees the same hawkish/dovish/neutral scale that drives the
# ``intra_meeting_stance_shift`` column.
_STANCE_SENTIMENT_ENCODING: dict[str, float] = {
    "hawkish": 1.0,
    "dovish": -1.0,
    "neutral": 0.0,
}

# Rows whose split-tag column carries this value are dropped before
# sequence construction. The Phase 8 packages currently emit
# ``{train, val, test}`` only; the sentinel is reserved for future
# packages that materialise an explicit holdout-from-training partition.
# Only rows partitioned as the training fold itself feed the loss. The
# walk-forward contract treats val + test as forward-looking holdouts;
# either must never appear on the training side. The explicit
# excluded_from_training sentinel from the data contract is also
# treated as out-of-training when present.
_TRAINING_PARTITION = "train"
_NON_TRAINING_SPLIT_TAGS = frozenset({"val", "test", "excluded_from_training"})

# Supported target-frame derivations for the training-package loader.
#
# ``event_study`` is the default and the production target. It derives
# the synthesised target close from ``abnormal_return`` (market-model
# residual against the trailing 252-day window written by
# ``app.data.event_dataset_builder``) and the target volatility from
# ``prior_bars[-1].vol_5d + volatility_shift`` (the post-event 10d
# realised vol). Both quantities carry genuine temporal signal that the
# forecaster has to actually learn rather than copy.
#
# ``realized_return`` reproduces the pre-fix behaviour: the close target
# becomes ``prior_bars[-1].close * (1 + realized_return)`` and the
# volatility target is a literal copy of ``prior_bars[-1].vol_5d``. The
# volatility column is then a trivial identity over the input window;
# linear-decomposition models (DLinear) win the volatility-RMSE column
# at the identity task by construction. The mode is preserved only for
# back-compat smoke tests that need to reproduce earlier sweep numbers.
TargetMode = Literal["event_study", "realized_return"]
_VALID_TARGET_MODES: frozenset[str] = frozenset({"event_study", "realized_return"})
DEFAULT_TARGET_MODE: TargetMode = "event_study"


def _coerce_finite_float(value: Any) -> float | None:
    """Return ``float(value)`` when the result is finite, else ``None``.

    Parquet nulls materialise as ``float('nan')`` through pandas; the
    event-study target derivation must surface those as ``None`` so the
    fallback path can kick in without comparing NaN.
    """

    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    # NaN comparison: ``x != x`` is the canonical Python NaN check.
    if result != result:
        return None
    return result


def _extract_required_float(record: dict[str, Any], keys: Sequence[str]) -> float:
    for key in keys:
        if key in record and record[key] not in {None, ""}:
            return float(record[key])
    raise ValueError(f"Missing required numeric field from keys: {', '.join(keys)}")


def build_feature_vectors(
    records: Sequence[dict[str, Any]],
    sentiment_score: float | None = None,
    document_date: str | None = None,
    text_embedding: list[float] | None = None,
) -> list[FeatureVector]:
    vectors: list[FeatureVector] = []
    previous_close: float | None = None
    previous_volatility: float | None = None

    parsed_doc_date: datetime.date | None = None
    if document_date:
        parsed_doc_date = datetime.date.fromisoformat(document_date)

    sorted_records = sorted(records, key=lambda item: str(item.get("date", item.get("timestamp", ""))))
    for record in sorted_records:
        date_value = str(record.get("date", record.get("timestamp", "")))
        if not date_value:
            continue

        elapsed_time = 0.0
        if parsed_doc_date is not None:
            record_date = datetime.date.fromisoformat(date_value[:10])
            elapsed_time = float((record_date - parsed_doc_date).days)

        close_value = _extract_required_float(record, ("close", "market_close"))
        volatility_value = _extract_required_float(
            record,
            ("volatility_5d", "market_volatility", "volatility"),
        )
        row_sentiment = float(record.get("sentiment_score", sentiment_score if sentiment_score is not None else 0.0))
        row_embedding = record.get("text_embedding") if isinstance(record.get("text_embedding"), list) else text_embedding
        vectors.append(
            FeatureVector.from_market_state(
                date=date_value,
                sentiment_score=row_sentiment,
                market_close=close_value,
                market_volatility=volatility_value,
                previous_close=previous_close,
                previous_volatility=previous_volatility,
                elapsed_time=elapsed_time,
                text_embedding=row_embedding,
            )
        )
        previous_close = close_value
        previous_volatility = volatility_value

    return vectors


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("records", "rows", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _load_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _is_record_mapping_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def _extract_record_groups(payload: Any) -> list[list[dict[str, Any]]]:
    if _is_record_mapping_list(payload):
        if payload and any(any(key in item for key in ("records", "rows", "data", "items")) for item in payload):
            nested_groups: list[list[dict[str, Any]]] = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                nested_groups.extend(_extract_record_groups(item))
            return nested_groups or [payload]
        return [payload]

    if isinstance(payload, dict):
        for key in ("sequences", "series", "groups"):
            nested = payload.get(key)
            if isinstance(nested, list):
                groups: list[list[dict[str, Any]]] = []
                for entry in nested:
                    groups.extend(_extract_record_groups(entry))
                if groups:
                    return groups

        for key in ("records", "rows", "data", "items"):
            nested = payload.get(key)
            if nested is not None and _is_record_mapping_list(nested):
                return [list(nested)]

    return []


def _load_record_groups(path: Path) -> tuple[list[list[dict[str, Any]]], str]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _extract_record_groups(payload), "json"
    if path.suffix == ".jsonl":
        return [_load_jsonl_records(path)], "jsonl"
    if path.suffix == ".csv":
        return [_load_csv_records(path)], "csv"
    return [], path.suffix.lstrip(".") or "unknown"


def inspect_training_data_sources(
    data_dir: str | Path | None = None,
) -> tuple[list[list[FeatureVector]], list[TrainingDataSourceSummary]]:
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    if not root.exists():
        return [], []

    sequences: list[list[FeatureVector]] = []
    summaries: list[TrainingDataSourceSummary] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue

        try:
            groups, format_name = _load_record_groups(path)
            if not groups:
                summaries.append(
                    TrainingDataSourceSummary(
                        path=path,
                        format=format_name,
                        record_groups=0,
                        records=0,
                        vectors=0,
                        usable_sequences=0,
                        status="ignored",
                        message="No trainable market-record groups detected.",
                    )
                )
                continue

            record_count = sum(len(group) for group in groups)
            vectors_for_path = [build_feature_vectors(group) for group in groups]
            usable = [vector_group for vector_group in vectors_for_path if len(vector_group) >= SEQUENCE_LENGTH + 1]
            sequences.extend(usable)
            summaries.append(
                TrainingDataSourceSummary(
                    path=path,
                    format=format_name,
                    record_groups=len(groups),
                    records=record_count,
                    vectors=sum(len(group) for group in vectors_for_path),
                    usable_sequences=len(usable),
                    status="usable" if usable else "insufficient",
                    message=(
                        "Usable training sequences detected."
                        if usable
                        else f"Need at least {SEQUENCE_LENGTH + 1} usable rows per sequence."
                    ),
                )
            )
        except Exception as exc:
            summaries.append(
                TrainingDataSourceSummary(
                    path=path,
                    format=path.suffix.lstrip(".") or "unknown",
                    record_groups=0,
                    records=0,
                    vectors=0,
                    usable_sequences=0,
                    status="error",
                    message=str(exc),
                )
            )
            continue

    return sequences, summaries


def load_training_sequences_from_data(data_dir: str | Path | None = None) -> list[list[FeatureVector]]:
    sequences, _ = inspect_training_data_sources(data_dir)
    return sequences


def _stance_to_sentiment(value: Any) -> float:
    """Map an ``axis_stance`` value to the sentiment proxy used by the forecaster.

    ``hawkish`` -> ``+1.0``, ``dovish`` -> ``-1.0``, ``neutral`` or any
    other / missing value -> ``0.0``. The forecaster has historically
    consumed FinBERT-style sentiment scores in ``[-1, 1]``; encoding the
    discrete stance label onto the same scale keeps the recurrent core's
    sentiment channel comparable across the legacy market-JSONL path and
    the Phase 8 event-row path.
    """

    if value is None:
        return 0.0
    # Parquet nulls materialise as float NaN through pandas; treat as neutral.
    if isinstance(value, float) and value != value:
        return 0.0
    if isinstance(value, str):
        return _STANCE_SENTIMENT_ENCODING.get(value.strip().lower(), 0.0)
    return 0.0


def _parse_prior_bars(payload: Any) -> list[dict[str, Any]]:
    """Decode ``prior_bars_json`` (string or already-list) into bar dicts."""

    if payload is None:
        return []
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return []
        decoded = json.loads(text)
    else:
        decoded = payload
    if not isinstance(decoded, list):
        return []
    bars: list[dict[str, Any]] = []
    for entry in decoded:
        if isinstance(entry, dict):
            bars.append(entry)
    return bars


def _bars_to_feature_vectors(
    bars: Sequence[dict[str, Any]],
    *,
    event_date: datetime.date,
    sentiment_score: float,
) -> list[FeatureVector]:
    """Convert a prior-window bar list into chronological ``FeatureVector`` rows."""

    vectors: list[FeatureVector] = []
    previous_close: float | None = None
    previous_volatility: float | None = None
    for bar in bars:
        date_value = str(bar.get("date", ""))
        if not date_value:
            continue
        try:
            bar_date = datetime.date.fromisoformat(date_value[:10])
        except ValueError:
            continue
        close_value = float(bar.get("close", 0.0))
        volatility_value = float(bar.get("vol_5d", 0.0))
        elapsed_time = float((bar_date - event_date).days)
        vectors.append(
            FeatureVector.from_market_state(
                date=date_value,
                sentiment_score=sentiment_score,
                market_close=close_value,
                market_volatility=volatility_value,
                previous_close=previous_close,
                previous_volatility=previous_volatility,
                elapsed_time=elapsed_time,
            )
        )
        previous_close = close_value
        previous_volatility = volatility_value
    return vectors


def _append_event_day_target(
    vectors: list[FeatureVector],
    *,
    event_date: datetime.date,
    realized_return: float | None,
    realized_date: str | None,
    sentiment_score: float,
    abnormal_return: float | None = None,
    volatility_shift: float | None = None,
    target_mode: TargetMode = DEFAULT_TARGET_MODE,
) -> None:
    """Append a single event-day target frame for the supervised window.

    The Phase 8 ``events.parquet`` carries 20 trading-day prior bars but
    no event-day bar; the downstream training-tensor builder needs a
    ``SEQUENCE_LENGTH + 1`` row to compute one supervised (window,
    target) pair per event.

    Two derivations are supported via ``target_mode``:

    - ``event_study`` (default): the target close projects the last
      prior bar via ``close * (1 + abnormal_return)`` -- the market-
      model residual after removing the trailing-252-day SPX beta /
      alpha. The target volatility is ``prior_bars[-1].vol_5d +
      volatility_shift``, i.e. the actual post-event 10d realised vol
      reconstructed from the prior vol plus the shift. Both quantities
      have to be learnt from the prior window; neither is a literal
      copy of an input feature.
    - ``realized_return``: legacy back-compat path. Projects the close
      via ``close * (1 + realized_return)`` and re-uses the last prior
      bar's ``vol_5d`` as the volatility target. The volatility column
      is then trivially identical to the last input volatility, which
      gives linear-decomposition models an artefactual edge on the
      volatility-RMSE column.

    When the event-study fields are NaN / missing the loader emits a
    ``UserWarning`` and falls back to the realized_return formula for
    that row so a downstream sweep against a package with broken target
    columns surfaces the gap immediately rather than silently training
    on the legacy target.
    """

    if not vectors:
        return
    last = vectors[-1]
    base_close = float(last.market_close)
    base_volatility = float(last.market_volatility)

    if target_mode == "event_study":
        if abnormal_return is None:
            warnings.warn(
                "event-study target requested but abnormal_return is missing; "
                "falling back to realized_return for this event-day target.",
                UserWarning,
                stacklevel=2,
            )
            close_shift = realized_return
        else:
            close_shift = abnormal_return

        if volatility_shift is None:
            warnings.warn(
                "event-study target requested but volatility_shift is missing; "
                "falling back to prior_bars[-1].vol_5d for this target.",
                UserWarning,
                stacklevel=2,
            )
            vol_offset: float = 0.0
        else:
            vol_offset = float(volatility_shift)
    else:
        close_shift = realized_return
        vol_offset = 0.0

    if close_shift is None or base_close <= 0.0:
        target_close = base_close
    else:
        target_close = base_close * (1.0 + float(close_shift))

    target_volatility = base_volatility + vol_offset
    # Volatility is a non-negative quantity (standard deviation). The
    # event-study shift can drive the sum below zero on a regime that
    # rotated from high to low realised vol; clip at zero so the target
    # tensor never carries a negative volatility row.
    if target_volatility < 0.0:
        target_volatility = 0.0

    target_date_str: str
    if realized_date:
        target_date_str = str(realized_date)
        try:
            target_date = datetime.date.fromisoformat(target_date_str[:10])
        except ValueError:
            target_date = event_date
    else:
        target_date = event_date
        target_date_str = event_date.isoformat()

    elapsed_time = float((target_date - event_date).days)
    vectors.append(
        FeatureVector.from_market_state(
            date=target_date_str,
            sentiment_score=sentiment_score,
            market_close=target_close,
            market_volatility=target_volatility,
            previous_close=base_close,
            previous_volatility=base_volatility,
            elapsed_time=elapsed_time,
        )
    )


def _resolve_training_package_dir(training_package_id: str) -> Path:
    """Resolve ``<id>`` to ``<DATA_DIR>/processed/<id>/``."""

    package_dir = DATA_DIR / "processed" / training_package_id
    if not package_dir.exists():
        raise FileNotFoundError(
            f"Training package directory not found: {package_dir}"
        )
    return package_dir


def _read_linguistic_lookup(package_dir: Path) -> dict[str, list[float]]:
    """Return ``text_hash -> 15-dim linguistic feature list``.

    Reads ``linguistic_features.parquet`` from the training-package
    directory when present. The parquet's column order is documented
    on :class:`app.features.linguistic.LinguisticVector`; this loader
    pins the slice to ``_LINGUISTIC_FEATURE_COLUMNS`` so a reordered
    upstream emission would surface as a missing-column error rather
    than a silently-misaligned input. NaN cells (e.g. ``pivot_distance``
    on the first statement) collapse to ``0.0`` here -- the absence
    flag is set at the row level by ``load_training_sequences_from_package``
    when the parquet is missing entirely; a per-cell NaN is treated
    as zero topic / density signal rather than "row missing".

    Returns an empty dict when the parquet is absent or unjoinable on
    ``text_hash``; callers then emit zeros and set the linguistic
    missing flag for every row.
    """

    import pandas as pd

    parquet_path = package_dir / "linguistic_features.parquet"
    if not parquet_path.exists():
        return {}
    frame = pd.read_parquet(parquet_path)
    if "text_hash" not in frame.columns:
        return {}
    # Older packages predate ``pivot_distance`` (PR #166) and so only
    # ship the first 14 linguistic columns. Missing columns are
    # tolerated and treated as ``0.0`` on every row so a sweep against
    # a pre-#166 package degrades cleanly rather than crashing; the
    # column-name match is exact so silent reorderings still surface
    # as the wrong-column-in-the-slot failure mode caught by the
    # downstream unit test.
    lookup: dict[str, list[float]] = {}
    for record in frame.to_dict("records"):
        text_hash = str(record.get("text_hash", "")).strip()
        if not text_hash:
            continue
        row: list[float] = []
        for column in _LINGUISTIC_FEATURE_COLUMNS:
            value = record.get(column) if column in frame.columns else None
            coerced = _coerce_finite_float(value)
            row.append(0.0 if coerced is None else coerced)
        lookup[text_hash] = row
    return lookup


def _read_mp_surprise_lookup(
    package_dir: Path,
) -> dict[str, dict[str, float]]:
    """Return ``event_date -> {mp_surprise_level, ..., is_intermeeting}``.

    Looks first inside the training package and then under the canonical
    location ``data/external/fred/mp_surprises.parquet``. Date keys are
    ``YYYY-MM-DD`` strings to match the event-row ``event_date`` column
    formatting. The lookup carries the four scalar columns the loader
    forwards into the rich-feature vector; the boolean
    ``is_intermeeting`` flag is left as a Python bool so the broadcaster
    can encode it as 0.0 / 1.0 at the bar level.

    Returns an empty dict when no parquet is found. Callers then emit
    zeros + a missing flag for every row.
    """

    import pandas as pd

    candidates = (
        package_dir / "mp_surprises.parquet",
        DATA_DIR / "external" / "fred" / "mp_surprises.parquet",
    )
    parquet_path: Path | None = None
    for candidate in candidates:
        if candidate.exists():
            parquet_path = candidate
            break
    if parquet_path is None:
        return {}
    frame = pd.read_parquet(parquet_path)
    if "event_date" not in frame.columns:
        return {}
    lookup: dict[str, dict[str, float]] = {}
    for record in frame.to_dict("records"):
        event_date_raw = record.get("event_date")
        if event_date_raw is None:
            continue
        event_date_str = str(event_date_raw)[:10]
        if not event_date_str:
            continue
        level = _coerce_finite_float(record.get("mp_surprise_level"))
        path_factor = _coerce_finite_float(record.get("mp_surprise_path_factor"))
        fed_info = _coerce_finite_float(record.get("fed_info_factor"))
        is_intermeeting_raw = record.get("is_intermeeting")
        if isinstance(is_intermeeting_raw, bool):
            is_intermeeting = 1.0 if is_intermeeting_raw else 0.0
        else:
            try:
                is_intermeeting = 1.0 if bool(int(is_intermeeting_raw)) else 0.0
            except (TypeError, ValueError):
                is_intermeeting = 0.0
        lookup[event_date_str] = {
            "mp_surprise_level": 0.0 if level is None else level,
            "mp_surprise_path_factor": 0.0 if path_factor is None else path_factor,
            "fed_info_factor": 0.0 if fed_info is None else fed_info,
            "mp_is_intermeeting": is_intermeeting,
        }
    return lookup


def _attach_rich_features(
    vectors: list[FeatureVector],
    *,
    event_row: dict[str, Any],
    linguistic_lookup: dict[str, list[float]],
    mp_surprise_lookup: dict[str, dict[str, float]],
    text_hash: str,
    event_date_str: str,
    use_credibility: bool,
    use_linguistic: bool,
    use_mp_surprise: bool,
    use_multi_axis: bool,
) -> None:
    """Broadcast event-level rich features onto every bar in a sequence.

    Per-family ablation flags zero the relevant slice (the per-bar
    feature size stays at ``RICH_FEATURE_SIZE``) so a downstream sweep
    can measure per-family lift without changing the model input
    shape. NaN inputs on the multi-axis fields collapse to ``0.0`` and
    flip the paired missing flag to ``1.0``.
    """

    # Credibility 4-vector is sourced directly off the event row.
    if use_credibility:
        cred_drift = _coerce_finite_float(event_row.get("credibility_drift_score"))
        cred_realized = _coerce_finite_float(
            event_row.get("credibility_realized_vs_stated_gap")
        )
        cred_market = _coerce_finite_float(
            event_row.get("credibility_market_implied_gap")
        )
        cred_months = _coerce_finite_float(
            event_row.get("credibility_months_since_reversal")
        )
    else:
        cred_drift = cred_realized = cred_market = cred_months = 0.0
    cred_drift = 0.0 if cred_drift is None else cred_drift
    cred_realized = 0.0 if cred_realized is None else cred_realized
    cred_market = 0.0 if cred_market is None else cred_market
    cred_months = 0.0 if cred_months is None else cred_months

    # Linguistic 15-vector. Zeros when the parquet is absent or the
    # text_hash is not joined; the row-level missing semantics are
    # captured by the linguistic-features ablation flag (no separate
    # per-row missing flag because every linguistic field already has
    # a well-defined zero baseline on the parquet).
    if use_linguistic:
        linguistic_row = linguistic_lookup.get(text_hash)
        if linguistic_row is None:
            linguistic_features = [0.0] * RICH_LINGUISTIC_DIM
        else:
            linguistic_features = list(linguistic_row)
    else:
        linguistic_features = [0.0] * RICH_LINGUISTIC_DIM

    # MP-surprise 4-vector. Joined on event_date. Missing parquet or
    # missing date both emit zeros.
    if use_mp_surprise:
        mp_row = mp_surprise_lookup.get(event_date_str, {})
        mp_level = float(mp_row.get("mp_surprise_level", 0.0))
        mp_path = float(mp_row.get("mp_surprise_path_factor", 0.0))
        fed_info = float(mp_row.get("fed_info_factor", 0.0))
        mp_intermeeting = float(mp_row.get("mp_is_intermeeting", 0.0))
    else:
        mp_level = mp_path = fed_info = mp_intermeeting = 0.0

    # Multi-axis 6-vector (3 values + 3 missing flags). NaN flips the
    # missing flag for that axis but the value still collapses to
    # zero, so the model sees "no signal" rather than "neutral
    # numeric value".
    if use_multi_axis:
        factor_raw = _coerce_finite_float(event_row.get("axis_factor"))
        certainty_raw = _coerce_finite_float(event_row.get("axis_certainty"))
        time_raw = _coerce_finite_float(event_row.get("axis_time"))
        axis_factor = factor_raw if factor_raw is not None else 0.0
        axis_factor_missing = 0.0 if factor_raw is not None else 1.0
        axis_certainty = certainty_raw if certainty_raw is not None else 0.0
        axis_certainty_missing = 0.0 if certainty_raw is not None else 1.0
        axis_time = time_raw if time_raw is not None else 0.0
        axis_time_missing = 0.0 if time_raw is not None else 1.0
    else:
        axis_factor = axis_certainty = axis_time = 0.0
        axis_factor_missing = axis_certainty_missing = axis_time_missing = 0.0

    for vector in vectors:
        vector.credibility_drift_score = cred_drift
        vector.credibility_realized_vs_stated_gap = cred_realized
        vector.credibility_market_implied_gap = cred_market
        vector.credibility_months_since_reversal = cred_months
        vector.linguistic_features = list(linguistic_features)
        vector.mp_surprise_level = mp_level
        vector.mp_surprise_path_factor = mp_path
        vector.fed_info_factor = fed_info
        vector.mp_is_intermeeting = mp_intermeeting
        vector.axis_factor = axis_factor
        vector.axis_factor_missing = axis_factor_missing
        vector.axis_certainty = axis_certainty
        vector.axis_certainty_missing = axis_certainty_missing
        vector.axis_time = axis_time
        vector.axis_time_missing = axis_time_missing
        vector.rich_payload = True


def _read_events_frame(package_dir: Path) -> "Any":
    import pandas as pd

    events_path = package_dir / "events.parquet"
    if not events_path.exists():
        raise FileNotFoundError(
            f"events.parquet missing from training package: {package_dir}"
        )
    return pd.read_parquet(events_path)


def _read_excluded_text_hashes(package_dir: Path) -> set[str]:
    """Return the ``text_hash`` set that must NOT enter the training loss.

    Uses ``splits_train_val_test.parquet`` when present and joinable via
    a ``text_hash`` column. The split-tag column is matched as either
    ``partition`` (forward-looking name from the data contract) or
    ``split_tag`` (current Phase 8 builder output).

    The training package builder writes ``split_tag`` ∈ {train, val,
    test}; everything except ``train`` is a forward-looking holdout
    under the walk-forward contract and must be excluded. The historical
    ``excluded_from_training`` sentinel is also accepted for forward
    compatibility with packages that materialise an explicit
    holdout-from-training partition.

    Returns an empty set when the file is absent or the schema is
    unjoinable.
    """

    import pandas as pd

    splits_path = package_dir / "splits_train_val_test.parquet"
    if not splits_path.exists():
        return set()
    frame = pd.read_parquet(splits_path)
    tag_column: str | None = None
    for candidate in ("partition", "split_tag"):
        if candidate in frame.columns:
            tag_column = candidate
            break
    if tag_column is None or "text_hash" not in frame.columns:
        return set()
    tags = frame[tag_column].astype(str)
    excluded_mask = tags.isin(_NON_TRAINING_SPLIT_TAGS) | (tags != _TRAINING_PARTITION)
    # Anything that is not explicitly "train" is excluded. The isin OR
    # is redundant but keeps the intent legible: val / test / explicit
    # excluded sentinel all drop.
    excluded = frame.loc[excluded_mask, "text_hash"]
    return {str(value) for value in excluded.tolist() if value}


def load_training_sequences_from_package(
    training_package_id: str,
    *,
    target_mode: TargetMode = DEFAULT_TARGET_MODE,
    rich_features: bool = True,
    use_credibility: bool = True,
    use_linguistic: bool = True,
    use_mp_surprise: bool = True,
    use_multi_axis: bool = True,
) -> list[list[FeatureVector]]:
    """Load one prior-window sequence per FOMC event in a training package.

    Reads ``data/processed/<training_package_id>/events.parquet`` (the
    collapsed view from ``app.data.event_dataset_builder``). Each row
    carries a ``prior_bars_json`` column with 20 trading-day bars
    immediately before the event ``as_of_ts``; the loader parses that
    JSON, derives a sentiment proxy from ``axis_stance``
    (``hawkish=+1, dovish=-1, neutral/None=0``), and constructs a
    ``FeatureVector`` per bar with ``elapsed_time`` set to the signed
    day count between the bar date and the event date. A single
    event-day target frame is appended per event so the downstream
    window slicer (``SEQUENCE_LENGTH=20``) sees the
    ``SEQUENCE_LENGTH + 1`` row it needs to materialise one supervised
    pair per event.

    The synthesised target frame's close and volatility derive from the
    ``target_mode`` selector:

    - ``event_study`` (default): target close projects the last prior
      bar via ``close * (1 + abnormal_return)`` (market-model residual
      against the 252-day window the events builder ships); target
      volatility is ``prior_bars[-1].vol_5d + volatility_shift``
      (the 10d post-event realised vol reconstructed from the prior
      vol plus the shift column).
    - ``realized_return``: legacy back-compat. Target close becomes
      ``close * (1 + realized_return)`` and target volatility is a
      literal copy of the last prior bar's ``vol_5d``. Preserved for
      smoke tests reproducing pre-event-study sweep results.

    NaN ``abnormal_return`` / ``volatility_shift`` rows fall back to
    the realized-return formula for that event and emit a
    ``UserWarning`` so a re-run against a package with missing target
    columns surfaces the gap immediately.

    Sequences are deduplicated to one per ``text_hash`` so the
    horizon-multiplied rows in ``events.parquet`` (h in {1, 5, 10, 30})
    do not inflate the training set; the prior window is a property of
    the event document, not the horizon. The deduplicator prefers the
    ``horizon=1`` row so the appended target frame is the next trading
    day's close, not a multi-day-forward projection. The result is
    sorted by ``event_date`` (then ``text_hash`` as a deterministic
    tiebreaker) so two calls with the same input produce the same
    ordering.

    When ``splits_train_val_test.parquet`` is present and exposes a
    ``partition`` / ``split_tag`` column joinable on ``text_hash``,
    rows tagged ``excluded_from_training`` are dropped before sequence
    construction.

    Returns a ``list[list[FeatureVector]]`` where each inner list holds
    20 prior bars followed by 1 event-day target row (21 vectors
    total). Events with fewer than 20 prior bars are skipped.

    When ``rich_features=True`` (the default) the loader joins
    ``linguistic_features.parquet`` on ``text_hash`` and
    ``mp_surprises.parquet`` on ``event_date``, reads the credibility
    and multi-axis columns straight off the event row, and broadcasts
    those event-level signals onto every bar of the prior window plus
    the appended event-day target frame. The resulting ``FeatureVector``
    rows emit 35 dims through ``as_rich_list``; the 6-dim ``as_list``
    output stays byte-identical. The per-family ablation flags
    (``use_credibility`` / ``use_linguistic`` / ``use_mp_surprise`` /
    ``use_multi_axis``) zero a family's slice while keeping the per-bar
    feature size constant, so a downstream sweep can measure per-family
    lift without retraining the model with a different input shape.

    Setting ``rich_features=False`` reproduces the pre-PR-#173 output:
    ``as_rich_list`` falls back to ``as_list`` plus zero-padding (no
    rich payload attached), and the per-family ablation flags are
    ignored.
    """

    if target_mode not in _VALID_TARGET_MODES:
        raise ValueError(
            f"Unsupported target_mode: {target_mode!r}. "
            f"Choose one of {sorted(_VALID_TARGET_MODES)}."
        )

    package_dir = _resolve_training_package_dir(training_package_id)
    frame = _read_events_frame(package_dir)
    if frame.empty:
        return []

    required_columns = {"event_date", "text_hash", "prior_bars_json"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(
            f"events.parquet at {package_dir} missing columns: {sorted(missing)}"
        )

    excluded_text_hashes = _read_excluded_text_hashes(package_dir)

    # Side-tables for the rich-feature join. Both lookups are read
    # once per package so the per-event broadcast does not re-open the
    # parquet on every iteration; the lookups are tiny in absolute
    # terms (one row per event for MP-surprise, one row per text_hash
    # for linguistic) so the memory footprint is negligible. When the
    # caller opted out via ``rich_features=False`` the lookups are
    # left empty -- ``_attach_rich_features`` is skipped entirely so
    # the legacy 6-dim path is undisturbed.
    linguistic_lookup: dict[str, list[float]] = {}
    mp_surprise_lookup: dict[str, dict[str, float]] = {}
    if rich_features:
        linguistic_lookup = _read_linguistic_lookup(package_dir)
        mp_surprise_lookup = _read_mp_surprise_lookup(package_dir)

    # Deduplicate to one row per text_hash. Prefer horizon=1 so the
    # appended target frame is the next trading day's close. Within a
    # text_hash bucket, lower horizons rank first; the chronological
    # sort on (event_date, text_hash) afterwards keeps the outer
    # iteration deterministic.
    def _row_rank(row: dict[str, Any]) -> tuple[int, int, str]:
        horizon = row.get("horizon")
        try:
            horizon_int = int(horizon) if horizon is not None else 10_000
        except (TypeError, ValueError):
            horizon_int = 10_000
        return (horizon_int, 0, str(row.get("source", "")))

    seen: set[str] = set()
    by_text_hash: dict[str, dict[str, Any]] = {}
    records = frame.to_dict("records")
    records.sort(key=_row_rank)
    for row in records:
        text_hash = str(row.get("text_hash", ""))
        if not text_hash:
            continue
        if text_hash in excluded_text_hashes:
            continue
        if text_hash in seen:
            continue
        seen.add(text_hash)
        by_text_hash[text_hash] = row

    ordered_rows = sorted(
        by_text_hash.values(),
        key=lambda row: (str(row.get("event_date", "")), str(row.get("text_hash", ""))),
    )

    sequences: list[list[FeatureVector]] = []
    for row in ordered_rows:
        event_date_str = str(row.get("event_date", ""))
        if not event_date_str:
            continue
        try:
            event_date = datetime.date.fromisoformat(event_date_str[:10])
        except ValueError:
            continue
        bars = _parse_prior_bars(row.get("prior_bars_json"))
        if len(bars) < SEQUENCE_LENGTH:
            continue
        row_text_hash = str(row.get("text_hash", ""))
        sentiment_score = _stance_to_sentiment(row.get("axis_stance"))
        vectors = _bars_to_feature_vectors(
            bars,
            event_date=event_date,
            sentiment_score=sentiment_score,
        )
        if len(vectors) < SEQUENCE_LENGTH:
            continue
        realized_return = _coerce_finite_float(row.get("realized_return"))
        abnormal_return = _coerce_finite_float(row.get("abnormal_return"))
        volatility_shift = _coerce_finite_float(row.get("volatility_shift"))
        realized_date_raw = row.get("realized_date")
        if (
            realized_date_raw is None
            or (isinstance(realized_date_raw, float) and realized_date_raw != realized_date_raw)
        ):
            realized_date = None
        else:
            text = str(realized_date_raw).strip()
            realized_date = text or None
        _append_event_day_target(
            vectors,
            event_date=event_date,
            realized_return=realized_return,
            realized_date=realized_date,
            sentiment_score=sentiment_score,
            abnormal_return=abnormal_return,
            volatility_shift=volatility_shift,
            target_mode=target_mode,
        )
        if rich_features:
            _attach_rich_features(
                vectors,
                event_row=row,
                linguistic_lookup=linguistic_lookup,
                mp_surprise_lookup=mp_surprise_lookup,
                text_hash=row_text_hash,
                event_date_str=event_date_str[:10],
                use_credibility=use_credibility,
                use_linguistic=use_linguistic,
                use_mp_surprise=use_mp_surprise,
                use_multi_axis=use_multi_axis,
            )
        sequences.append(vectors)
    return sequences


def fit_close_scale(sequence_groups: Sequence[Sequence[FeatureVector]]) -> float:
    """Compute the per-fold close-price normaliser from the training rows.

    The forecaster normalises the close target by dividing by a positive
    scale so the regression head outputs O(1) values. Earlier code used a
    global constant (``DEFAULT_CLOSE_SCALE = 10000``), which was correct
    only when the asset happened to trade around 5000 — for crypto, bonds,
    or pre-2000 history the constant under- or over-shoots by an order of
    magnitude and the loss surface tilts.

    The fit is the mean of the strictly-positive close values in the
    training-target windows (i.e., the rows that actually become a y row,
    not the lookback frames). When the data is empty or has no positive
    closes (e.g., a synthetic-zero fixture) we fall back to
    ``DEFAULT_CLOSE_SCALE`` so the legacy LSTM smoke-train path stays
    valid.

    Deterministic given identical input — same vectors -> same scale, no
    randomness. The regression test
    (``tests/regression/test_forecaster_determinism.py``) relies on this.
    """

    target_closes: list[float] = []
    for group in sequence_groups:
        if len(group) < SEQUENCE_LENGTH + 1:
            continue
        for idx in range(SEQUENCE_LENGTH, len(group)):
            target = group[idx]
            value = float(target.market_close)
            if value > 0.0:
                target_closes.append(value)
    if not target_closes:
        return float(DEFAULT_CLOSE_SCALE)
    return float(sum(target_closes) / len(target_closes))


def _build_training_tensors(
    sequence_groups: Sequence[Sequence[FeatureVector]],
    close_scale: float | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, float]:
    """Materialise the (x, y, close_scale) triple for the training path.

    The third return value is the close-scale that was used for
    normalisation. Callers that need to persist the scaler (training
    loop) read it from the tuple; callers that only want the tensors
    can keep the first two and discard the third. When ``close_scale``
    is supplied, the caller has already fitted it (e.g. on a strict
    train-only window for walk-forward); when ``None`` we fit it on
    the fly from the same sequences.

    Per-bar feature size is chosen by inspecting the first feature
    vector in the input: rows whose ``rich_payload`` flag is True
    emit through :meth:`FeatureVector.as_rich_list` (35 dims),
    everything else stays on the legacy :meth:`FeatureVector.as_list`
    (6 dims). The dispatch keeps the pre-PR-#173 byte-identical
    contract for any sequence group built from the legacy
    ``data_dir`` JSON scan (rich_payload defaults to False on every
    ``FeatureVector`` constructed without the rich-feature loader).
    """

    fitted_scale = float(close_scale) if close_scale is not None else fit_close_scale(sequence_groups)

    use_rich = False
    for sequence_group in sequence_groups:
        for item in sequence_group:
            if getattr(item, "rich_payload", False):
                use_rich = True
                break
        if use_rich:
            break

    sequences: list[list[list[float]]] = []
    targets: list[list[float]] = []

    for sequence_group in sequence_groups:
        if len(sequence_group) < SEQUENCE_LENGTH + 1:
            continue
        for idx in range(SEQUENCE_LENGTH, len(sequence_group)):
            window = sequence_group[idx - SEQUENCE_LENGTH : idx]
            target = sequence_group[idx]
            if use_rich:
                row_list = [item.as_rich_list() for item in window]
            else:
                row_list = [item.as_list() for item in window]
            sequences.append(row_list)
            targets.append(
                [
                    target.market_close / fitted_scale,
                    max(target.market_volatility, 0.0),
                ]
            )

    if not sequences:
        return None, None, fitted_scale

    x = torch.tensor(sequences, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    return x, y, fitted_scale


def _split_train_validation(
    x: torch.Tensor,
    y: torch.Tensor,
    validation_split: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(x) < 2:
        return x, y, x, y

    val_size = max(1, int(len(x) * validation_split))
    train_size = max(1, len(x) - val_size)
    if train_size >= len(x):
        train_size = len(x) - 1
        val_size = 1

    return x[:train_size], y[:train_size], x[train_size:], y[train_size:]
