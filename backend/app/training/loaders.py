from __future__ import annotations

import csv
import dataclasses
import datetime
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Literal, Sequence

import torch

from app.config import DATA_DIR
from app.evaluation.metrics import TrainingDataSourceSummary
from app.models.config import (
    DEFAULT_CLOSE_SCALE,
    DEFAULT_DATA_DIR,
    DEFAULT_TEXT_ADAPTER_DIM,
    DEFAULT_TEXT_POOL_LAMBDA_INV_DAYS,
    FEATURE_SIZE,
    MULTI_TASK_CERTAINTY_LABELS,
    MULTI_TASK_STANCE_LABELS,
    MULTI_TASK_TOPIC_LABELS,
    RICH_FEATURE_SIZE,
    RICH_LINGUISTIC_DIM,
    RichFeatureScalerParams,
    SEQUENCE_LENGTH,
    FeatureVector,
)

_logger = logging.getLogger(__name__)

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


@dataclasses.dataclass(frozen=True)
class WalkForwardSplit:
    """Pre-split sequence groups for a single walk-forward fold.

    Holds the three sequence-group lists the trainer consumes
    independently: the training partition the optimiser fits, the
    validation partition early-stopping watches, and the held-out test
    partition the aggregator reports as the headline RMSE. Per-list
    ``event_dates`` mirror the per-list ``text_hash`` order so the
    aggregator can audit fold boundaries without reopening the parquet.

    ``fold_id`` is ``None`` on the single-fold default path (the
    package's ``splits_train_val_test.parquet`` already names the
    partition per row). Multi-fold callers populate it with the
    manifest's fold id (``wf_fold_1`` ...).

    ``protocol`` distinguishes ``single-fold`` (legacy split-tag
    partition) from ``walk-forward`` (expanding training window read
    off ``fold_manifest_expanding_walk_forward.json``) so the
    aggregator can label rows without re-deriving the path from the
    fold id.
    """

    train: list[list[FeatureVector]]
    val: list[list[FeatureVector]]
    test: list[list[FeatureVector]]
    train_event_dates: list[str]
    val_event_dates: list[str]
    test_event_dates: list[str]
    fold_id: str | None = None
    protocol: str = "single-fold"


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
        # A2 (#207) per-bar realised-vol horizons. Older events.parquet
        # vintages predating PR #218+ won't have these keys; .get() with
        # default 0.0 keeps the deser tolerant.
        vol_20d_value = float(bar.get("vol_20d", 0.0))
        vol_60d_value = float(bar.get("vol_60d", 0.0))
        # A3 (#208) cross-asset close levels. Same back-compat
        # contract as A2 -- pre-A3 events.parquet emits 0.0 for every
        # cross-asset axis and the rich-feature block keeps loading.
        vix_close_value = float(bar.get("vix_close", 0.0))
        dxy_close_value = float(bar.get("dxy_close", 0.0))
        tnx_close_value = float(bar.get("tnx_close", 0.0))
        gold_close_value = float(bar.get("gold_close", 0.0))
        # Path B Chunk 1 (vol-regime macro features). Missing keys on
        # pre-widen events.parquet default to 0.0 — same back-compat
        # contract as the A3 cross-asset bundle.
        vix3m_close_value = float(bar.get("vix3m_close", 0.0))
        irx_close_value = float(bar.get("irx_close", 0.0))
        vix_term_slope_value = float(bar.get("vix_term_slope", 0.0))
        yield_curve_slope_value = float(bar.get("yield_curve_slope_10y_3m", 0.0))
        elapsed_time = float((bar_date - event_date).days)
        fv = FeatureVector.from_market_state(
            date=date_value,
            sentiment_score=sentiment_score,
            market_close=close_value,
            market_volatility=volatility_value,
            previous_close=previous_close,
            previous_volatility=previous_volatility,
            elapsed_time=elapsed_time,
        )
        fv.realized_vol_20d = vol_20d_value
        fv.realized_vol_60d = vol_60d_value
        fv.vix_close = vix_close_value
        fv.dxy_close = dxy_close_value
        fv.tnx_close = tnx_close_value
        fv.gold_close = gold_close_value
        fv.vix3m_close = vix3m_close_value
        fv.irx_close = irx_close_value
        fv.vix_term_slope = vix_term_slope_value
        fv.yield_curve_slope_10y_3m = yield_curve_slope_value
        vectors.append(fv)
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
    """Resolve ``<id>`` to a local package dir.

    Accepts either:

    - a local package id like ``tp_v3_macro_aug_2026_05_25`` which
      maps to ``<DATA_DIR>/processed/<id>/`` (legacy behaviour), or
    - an ``hf://datasets/owner/name[:revision]`` URI which is pulled
      via :func:`huggingface_hub.snapshot_download` into the HF cache
      and treated as the package directory (deployability lane #302).

    Also verifies the manifest sidecar (``dataset_metadata.sha256``)
    when present. A mismatch raises ``ManifestShaMismatch``; a missing
    sidecar emits a warning so the package surfaces for backfill but
    the load proceeds.
    """

    from app.models.registry import is_hf_uri, resolve_hf_uri

    if is_hf_uri(training_package_id):
        package_dir = resolve_hf_uri(training_package_id)
    else:
        package_dir = DATA_DIR / "processed" / training_package_id
    if not package_dir.exists():
        raise FileNotFoundError(
            f"Training package directory not found: {package_dir}"
        )
    from app.data.manifest_sha import verify_manifest_sha

    verify_manifest_sha(package_dir)
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


def _read_sep_projections_lookup(
    package_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Return ``meeting_date -> {ffr_median_*, ffr_range_*}``.

    Looks first inside the training package and then under the canonical
    location ``data/external/fred/sep_projections.parquet``. Date keys
    are ``YYYY-MM-DD`` strings to match the SEP composer's
    ``meeting_date`` field. Each value carries the five scalar columns
    the composer reads (current-year median + next-year median +
    longer-run median + current-year range upper/lower). The next-year
    median may be ``None`` on pre-2014 vintages where FRED has no
    year-specific ``FEDTARMD<YYYY>`` series; the composer collapses
    that to ``0.0`` so the block shape stays fixed.

    Returns an empty dict when no parquet is found. The composer then
    returns ``None`` for every event and the loader collapses the slot
    to the all-zeros + missing-flag-1.0 state (graceful degrade — the
    code path stays live without the parquet on disk). See #215.
    """

    import pandas as pd

    candidates = (
        package_dir / "sep_projections.parquet",
        DATA_DIR / "external" / "fred" / "sep_projections.parquet",
    )
    parquet_path: Path | None = None
    for candidate in candidates:
        if candidate.exists():
            parquet_path = candidate
            break
    if parquet_path is None:
        return {}
    frame = pd.read_parquet(parquet_path)
    if "meeting_date" not in frame.columns:
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for record in frame.to_dict("records"):
        meeting_raw = record.get("meeting_date")
        if meeting_raw is None:
            continue
        meeting_str = str(meeting_raw)[:10]
        if not meeting_str:
            continue
        lookup[meeting_str] = {
            "meeting_date": meeting_str,
            "ffr_median_current_year": _coerce_finite_float(
                record.get("ffr_median_current_year")
            ),
            "ffr_median_next_year": _coerce_finite_float(
                record.get("ffr_median_next_year")
            ),
            "ffr_median_longer_run": _coerce_finite_float(
                record.get("ffr_median_longer_run")
            ),
            "ffr_range_upper_current": _coerce_finite_float(
                record.get("ffr_range_upper_current")
            ),
            "ffr_range_lower_current": _coerce_finite_float(
                record.get("ffr_range_lower_current")
            ),
        }
    return lookup


def _read_press_conf_qa_lookup(
    package_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Return ``event_date -> {qa_text, prepared_remarks_text, has_press_conf}``.

    Reads the #214 FOMC press-conference Q&A corpus from either the
    training package or the canonical location
    ``data/external/fomc_press_conferences/qa_lookup.parquet``. The
    parquet is produced by the press-conference scraper's
    ``build_qa_lookup`` helper; absence on disk collapses to an empty
    dict and the loader treats every event as ``has_press_conf=0``
    (pre-2011 covariate-shift handling under route 1 of #214 — see
    ADR 0037).

    Date keys are ``YYYY-MM-DD`` strings matching the events.parquet
    ``event_date`` column. ``qa_text`` and ``prepared_remarks_text`` may
    be empty on rows where the PDF Q&A boundary was not locatable; the
    ``has_press_conf`` flag still fires for the covariate-shift
    distinction (the press conference happened, the text just did not
    survive the split heuristic). The loader treats empty ``qa_text``
    as "no LoRA-side text to concat", and the static-cache path sees
    the same scalar flag regardless.
    """

    import pandas as pd

    candidates = (
        package_dir / "qa_lookup.parquet",
        DATA_DIR / "external" / "fomc_press_conferences" / "qa_lookup.parquet",
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
    lookup: dict[str, dict[str, Any]] = {}
    for record in frame.to_dict("records"):
        event_raw = record.get("event_date")
        if event_raw is None:
            continue
        event_str = str(event_raw)[:10]
        if not event_str:
            continue
        lookup[event_str] = {
            "qa_text": str(record.get("qa_text") or ""),
            "prepared_remarks_text": str(record.get("prepared_remarks_text") or ""),
            "has_press_conf": 1.0
            if str(record.get("has_press_conf") or "0") not in ("0", "0.0", "", "False", "false")
            else 0.0,
        }
    return lookup


def _compute_press_conf_features_for_event(
    *,
    event_date_str: str,
    press_conf_lookup: dict[str, dict[str, Any]],
) -> list[float]:
    """Per-event #214 press-conf feature block.

    Returns the single-element ``[has_press_conf]`` block. The press-conf
    lookup is keyed on the supervised event's ISO date; a hit emits
    ``1.0`` (Q&A transcript landed in the joint corpus for this event),
    a miss emits ``0.0`` (pre-2011 era or otherwise no press conference
    on this date — the canonical zero-impute handling per ADR 0037).

    Unlike the regime / SEP composers this helper always returns a
    populated list rather than ``None``: the caller is expected to set
    the slot unconditionally when ``--use-press-conf`` is on so the
    covariate-shift flag is present on every row in the joint corpus.
    """

    record = press_conf_lookup.get(event_date_str[:10])
    if not record:
        return [0.0]
    return [float(record.get("has_press_conf", 0.0))]


def _compute_sep_features_for_event(
    *,
    event_date: datetime.date,
    sep_lookup: dict[str, dict[str, Any]],
) -> list[float] | None:
    """Per-event #215 SEP feature block -- strict-prior by construction.

    Returns the 5-scalar block (current-year / next-year / longer-run
    medians + central-tendency range + release flag) when the lookup
    carries an SEP release on or before ``event_date``; returns ``None``
    on cold-start (no eligible release). The caller treats ``None`` as
    "no signal" and flips the missing flag to 1.0.

    The composer reads strictly-prior or T-snapshot rows only; see
    :mod:`app.training.sep_features` for the per-feature contract and
    ``docs/feature-provenance-audit.md`` for the audit row.
    """

    from app.training.sep_features import compute_sep_features_for_event

    features = compute_sep_features_for_event(
        event_date=event_date,
        sep_lookup=sep_lookup,
    )
    if features is None:
        return None
    return features.as_list()


def _read_mp_surprise_lookup(
    package_dir: Path,
) -> dict[str, dict[str, float]]:
    """Return ``event_date -> {mp_surprise_level, ..., is_intermeeting}``.

    Looks first inside the training package and then under the canonical
    location ``data/external/fred/mp_surprises.parquet``. Date keys are
    ``YYYY-MM-DD`` strings to match the event-row ``event_date`` column
    formatting. The lookup carries the four scalar columns the loader
    forwards into the rich-feature vector, plus the ``ff_target_prior``
    column (the strict-prior band midpoint published the day before
    each meeting's announcement) that the #307 macro-regime conditioning
    helper consumes when scoring the trailing-12-month policy cycle.
    The boolean ``is_intermeeting`` flag is left as a Python bool so
    the broadcaster can encode it as 0.0 / 1.0 at the bar level.

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
        # ``ff_target_prior`` is the strict-prior band midpoint observed
        # the day before each meeting's announcement. The #307 macro-
        # regime helper reads it across trailing meetings to score the
        # 12-month policy-cycle direction. Stored as ``NaN`` (not
        # ``None`` and not ``0.0``) when missing so the dict value-type
        # stays ``float`` (mypy happy) and the regime helper can drop
        # the row via its ``v != v`` check rather than misread a
        # placeholder zero as a real zero-rate observation.
        target_prior_raw = _coerce_finite_float(record.get("ff_target_prior"))
        target_prior = float("nan") if target_prior_raw is None else target_prior_raw
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
            "ff_target_prior": target_prior,
        }
    return lookup


def _read_chunk_embedding_lookup(
    encoder_alias: str,
    *,
    cache_dir: Path | None = None,
    registry_path: Path | None = None,
) -> tuple[dict[str, "Any"], dict[str, str]]:
    """Return ``(text_hash -> pooled embedding, text_hash -> event_date)``.

    Reads ``data/raw/embeddings/<encoder_alias>_<rev>.parquet`` (the
    artefact ``app.data.embedding_cache.build_cache`` writes per
    encoder x training-package). The cache stores one row per
    document chunk for classifier/MLM encoders and one row per
    document for sentence-embedding encoders; this helper collapses
    all rows under a given ``record_id`` to a single mean-pooled
    vector so the downstream prior-4 weighting sees one embedding per
    statement.

    The cache schema persists ``record_id`` rather than ``text_hash``.
    When ``registry_path`` is supplied the loader walks
    ``registry_normalized.jsonl`` to build a ``record_id -> text_hash``
    mapping; the returned dicts are then keyed on ``text_hash`` so the
    per-event lookup matches the ``events.parquet`` join key directly.
    When the registry is unavailable the lookup falls back to
    ``record_id`` keys and the caller's text_hash lookup will miss.

    The two return dicts share the same key space. The first carries
    pooled embeddings (``numpy.ndarray``), the second carries the
    ISO-formatted event date for each key. The split replaces the
    previous ``__event_dates__`` sentinel-key layout so the types stay
    clean and the pooler doesn't need to filter prefix-reserved keys.

    Returns a pair of empty dicts when the parquet is missing; the
    caller emits zeros + a missing flag for every row.
    """

    import numpy as np
    import pandas as pd

    # Local import keeps the heavy embedding-cache module out of the
    # training-time import path on the legacy ``--data-dir`` flow.
    from app.data.embedding_cache import (
        EmbeddingCacheUnavailable,
        ensure_local,
        resolve_cache_paths,
    )
    from app.models.registry import revision_for

    revision = revision_for(encoder_alias)
    if revision is None:
        _logger.warning(
            "encoder %r is not pinned in models/registry.yaml; "
            "text-embedding lookup will return empty",
            encoder_alias,
        )
        return {}, {}
    paths = resolve_cache_paths(encoder_alias, revision=revision, cache_dir=cache_dir)
    if not paths.parquet.exists():
        # Production path (#302): if the parquet is absent locally, try
        # the lazy HF Hub fetch before degrading to empty. The training-
        # time flow that explicitly disables network access keys off
        # ``FED_PULSE_ALLOW_HF_FETCH=0``; production leaves it unset so
        # the default ``True`` route hits the Hub.
        allow_fetch = os.environ.get("FED_PULSE_ALLOW_HF_FETCH", "1") != "0"
        if allow_fetch:
            try:
                ensure_local(encoder_alias, revision=revision, cache_dir=cache_dir)
            except EmbeddingCacheUnavailable as exc:
                _logger.warning(
                    "embedding cache lazy-fetch failed for encoder=%r: %s; "
                    "text-embedding lookup will return empty",
                    encoder_alias,
                    exc,
                )
                return {}, {}
        if not paths.parquet.exists():
            _logger.warning(
                "embedding cache parquet missing for encoder=%r at %s; "
                "text-embedding lookup will return empty",
                encoder_alias,
                paths.parquet,
            )
            return {}, {}

    frame = pd.read_parquet(paths.parquet)
    if "embedding" not in frame.columns or "event_date" not in frame.columns:
        _logger.warning(
            "embedding cache parquet at %s missing required columns "
            "(embedding / event_date); text-embedding lookup empty",
            paths.parquet,
        )
        return {}, {}
    # Prefer ``record_id`` (Phase 8 cache builder) and fall back to
    # ``doc_id`` for backwards compatibility with older caches.
    key_column: str | None = None
    for candidate in ("record_id", "doc_id"):
        if candidate in frame.columns:
            key_column = candidate
            break
    if key_column is None:
        _logger.warning(
            "embedding cache parquet at %s carries neither record_id "
            "nor doc_id; text-embedding lookup empty",
            paths.parquet,
        )
        return {}, {}

    # Build the ``record_id -> text_hash`` mapping when the registry is
    # available. Without it the lookup stays keyed on record_id and
    # the per-event text_hash join misses every row.
    record_to_text_hash: dict[str, str] = {}
    if registry_path is not None and registry_path.exists():
        with registry_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                rec_id = str(record.get("record_id") or "")
                text_hash_str = str(record.get("text_hash") or "")
                if rec_id and text_hash_str:
                    record_to_text_hash[rec_id] = text_hash_str

    lookup: dict[str, "Any"] = {}
    event_dates: dict[str, str] = {}
    grouped = frame.groupby(key_column, sort=False)
    for key_value, chunk in grouped:
        key_str = str(key_value).strip()
        if not key_str:
            continue
        stacked: list[np.ndarray] = []
        for embedding in chunk["embedding"].tolist():
            try:
                vec = np.asarray(embedding, dtype=np.float32)
            except (TypeError, ValueError):
                continue
            if vec.ndim != 1 or vec.size == 0:
                continue
            stacked.append(vec)
        if not stacked:
            continue
        # Drop rows whose embedding dim mismatches the first chunk; a
        # mixed-dim parquet means the cache was rebuilt with two
        # different encoders and silently appended -- safer to skip
        # than to mean-pool across dims.
        ref_dim = stacked[0].shape[0]
        cleaned = [vec for vec in stacked if vec.shape[0] == ref_dim]
        if not cleaned:
            continue
        pooled = np.mean(np.stack(cleaned, axis=0), axis=0)
        # Prefer the text_hash key when the registry resolved one for
        # this record_id; fall back to record_id otherwise.
        join_key = record_to_text_hash.get(key_str, key_str)
        lookup[join_key] = pooled
        event_dates[join_key] = str(chunk["event_date"].iloc[0])[:10]

    if not lookup:
        return {}, {}
    return lookup, event_dates


def _compute_prior4_pooled_embedding(
    *,
    text_hash: str,
    event_row_text_hash: str,
    current_event_date: datetime.date,
    embedding_lookup: dict[str, "Any"],
    prior_text_hashes: Sequence[tuple[datetime.date, str]],
    lambda_inv_days: float,
    max_prior: int = 4,
) -> "Any | None":
    """Pool the four most recent prior-statement embeddings with time decay.

    Parameters
    ----------
    text_hash:
        ``text_hash`` (== embedding-cache ``record_id``) of the event
        being processed. Used only for diagnostics; the pooler reads
        prior statements off ``prior_text_hashes``.
    current_event_date:
        Statement date of the event being processed. Prior statements
        are restricted to those strictly before this date.
    embedding_lookup:
        First element of the tuple returned by
        :func:`_read_chunk_embedding_lookup`. Keys are ``record_id``
        strings, values are 1-D ``numpy.ndarray`` per statement. The
        companion ``event_dates`` dict from the same tuple is not
        consumed here -- the pooler reads dates off ``prior_text_hashes``
        which already carries the chronology.
    prior_text_hashes:
        Chronologically-sorted list of ``(statement_date, text_hash)``
        tuples for every statement in the corpus. The pooler filters
        on ``statement_date < current_event_date`` and picks the
        ``max_prior`` most recent surviving rows.
    lambda_inv_days:
        Time-decay window. Weights derive from
        ``softmax(-Delta t_days / lambda_inv_days)``; smaller values
        concentrate the weight on the most recent statement, larger
        values spread it across the four.

    Returns
    -------
    ``numpy.ndarray`` (the weighted mean over ``min(4, n_prior)``
    statements) or ``None`` when no usable prior is found.
    """

    import numpy as np

    if not embedding_lookup:
        return None
    if lambda_inv_days <= 0:
        raise ValueError(
            f"lambda_inv_days must be positive; got {lambda_inv_days}"
        )

    # Filter to statements strictly before the current event date that
    # actually have a pooled embedding in the lookup, then keep the
    # ``max_prior`` most recent.
    candidates: list[tuple[datetime.date, str]] = []
    for statement_date, prior_hash in prior_text_hashes:
        if statement_date >= current_event_date:
            continue
        if prior_hash not in embedding_lookup:
            continue
        candidates.append((statement_date, prior_hash))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:max_prior]

    delta_days = np.array(
        [float((current_event_date - statement_date).days) for statement_date, _ in selected],
        dtype=np.float64,
    )
    logits = -delta_days / float(lambda_inv_days)
    # Numerically-stable softmax: subtract max so exp does not overflow.
    logits -= logits.max()
    weights = np.exp(logits)
    weights_sum = weights.sum()
    if weights_sum <= 0:
        return None
    weights = weights / weights_sum

    vectors = [np.asarray(embedding_lookup[h], dtype=np.float32) for _, h in selected]
    # Tolerate ragged-dim entries by trimming to the shortest -- the
    # lookup builder already filters mixed dims per record_id, but a
    # cross-encoder corpus could still mix dims across records. Skip
    # the pool entirely if the dims are not coherent.
    ref_dim = vectors[0].shape[0]
    if any(vec.shape[0] != ref_dim for vec in vectors):
        return None
    matrix = np.stack(vectors, axis=0)
    pooled = (matrix * weights[:, None]).sum(axis=0)
    return pooled.astype(np.float32)


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

    # Option-A multi-axis slot: stance one-hot + gtfintechlab indicators
    # + stance_missing flag. Replaces the pre-2026-05-17 numeric axes
    # (axis_factor / axis_certainty / axis_time) that were 0% populated
    # upstream. ``use_multi_axis=False`` zeros every position including
    # ``stance_missing`` so the per-family ablation reads as a true
    # "no slot" rather than "stance is unknown".
    if use_multi_axis:
        stance_raw = event_row.get("axis_stance")
        stance = (
            str(stance_raw).strip().lower()
            if isinstance(stance_raw, str) and stance_raw.strip()
            else None
        )
        stance_hawk = 1.0 if stance == "hawkish" else 0.0
        stance_dove = 1.0 if stance == "dovish" else 0.0
        stance_neutral = 1.0 if stance == "neutral" else 0.0
        stance_missing = 1.0 if stance is None else 0.0

        time_raw = event_row.get("axis_time_label")
        time_label_forward = (
            1.0
            if isinstance(time_raw, str)
            and time_raw.strip().lower() == "forward looking"
            else 0.0
        )

        certain_raw = event_row.get("axis_certain_label")
        certain_label_certain = (
            1.0
            if isinstance(certain_raw, str)
            and certain_raw.strip().lower() == "certain"
            else 0.0
        )
    else:
        stance_hawk = stance_dove = stance_neutral = 0.0
        time_label_forward = certain_label_certain = 0.0
        stance_missing = 0.0

    # Multi-task head (#78) per-axis training targets. Independent of
    # ``use_multi_axis`` (which controls the rich-feature INPUT block);
    # targets are always lifted off the event row when present so the
    # masked loss can contribute on every supervised row that carries a
    # label. Missing labels leave the target at its default and the
    # mask flag at False; the loss reads the flag to skip that axis on
    # that row.
    target_stance_str = (
        str(event_row.get("axis_stance")).strip().lower()
        if isinstance(event_row.get("axis_stance"), str)
        and str(event_row.get("axis_stance")).strip()
        else None
    )
    target_stance_idx = -1
    target_stance_present = False
    if target_stance_str in MULTI_TASK_STANCE_LABELS:
        target_stance_idx = MULTI_TASK_STANCE_LABELS.index(target_stance_str)
        target_stance_present = True

    target_factor_value = _coerce_finite_float(event_row.get("axis_factor"))
    if target_factor_value is None:
        target_factor = 0.0
        target_factor_present = False
    else:
        target_factor = max(min(float(target_factor_value), 1.0), -1.0)
        target_factor_present = True

    # Certainty: prefer the categorical ``axis_certain_label`` (string)
    # from gtfintechlab rows; fall back to the numeric ``axis_certainty``
    # (float in [0, 1]) binned into 3 classes when only the float is
    # populated. Tertiles fit the {certain, uncertain, neutral} taxonomy.
    target_certainty_idx = -1
    target_certainty_present = False
    certainty_str_raw = event_row.get("axis_certain_label")
    certainty_str = (
        str(certainty_str_raw).strip().lower()
        if isinstance(certainty_str_raw, str) and str(certainty_str_raw).strip()
        else None
    )
    if certainty_str in MULTI_TASK_CERTAINTY_LABELS:
        target_certainty_idx = MULTI_TASK_CERTAINTY_LABELS.index(certainty_str)
        target_certainty_present = True
    else:
        certainty_float = _coerce_finite_float(event_row.get("axis_certainty"))
        if certainty_float is not None:
            if certainty_float >= 0.66:
                target_certainty_idx = MULTI_TASK_CERTAINTY_LABELS.index("certain")
            elif certainty_float <= 0.33:
                target_certainty_idx = MULTI_TASK_CERTAINTY_LABELS.index("uncertain")
            else:
                target_certainty_idx = MULTI_TASK_CERTAINTY_LABELS.index("neutral")
            target_certainty_present = True

    target_topic_idx = -1
    target_topic_present = False
    topic_raw = event_row.get("axis_topic")
    topic_str = (
        str(topic_raw).strip().lower()
        if isinstance(topic_raw, str) and str(topic_raw).strip()
        else None
    )
    if topic_str is not None:
        for canonical in MULTI_TASK_TOPIC_LABELS[:-1]:
            if canonical in topic_str:
                target_topic_idx = MULTI_TASK_TOPIC_LABELS.index(canonical)
                target_topic_present = True
                break
        if not target_topic_present:
            target_topic_idx = MULTI_TASK_TOPIC_LABELS.index("other")
            target_topic_present = True

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
        vector.stance_hawk = stance_hawk
        vector.stance_dove = stance_dove
        vector.stance_neutral = stance_neutral
        vector.time_label_forward = time_label_forward
        vector.certain_label_certain = certain_label_certain
        vector.stance_missing = stance_missing
        vector.target_stance_idx = target_stance_idx
        vector.target_stance_present = target_stance_present
        vector.target_factor = target_factor
        vector.target_factor_present = target_factor_present
        vector.target_certainty_idx = target_certainty_idx
        vector.target_certainty_present = target_certainty_present
        vector.target_topic_idx = target_topic_idx
        vector.target_topic_present = target_topic_present
        vector.rich_payload = True


def _compute_analog_features_for_event(
    *,
    event_text: str,
    event_date: datetime.date,
    event_stance: Any,
) -> list[float] | None:
    """Per-event retrieval query + derived summary block (#306).

    Calls the runtime ``app.services.analogs`` singleton with a strict-
    backward ``as_of_date=event_date`` filter so only analog rows whose
    indexed ``event_date`` is strictly less than the supervised event
    enter the top-K. Returns ``None`` when the retrieval bundle is
    absent on disk (graceful degrade — the loader then leaves every
    event in the all-zeros + missing-flag-1.0 state); returns the
    derived 5-dim feature list otherwise.

    The block is contextual only: similarity moments + stance-agreement
    score. The analog's post-event observed move is NOT in the block —
    admitting it would be a label leak via similarity. See ADR 0028 and
    the row in ``docs/feature-provenance-audit.md``.
    """

    from app.training.retrieval_features import (
        compute_analog_summary_features,
        lookup_analog_hits,
    )

    hits = lookup_analog_hits(text=event_text, event_date=event_date)
    if hits is None:
        # Bundle absent on disk; the caller emits zeros + missing=1.0.
        return None
    if not hits:
        # Bundle present but no row clears the strict-backward
        # ``analog_event_date < event_date`` cutoff (first event in the
        # corpus). Same all-zeros + missing-flag-1.0 contract as the
        # absent-bundle case so the model sees a consistent "no
        # retrieval signal" representation in both branches.
        return None
    stance = (
        str(event_stance).strip().lower()
        if isinstance(event_stance, str) and str(event_stance).strip()
        else None
    )
    summary = compute_analog_summary_features(hits, event_stance=stance)
    return summary.as_list()


def _compute_macro_regime_features_for_event(
    *,
    event_date: datetime.date,
    bars: Sequence[dict[str, Any]],
    mp_surprise_lookup: dict[str, dict[str, Any]],
) -> list[float] | None:
    """Per-event #307 macro-regime indicators -- strict-prior throughout.

    Returns the 3-scalar block (policy-cycle phase, VIX-level regime,
    term-spread sign) when the strict-prior inputs are available;
    returns ``None`` only when no meaningful trailing data can be read
    (cold-start of the corpus on every input axis). The caller treats
    ``None`` as "no signal" and flips the missing flag to 1.0.

    Each scalar lives in ``{-1.0, 0.0, +1.0}`` so the gating layer
    downstream consumes them without per-fold rescaling. See
    :mod:`app.training.regime_features` for the per-feature contract
    and ``docs/feature-provenance-audit.md`` for the audit row.
    """

    from app.training.regime_features import compute_macro_regime_features

    vix_values: list[float] = []
    tnx_last: float | None = None
    irx_last: float | None = None
    for bar in bars:
        v = bar.get("vix_close")
        if v is not None:
            try:
                vix_values.append(float(v))
            except (TypeError, ValueError):
                pass
    if bars:
        # Defensive sort by date string -- the events-builder serialises
        # ``prior_bars`` in ascending-date order today, so ``bars[-1]`` is
        # the strict-T-1 bar. The sort makes the contract self-enforcing:
        # a future serialiser change (e.g., reverse-chrono for a frontend
        # display path) would otherwise silently flip ``bars[-1]`` to the
        # oldest bar and feed stale yields into the term-spread sign.
        sortable = [b for b in bars if b.get("date")]
        if sortable:
            last_bar = max(sortable, key=lambda b: str(b.get("date", "")))
        else:
            last_bar = bars[-1]
        try:
            tnx_last = float(last_bar.get("tnx_close", 0.0))
        except (TypeError, ValueError):
            tnx_last = None
        try:
            irx_last = float(last_bar.get("irx_close", 0.0))
        except (TypeError, ValueError):
            irx_last = None
    features = compute_macro_regime_features(
        event_date=event_date,
        mp_surprise_lookup=mp_surprise_lookup,
        prior_bar_vix_values=vix_values,
        t_minus_one_tnx_close=tnx_last,
        t_minus_one_irx_close=irx_last,
    )
    return features.as_list()


# Canonical FOMC voting-member cap. The committee seats 12 voting
# members each year (7 board governors + the NY Fed president + 4
# rotating Reserve Bank presidents). Dividing the raw counts by 12
# keeps the scalars in the same unit-ish band as the other
# RobustScaler-fittable rich-feature axes so the per-fold scaler does
# not need to learn a magnitude-3-OOM scale gap on a 4-vector.
_VOTE_NORM_DIVISOR: float = 12.0

# Dissent-direction sign map. The hawkish / dovish convention matches
# ``mp_surprise_level``: positive = tighter-than-action, negative =
# easier-than-action. Unanimous / unparseable rows collapse to 0.0
# (no signed signal) and the per-row missing flag carries the actual
# "no data" distinction.
_DISSENT_DIRECTION_SIGN: dict[str, float] = {
    "hawkish_dissent": 1.0,
    "dovish_dissent": -1.0,
}


def _compute_vote_features_for_event(
    row: Any,
) -> list[float] | None:
    """Compose the #444 4-vector off the events.parquet vote columns.

    Returns ``None`` when the row carries no parseable vote tally (a
    non-statement event kind, a row with missing ``votes_for``, or a
    pre-#444 events.parquet without the vote columns at all). The
    caller flips the missing flag in that case.

    Output order matches the audit doc: ``[votes_for_norm,
    votes_against_norm, is_unanimous_float, dissent_direction_signed]``.
    """

    raw_votes_for = row.get("votes_for") if hasattr(row, "get") else None
    votes_for = _coerce_finite_float(raw_votes_for)
    if votes_for is None:
        return None
    votes_against = _coerce_finite_float(
        row.get("votes_against") if hasattr(row, "get") else None
    )
    if votes_against is None:
        votes_against = 0.0
    is_unanimous_raw = row.get("is_unanimous") if hasattr(row, "get") else None
    if is_unanimous_raw is None:
        is_unanimous = 1.0 if votes_against == 0.0 else 0.0
    else:
        try:
            is_unanimous = 1.0 if bool(is_unanimous_raw) else 0.0
        except (TypeError, ValueError):
            is_unanimous = 1.0 if votes_against == 0.0 else 0.0
    direction_raw = (
        row.get("dissent_direction") if hasattr(row, "get") else None
    )
    direction_sign = 0.0
    if direction_raw is not None:
        key = str(direction_raw).strip().lower()
        direction_sign = _DISSENT_DIRECTION_SIGN.get(key, 0.0)
    return [
        votes_for / _VOTE_NORM_DIVISOR,
        votes_against / _VOTE_NORM_DIVISOR,
        is_unanimous,
        direction_sign,
    ]


def _read_statement_delta_embedding(
    row: Any,
) -> list[float] | None:
    """Extract the #443 statement-delta embedding off an events.parquet row.

    Returns ``None`` when the column is absent (pre-#443 events.parquet),
    when the row is a non-statement event kind (the builder writes
    ``None``), or when the supervised event is cold-start (no strict-prior
    statement exists, builder also wrote ``None``). The caller flips the
    missing flag in that case.
    """

    if not hasattr(row, "get"):
        return None
    raw = row.get("statement_delta_embedding")
    if raw is None:
        return None
    # Parquet round-trips list[float] columns as numpy arrays; tolerate
    # both shapes.
    try:
        values = list(raw)
    except TypeError:
        return None
    if not values:
        return None
    out: list[float] = []
    for v in values:
        f = _coerce_finite_float(v)
        if f is None:
            return None
        out.append(f)
    return out


def _read_events_frame(package_dir: Path) -> "Any":
    import pandas as pd

    events_path = package_dir / "events.parquet"
    if not events_path.exists():
        raise FileNotFoundError(
            f"events.parquet missing from training package: {package_dir}"
        )
    return pd.read_parquet(events_path)


def _load_llm_feature_lookup(
    training_package_id: str,
) -> dict[str, list[float]]:
    """B1 (#212) loader hook.

    Read the LLM-features cache parquet for the configured training
    package and return a ``text_hash -> one-hot-vector`` lookup. The
    one-hot vector has exactly ``RICH_LLM_FEATURE_DIM`` floats in the
    documented catalogue order.

    Returns ``{}`` when the cache does not exist; the rich-feature
    attachment then leaves every row with the all-zeros block + the
    missing flag set to 1.0.
    """

    from app.data.llm_feature_catalog import CATALOG_VERSION, CATALOG, MODEL_ID

    candidates = (
        Path(f"/data/raw/llm_features/{MODEL_ID}_{CATALOG_VERSION}/{training_package_id}.parquet"),
        Path(f"data/raw/llm_features/{MODEL_ID}_{CATALOG_VERSION}/{training_package_id}.parquet"),
        Path(f"backend/data/raw/llm_features/{MODEL_ID}_{CATALOG_VERSION}/{training_package_id}.parquet"),
    )
    cache_path = next((p for p in candidates if p.exists()), None)
    if cache_path is None:
        return {}

    import pandas as pd

    frame = pd.read_parquet(cache_path)
    # Pre-build the contiguous one-hot slot offsets so the per-row
    # encoding is a single dict lookup + index write.
    feature_levels: list[tuple[str, tuple[str, ...]]] = [
        (f.name, f.levels) for f in CATALOG
    ]
    total_dim = sum(len(levels) for _, levels in feature_levels)

    out: dict[str, list[float]] = {}
    for _, row in frame.iterrows():
        if str(row.get("status", "")) != "ok":
            continue
        text_hash = str(row.get("text_hash", ""))
        if not text_hash:
            continue
        vec = [0.0] * total_dim
        offset = 0
        ok = True
        for feature_name, levels in feature_levels:
            raw_value = row.get(feature_name)
            if not isinstance(raw_value, str) or raw_value not in levels:
                # Defensive: skip this row entirely if any feature is
                # out-of-vocab or missing in the cache. The loader
                # then leaves the row in the all-zeros + missing-flag
                # state, treating the extraction as failed.
                ok = False
                break
            idx = levels.index(raw_value)
            vec[offset + idx] = 1.0
            offset += len(levels)
        if ok:
            out[text_hash] = vec
    return out


def _read_fold_manifest(package_dir: Path) -> dict[str, dict[str, str]]:
    """Return ``fold_id -> {train_start, train_end, val_start, val_end, test_start, test_end}``.

    Reads ``fold_manifest_expanding_walk_forward.json`` from the
    training-package directory. The manifest ships one entry per
    expanding-window fold (``wf_fold_1`` ... ``wf_fold_N``) with the
    chronological date ranges that define the train, val and test
    windows for that fold. Returns an empty dict when the file is
    absent or the schema is unparseable; callers handle that as a
    missing-fold error.
    """

    path = package_dir / "fold_manifest_expanding_walk_forward.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    folds = payload.get("folds") if isinstance(payload, dict) else None
    if not isinstance(folds, list):
        return {}
    result: dict[str, dict[str, str]] = {}
    for fold in folds:
        if not isinstance(fold, dict):
            continue
        fold_id = str(fold.get("fold_id", "")).strip()
        if not fold_id:
            continue
        result[fold_id] = {
            "train_start": str(fold.get("train_start", "")),
            "train_end": str(fold.get("train_end", "")),
            "val_start": str(fold.get("val_start", "")),
            "val_end": str(fold.get("val_end", "")),
            "test_start": str(fold.get("test_start", "")),
            "test_end": str(fold.get("test_end", "")),
        }
    return result


def _read_split_tag_lookup(package_dir: Path) -> dict[str, str]:
    """Return ``text_hash -> split_tag`` from ``splits_train_val_test.parquet``.

    Accepts either ``split_tag`` (current builder column name) or
    ``partition`` (the forward-looking name from the data contract).
    Returns an empty dict when the parquet is absent or the schema
    does not expose a joinable ``text_hash`` column; downstream
    callers then fall through to the legacy "train-only" filter.
    """

    import pandas as pd

    splits_path = package_dir / "splits_train_val_test.parquet"
    if not splits_path.exists():
        return {}
    frame = pd.read_parquet(splits_path)
    tag_column: str | None = None
    for candidate in ("partition", "split_tag"):
        if candidate in frame.columns:
            tag_column = candidate
            break
    if tag_column is None or "text_hash" not in frame.columns:
        return {}
    lookup: dict[str, str] = {}
    for record in frame.to_dict("records"):
        text_hash = str(record.get("text_hash", "")).strip()
        if not text_hash:
            continue
        tag = str(record.get(tag_column, "")).strip().lower()
        if not tag:
            continue
        lookup[text_hash] = tag
    return lookup


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


def _load_package_sequences_with_metadata(
    training_package_id: str,
    *,
    target_mode: TargetMode = DEFAULT_TARGET_MODE,
    rich_features: bool = True,
    use_credibility: bool = True,
    use_linguistic: bool = True,
    use_mp_surprise: bool = True,
    use_multi_axis: bool = True,
    use_llm_features: bool = False,
    use_retrieval_analogs: bool = False,
    use_regime_conditioning: bool = False,
    use_sep: bool = False,
    use_press_conf: bool = False,
    use_statement_delta: bool = False,
    use_vote_features: bool = False,
    text_encoder: str | None = None,
    text_adapter_dim: int = DEFAULT_TEXT_ADAPTER_DIM,
    text_pool_lambda_inv_days: float = DEFAULT_TEXT_POOL_LAMBDA_INV_DAYS,
    use_text_embeddings: bool = True,
    text_embedding_cache_dir: Path | str | None = None,
    encoder_lora: bool = False,
) -> list[tuple[list[FeatureVector], str, str]]:
    """Materialise every event in a training package as a sequence triple.

    Returns ``[(sequence, text_hash, event_date), ...]`` for every event
    whose ``prior_bars_json`` carries the full 20-bar prior window. No
    split-tag or fold filter is applied here -- the caller partitions
    on top of the returned list using either the package's
    ``split_tag`` column or the walk-forward fold manifest.

    Sequences are sorted by ``(event_date, text_hash)`` so the returned
    order is deterministic across runs. Per-event prior-bar / target /
    rich-feature / text-embedding logic is identical to the legacy
    :func:`load_training_sequences_from_package`; only the partition
    step changes.
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

    linguistic_lookup: dict[str, list[float]] = {}
    # The mp_surprise lookup feeds two consumers: rich-feature input
    # columns (gated by ``rich_features``) AND the fomc-attributable
    # rates-target projection (always computed; the trainer reads it
    # off the FeatureVector only when ``rates_target_mode`` opts in).
    # Loading it unconditionally costs one cheap parquet read and
    # prevents the silent all-None projection that would otherwise
    # fire under ``--no-rich-features --rates-target-mode=fomc_attributable``.
    mp_surprise_lookup: dict[str, dict[str, float]] = _read_mp_surprise_lookup(
        package_dir
    )
    # #215 SEP dot-plot lookup. Loaded only when the opt-in flag fires so
    # the legacy / opt-out path stays byte-identical to pre-#215 (a
    # parquet on disk doesn't change behaviour unless --use-sep is set).
    sep_lookup: dict[str, dict[str, Any]] = (
        _read_sep_projections_lookup(package_dir) if use_sep else {}
    )
    # #214 FOMC press-conference Q&A lookup. Loaded only when the opt-in
    # flag fires so the legacy path stays byte-identical to pre-#214 (a
    # parquet on disk doesn't change behaviour unless --use-press-conf
    # is set). Empty dict when no parquet is found — the loader then
    # treats every event as ``has_press_conf=0`` and the LoRA path
    # leaves ``raw_text`` at the statement text alone.
    press_conf_lookup: dict[str, dict[str, Any]] = (
        _read_press_conf_qa_lookup(package_dir) if use_press_conf else {}
    )
    llm_lookup: dict[str, list[float]] = {}
    if rich_features:
        linguistic_lookup = _read_linguistic_lookup(package_dir)
        if use_llm_features:
            llm_lookup = _load_llm_feature_lookup(training_package_id)

    embedding_lookup: dict[str, Any] = {}
    embedding_event_dates: dict[str, str] = {}
    use_text_path = bool(text_encoder) and bool(use_text_embeddings)
    text_adapter_dim_int = int(text_adapter_dim)
    if use_text_path:
        if text_adapter_dim_int <= 0:
            raise ValueError(
                f"text_adapter_dim must be a positive integer; got {text_adapter_dim}"
            )
        cache_dir = (
            Path(text_embedding_cache_dir)
            if text_embedding_cache_dir is not None
            else None
        )
        registry_parquet = package_dir / "registry_normalized.parquet"
        registry_jsonl = package_dir / "registry_normalized.jsonl"
        registry_for_lookup: Path | None
        if registry_jsonl.exists():
            registry_for_lookup = registry_jsonl
        elif registry_parquet.exists():
            import pandas as pd

            reg_frame = pd.read_parquet(registry_parquet)
            reg_subset = reg_frame[[c for c in ("record_id", "text_hash") if c in reg_frame.columns]]
            if not reg_subset.empty and "record_id" in reg_subset.columns and "text_hash" in reg_subset.columns:
                tmp_path = package_dir / "_registry_record_to_text_hash.jsonl"
                with tmp_path.open("w", encoding="utf-8") as handle:
                    for record in reg_subset.to_dict("records"):
                        handle.write(
                            json.dumps(
                                {
                                    "record_id": str(record.get("record_id") or ""),
                                    "text_hash": str(record.get("text_hash") or ""),
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                registry_for_lookup = tmp_path
            else:
                registry_for_lookup = None
        else:
            registry_for_lookup = None
        assert text_encoder is not None  # narrowed by use_text_path
        embedding_lookup, embedding_event_dates = _read_chunk_embedding_lookup(
            text_encoder,
            cache_dir=cache_dir,
            registry_path=registry_for_lookup,
        )
    # ``embedding_event_dates`` is captured for symmetry with the
    # ``_read_chunk_embedding_lookup`` contract; the prior-4 pooler
    # consumes statement dates off ``prior_chronology`` further down,
    # so the dict is not read here. Kept in scope so a future
    # consumer can pick it up without re-reading the cache.
    _ = embedding_event_dates

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
    prior_chronology_set: set[tuple[datetime.date, str]] = set()
    for row in records:
        candidate_hash = str(row.get("text_hash", ""))
        if not candidate_hash:
            continue
        candidate_date_raw = str(row.get("event_date", ""))[:10]
        if not candidate_date_raw:
            continue
        try:
            candidate_date = datetime.date.fromisoformat(candidate_date_raw)
        except ValueError:
            continue
        prior_chronology_set.add((candidate_date, candidate_hash))
        if candidate_hash in seen:
            continue
        seen.add(candidate_hash)
        by_text_hash[candidate_hash] = row

    prior_chronology: list[tuple[datetime.date, str]] = sorted(prior_chronology_set)

    ordered_rows = sorted(
        by_text_hash.values(),
        key=lambda row: (str(row.get("event_date", "")), str(row.get("text_hash", ""))),
    )

    results: list[tuple[list[FeatureVector], str, str]] = []
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
        # Phase 9 V2 (#195) classification target rides on every supervised
        # row regardless of rich-features being on or off -- it is the y
        # axis, not an input feature. Tier 1 (Market-Only) needs it just
        # like tier 3 (Market+Rich+NLP); a missing target would crash the
        # per-fold quantile fit with "0 valid rows for n_classes=3".
        forward_vol_value = _coerce_finite_float(
            row.get("forward_realized_vol_10d")
        )
        # #236 GARCH(1,1) baseline + residual. Frozen into the events
        # parquet at build time (see ``app.data.garch_residual``);
        # the loader only reads them off the row and broadcasts onto
        # the target slot. Older events.parquet files without these
        # columns degrade cleanly to ``None`` here.
        garch_baseline_value = _coerce_finite_float(
            row.get("forward_realized_vol_10d_garch_baseline")
        )
        garch_residual_value = _coerce_finite_float(
            row.get("forward_realized_vol_10d_garch_residual")
        )
        # #292 rates-complex strict-forward 5d yield change targets.
        # Each value rides on the same event row alongside
        # forward_realized_vol_10d; the per-fold target builder
        # (:func:`app.training.rates_targets.build_partition_rates_targets`)
        # reads them off the target row of each supervised sequence.
        rates_2y_value = _coerce_finite_float(row.get("yield_2y_change_5d"))
        rates_5y_value = _coerce_finite_float(row.get("yield_5y_change_5d"))
        rates_terminal_value = _coerce_finite_float(
            row.get("terminal_rate_change_5d")
        )
        # #305 FOMC-attributable rates targets. Compute the 1-D
        # projection of each observed bps move onto the strict-prior
        # surprise direction ``sign(mp_surprise_level)``. The level is
        # read off the strict-prior mp_surprises lookup so the target
        # is leak-clean by construction (ADR 0024 / #350). No-change
        # meetings (|surprise| < epsilon) emit ``None`` so the
        # partition builder masks the row instead of writing a zero
        # label. See ADR 0027 for the projection definition.
        from app.training.rates_targets import fomc_attributable_projection

        mp_level_lookup = mp_surprise_lookup.get(event_date_str[:10], {})
        mp_level_for_projection = _coerce_finite_float(
            mp_level_lookup.get("mp_surprise_level")
        )
        rates_2y_attributable = fomc_attributable_projection(
            rates_2y_value, mp_level_for_projection
        )
        rates_5y_attributable = fomc_attributable_projection(
            rates_5y_value, mp_level_for_projection
        )
        rates_terminal_attributable = fomc_attributable_projection(
            rates_terminal_value, mp_level_for_projection
        )
        # B1 (#212) LLM-features lookup -- one-hot block + missing flag
        # per event row. Lookup is built once per package outside the
        # loop. Hashes absent from the lookup (failed extraction or
        # below-min-chars document) leave the row in the all-zeros +
        # missing=1.0 state so the rich-feature input shape stays
        # constant regardless of extraction coverage.
        llm_vector = llm_lookup.get(row_text_hash)
        # #306 retrieval-augmented analog summary block. The lookup is
        # gated by ``use_retrieval_analogs`` so the legacy path stays
        # byte-identical when the flag is off; the helper enforces the
        # strict-backward ``analog_event_date < event_date`` filter on
        # the retrieval call so an analog row whose own event_date ties
        # or post-dates the supervised event is never eligible. A bundle
        # absent on disk returns ``None`` and the loader collapses every
        # event to the all-zeros + missing-flag-1.0 state (graceful
        # degrade). See ADR 0028.
        if use_retrieval_analogs:
            event_text = str(row.get("text", "") or "")
            analog_features_list = _compute_analog_features_for_event(
                event_text=event_text,
                event_date=event_date,
                event_stance=row.get("axis_stance"),
            )
        else:
            analog_features_list = None
        # #307 macro-regime block. Gated by ``use_regime_conditioning`` so
        # the legacy path stays byte-identical when the flag is off; the
        # helper composes the three strict-prior scalars off the MP-
        # surprise lookup (policy-cycle phase) and the per-event prior
        # bars (VIX tertile + 10y-3m spread sign). No new I/O, the data
        # is already in scope. See ADR 0029.
        if use_regime_conditioning:
            regime_block_list = _compute_macro_regime_features_for_event(
                event_date=event_date,
                bars=bars,
                mp_surprise_lookup=mp_surprise_lookup,
            )
        else:
            regime_block_list = None
        # #215 SEP dot-plot block. Gated by ``use_sep`` so the legacy
        # path stays byte-identical when the flag is off; the helper
        # composes the 5-scalar block off the SEP-projections lookup,
        # forward-filling the most recent prior SEP on non-SEP meetings
        # and stamping ``sep_release_flag=1.0`` only on the meeting that
        # actually refreshed the projections.
        if use_sep:
            sep_block_list = _compute_sep_features_for_event(
                event_date=event_date,
                sep_lookup=sep_lookup,
            )
        else:
            sep_block_list = None
        # #214 FOMC press-conf Q&A block. Composed unconditionally when
        # the flag is on so pre-2011 events land with ``has_press_conf=0``
        # and post-2011 events with a Q&A transcript land with
        # ``has_press_conf=1`` — the zero-impute covariate-shift handling
        # rejected fragmenting the walk-forward fold protocol for an
        # era-specific subset (see ADR 0037).
        if use_press_conf:
            press_conf_block_list = _compute_press_conf_features_for_event(
                event_date_str=event_date_str,
                press_conf_lookup=press_conf_lookup,
            )
        else:
            press_conf_block_list = None
        # #443 statement-delta embedding. Gated by ``use_statement_delta``
        # so the legacy path stays byte-identical when the flag is off.
        # Cold-start rows (no strict-prior statement available) and
        # non-statement event kinds carry ``None`` on the events.parquet
        # column; the loader collapses to the missing-1.0 slot.
        if use_statement_delta:
            statement_delta_list = _read_statement_delta_embedding(row)
        else:
            statement_delta_list = None
        # #444 vote-tally feature block. Gated by ``use_vote_features``;
        # missing column / non-statement / unparseable row → None and
        # the missing flag fires.
        if use_vote_features:
            vote_features_list = _compute_vote_features_for_event(row)
        else:
            vote_features_list = None
        for vector in vectors:
            vector.forward_realized_vol_10d = forward_vol_value
            vector.forward_realized_vol_10d_garch_baseline = garch_baseline_value
            vector.forward_realized_vol_10d_garch_residual = garch_residual_value
            vector.target_yield_2y_change_5d = rates_2y_value
            vector.target_yield_5y_change_5d = rates_5y_value
            vector.target_terminal_rate_change_5d = rates_terminal_value
            vector.target_yield_2y_change_5d_fomc_attributable = rates_2y_attributable
            vector.target_yield_5y_change_5d_fomc_attributable = rates_5y_attributable
            vector.target_terminal_rate_change_5d_fomc_attributable = (
                rates_terminal_attributable
            )
            if llm_vector is not None:
                vector.llm_features = list(llm_vector)
                vector.llm_features_missing = 0.0
            else:
                vector.llm_features = None
                vector.llm_features_missing = 1.0
            if analog_features_list is not None:
                vector.analog_features = list(analog_features_list)
                vector.analog_features_missing = 0.0
            else:
                vector.analog_features = None
                vector.analog_features_missing = 1.0
            # #307 macro-regime block broadcast onto every bar. When the
            # flag is off, ``regime_block_list`` is ``None`` and the
            # slot stays at the all-zeros + missing=1.0 default; the
            # conditional emission in ``FeatureVector.as_rich_list``
            # then skips appending the block entirely, preserving the
            # byte-identical pre-#307 per-bar feature size.
            if regime_block_list is not None:
                vector.macro_regime_features = list(regime_block_list)
                vector.macro_regime_features_missing = 0.0
            else:
                vector.macro_regime_features = None
                vector.macro_regime_features_missing = 1.0
            # #215 SEP block broadcast onto every bar. Same conditional-
            # emission contract as the regime block: a ``None`` slot
            # keeps ``as_rich_list`` from appending anything, so the
            # default flag-off path preserves byte-identical pre-#215
            # per-bar feature size. The block is appended after the
            # regime block when both are populated.
            if sep_block_list is not None:
                vector.sep_features = list(sep_block_list)
                vector.sep_features_missing = 0.0
            else:
                vector.sep_features = None
                vector.sep_features_missing = 1.0
            # #214 press-conf Q&A block broadcast onto every bar.
            if press_conf_block_list is not None:
                vector.press_conf_features = list(press_conf_block_list)
            else:
                vector.press_conf_features = None
            # #443 statement-delta embedding broadcast.
            if statement_delta_list is not None:
                vector.statement_delta_embedding = list(statement_delta_list)
                vector.statement_delta_embedding_missing = 0.0
            else:
                vector.statement_delta_embedding = None
                vector.statement_delta_embedding_missing = 1.0
            # #444 vote-tally feature broadcast.
            if vote_features_list is not None:
                vector.vote_features = list(vote_features_list)
                vector.vote_features_missing = 0.0
            else:
                vector.vote_features = None
                vector.vote_features_missing = 1.0
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
        if use_text_path:
            pooled = _compute_prior4_pooled_embedding(
                text_hash=row_text_hash,
                event_row_text_hash=row_text_hash,
                current_event_date=event_date,
                embedding_lookup=embedding_lookup,
                prior_text_hashes=prior_chronology,
                lambda_inv_days=float(text_pool_lambda_inv_days),
                max_prior=4,
            )
            if pooled is None:
                missing_flag = 1.0
                pooled_list: list[float] = []
            else:
                missing_flag = 0.0
                pooled_list = [float(v) for v in pooled.tolist()]
            for vector in vectors:
                vector.text_embedding_pooled = list(pooled_list)
                vector.text_embedding_missing = missing_flag
        if encoder_lora:
            # Round 5 (#244) per-event raw text for in-loop LoRA
            # tokenisation. Only the target-row bar (last in the
            # sequence after ``_append_event_day_target``) carries the
            # text -- the lookback bars do not need it since the
            # tokeniser reads ``sequence[-1].raw_text`` per sequence.
            # The text comes from the event's own ``text`` column,
            # not from the prior-4 statement pool, so the LoRA path
            # learns gradients w.r.t. the event's actual content
            # (statement / minutes / press conference / scrape).
            row_text = str(row.get("text", "") or "").strip()
            # #214 route 1: when the press-conf opt-in is on AND the
            # supervised event is the FOMC statement, concat the
            # same-date Q&A onto the statement text so the LoRA
            # encoder sees a single joint document per route 1 of the
            # scope brief. The press conference itself rides on a
            # separate event_kind row in events.parquet and is left
            # untouched (the encoder learns the Q&A signal off the
            # statement row's joint text, not by training twice on the
            # same Q&A). The append is conditional on a non-empty
            # ``qa_text`` lookup hit; missing-Q&A statement rows
            # collapse to the byte-identical pre-#214 raw_text.
            if use_press_conf and str(row.get("event_kind", "")) == "statement":
                pc_record = press_conf_lookup.get(event_date_str[:10])
                if pc_record:
                    qa_text = str(pc_record.get("qa_text") or "").strip()
                    if qa_text:
                        row_text = f"{row_text}\n\n{qa_text}" if row_text else qa_text
            if vectors:
                vectors[-1].raw_text = row_text
        results.append((vectors, row_text_hash, event_date_str[:10]))
    return results


def load_walk_forward_split(
    training_package_id: str,
    *,
    fold_id: str | None = None,
    target_mode: TargetMode = DEFAULT_TARGET_MODE,
    rich_features: bool = True,
    use_credibility: bool = True,
    use_linguistic: bool = True,
    use_mp_surprise: bool = True,
    use_multi_axis: bool = True,
    use_llm_features: bool = False,
    use_retrieval_analogs: bool = False,
    use_regime_conditioning: bool = False,
    use_sep: bool = False,
    use_press_conf: bool = False,
    use_statement_delta: bool = False,
    use_vote_features: bool = False,
    text_encoder: str | None = None,
    text_adapter_dim: int = DEFAULT_TEXT_ADAPTER_DIM,
    text_pool_lambda_inv_days: float = DEFAULT_TEXT_POOL_LAMBDA_INV_DAYS,
    use_text_embeddings: bool = True,
    text_embedding_cache_dir: Path | str | None = None,
    embargo_days: int = 0,
    encoder_lora: bool = False,
) -> WalkForwardSplit:
    """Return the (train, val, test) sequence partitions for one fold.

    Two protocols are supported:

    - ``fold_id=None`` (default, "single-fold"): partition events by
      the ``split_tag`` column on ``splits_train_val_test.parquet``.
      Rows tagged ``train`` enter the training list; ``val`` rows
      drive early stopping; ``test`` rows are the held-out evaluation
      set. The ``excluded_from_training`` sentinel (when present) is
      treated as a fourth bucket and dropped from all three lists.
    - ``fold_id="wf_fold_K"`` (multi-fold "walk-forward"): partition
      events by chronological date against the fold's manifest entry
      in ``fold_manifest_expanding_walk_forward.json``. Every event
      with ``event_date < val_start`` belongs to the training list
      (expanding-window contract: the train partition grows fold over
      fold). The val list spans ``[val_start, val_end]``; the test
      list spans ``[test_start, test_end]``.

    The test partition is the canonical held-out set the forecaster
    sweep aggregator reports as ``test_rmse``; the val partition is
    consumed by the training loop for early stopping. Both paths
    preserve the prior-bars + rich-feature + text-embedding attachment
    logic; only the partition step changes.

    ``embargo_days`` inserts a purged buffer between adjacent partitions
    (López de Prado, *Advances in Financial ML*, ch. 7). With a
    10-bar forward target and a ``SEQUENCE_LENGTH``-bar input window,
    consecutive event rows can share bars across the train/val
    boundary: train_event_T's 10-day forward target spans bars
    [T+1, T+10] and val_event_V's input window spans bars
    [V - SEQUENCE_LENGTH + 1, V], so the two windows overlap whenever
    V - T <= SEQUENCE_LENGTH + 10 - 1 calendar bars. Setting
    ``embargo_days`` drops val rows whose event date sits within
    ``embargo_days`` calendar days of the fold's ``train_end``, and
    test rows within ``embargo_days`` of ``val_end``. The function
    default is ``0`` (back-compat); ``app.train_forecaster`` and
    other production callers pass a non-zero value via their
    ``--embargo-days`` CLI flag. ``embargo_days`` is currently honoured
    only on the multi-fold walk-forward path (single-fold partitions
    are split-tag-driven and have no manifest dates to anchor the
    embargo against).

    Raises ``ValueError`` when the chosen partition produces an empty
    test list: a sweep that silently runs on no held-out events is the
    research-quality footgun this refactor was written to remove.
    """

    package_dir = _resolve_training_package_dir(training_package_id)
    fold_window: dict[str, str] | None = None
    if fold_id is not None:
        manifest = _read_fold_manifest(package_dir)
        if fold_id not in manifest:
            raise ValueError(
                f"fold_id={fold_id!r} not found in fold manifest at {package_dir}; "
                f"known fold ids: {sorted(manifest.keys())}"
            )
        fold_window = manifest[fold_id]

    items = _load_package_sequences_with_metadata(
        training_package_id,
        target_mode=target_mode,
        rich_features=rich_features,
        use_credibility=use_credibility,
        use_linguistic=use_linguistic,
        use_mp_surprise=use_mp_surprise,
        use_multi_axis=use_multi_axis,
        use_llm_features=use_llm_features,
        use_retrieval_analogs=use_retrieval_analogs,
        use_regime_conditioning=use_regime_conditioning,
        use_sep=use_sep,
        use_press_conf=use_press_conf,
        use_statement_delta=use_statement_delta,
        use_vote_features=use_vote_features,
        text_encoder=text_encoder,
        text_adapter_dim=text_adapter_dim,
        text_pool_lambda_inv_days=text_pool_lambda_inv_days,
        use_text_embeddings=use_text_embeddings,
        text_embedding_cache_dir=text_embedding_cache_dir,
        encoder_lora=encoder_lora,
    )

    train: list[list[FeatureVector]] = []
    val: list[list[FeatureVector]] = []
    test: list[list[FeatureVector]] = []
    train_dates: list[str] = []
    val_dates: list[str] = []
    test_dates: list[str] = []

    if fold_window is None:
        # Single-fold path: partition by split_tag.
        protocol = "single-fold"
        tag_lookup = _read_split_tag_lookup(package_dir)
        for sequence, text_hash, event_date_str in items:
            tag = tag_lookup.get(text_hash, "").lower()
            if tag == "train":
                train.append(sequence)
                train_dates.append(event_date_str)
            elif tag == "val":
                val.append(sequence)
                val_dates.append(event_date_str)
            elif tag == "test":
                test.append(sequence)
                test_dates.append(event_date_str)
            else:
                # Untagged or excluded -- skip from every partition.
                continue
    else:
        protocol = "walk-forward"
        train_end_str = fold_window.get("train_end", "")
        val_start = fold_window.get("val_start", "")
        val_end = fold_window.get("val_end", "")
        test_start = fold_window.get("test_start", "")
        test_end = fold_window.get("test_end", "")
        if not (val_start and val_end and test_start and test_end):
            raise ValueError(
                f"fold {fold_id!r} manifest entry missing one of "
                "val_start/val_end/test_start/test_end"
            )
        embargo_active = int(embargo_days) > 0
        if embargo_active and not train_end_str:
            raise ValueError(
                f"fold {fold_id!r} manifest entry is missing ``train_end``; "
                "cannot apply a non-zero embargo without an anchored "
                "train-boundary date"
            )
        train_end_dt = (
            datetime.date.fromisoformat(train_end_str) if embargo_active else None
        )
        val_end_dt = (
            datetime.date.fromisoformat(val_end) if embargo_active else None
        )
        embargo = datetime.timedelta(days=int(embargo_days))
        for sequence, _text_hash, event_date_str in items:
            # Expanding-window contract: any event chronologically
            # before the val window belongs to the training partition,
            # so train_k strictly grows with k.
            if event_date_str < val_start:
                train.append(sequence)
                train_dates.append(event_date_str)
            elif val_start <= event_date_str <= val_end:
                if embargo_active and train_end_dt is not None:
                    ed = datetime.date.fromisoformat(event_date_str)
                    if (ed - train_end_dt) < embargo:
                        # Purged: too close to train_end. Drops the
                        # row to break the train-target / val-input
                        # bar-window overlap; see docstring for the
                        # exact arithmetic.
                        continue
                val.append(sequence)
                val_dates.append(event_date_str)
            elif test_start <= event_date_str <= test_end:
                if embargo_active and val_end_dt is not None:
                    ed = datetime.date.fromisoformat(event_date_str)
                    if (ed - val_end_dt) < embargo:
                        continue
                test.append(sequence)
                test_dates.append(event_date_str)
            else:
                # Falls into the post-test gap (between the fold's
                # test_end and the next fold's val_start) -- drop.
                continue

    if not test:
        raise ValueError(
            f"WalkForwardSplit for training_package_id={training_package_id!r} "
            f"fold_id={fold_id!r} produced an empty test partition; refusing "
            "to silently train on no held-out events"
        )

    return WalkForwardSplit(
        train=train,
        val=val,
        test=test,
        train_event_dates=train_dates,
        val_event_dates=val_dates,
        test_event_dates=test_dates,
        fold_id=fold_id,
        protocol=protocol,
    )


def load_training_sequences_from_package(
    training_package_id: str,
    *,
    target_mode: TargetMode = DEFAULT_TARGET_MODE,
    rich_features: bool = True,
    use_credibility: bool = True,
    use_linguistic: bool = True,
    use_mp_surprise: bool = True,
    use_multi_axis: bool = True,
    use_llm_features: bool = False,
    use_retrieval_analogs: bool = False,
    use_regime_conditioning: bool = False,
    use_sep: bool = False,
    use_press_conf: bool = False,
    use_statement_delta: bool = False,
    use_vote_features: bool = False,
    text_encoder: str | None = None,
    text_adapter_dim: int = DEFAULT_TEXT_ADAPTER_DIM,
    text_pool_lambda_inv_days: float = DEFAULT_TEXT_POOL_LAMBDA_INV_DAYS,
    use_text_embeddings: bool = True,
    text_embedding_cache_dir: Path | str | None = None,
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

    When ``text_encoder`` is set and ``use_text_embeddings`` is True
    the loader pulls the per-statement embeddings from
    ``data/raw/embeddings/<encoder>_<rev>.parquet`` (built by
    ``app.data.embedding_cache``), pools the four most recent prior
    statements with ``softmax(-Delta t_days / text_pool_lambda_inv_days)``
    weights, and attaches the resulting ``in_dim``-vector to every
    bar of the prior window plus the event-day target frame as
    ``FeatureVector.text_embedding_pooled``. The pooled vector stays
    encoder-native (FinBERT 768, voyage-finance-2 1024, ...); the
    adapter projection to ``text_adapter_dim`` runs inside the model
    forward so the recurrent core sees a fixed per-bar feature size.
    When fewer than one prior statement is available (e.g. the first
    event in the corpus), the pooled vector is filled with zeros and
    ``FeatureVector.text_embedding_missing`` flips to ``1.0`` so the
    model can tell "no prior signal" apart from "neutral prior
    signal". Setting ``text_encoder=None`` or
    ``use_text_embeddings=False`` skips the pooling step entirely and
    every row keeps the default empty pooled list + missing-flag at
    ``1.0``.

    .. deprecated::
        Returning only the training partition collapses the package's
        ``val`` and ``test`` partitions back into "drop on the floor"
        semantics, which is the bug that motivated
        :func:`load_walk_forward_split`. New callers should use the
        latter and consume the three lists separately. This wrapper
        stays callable so the legacy regression-test contract holds
        until in-tree callers migrate.
    """

    warnings.warn(
        "load_training_sequences_from_package returns only the train "
        "partition and drops val + test on the floor; use "
        "load_walk_forward_split() for proper train/val/test partitions.",
        DeprecationWarning,
        stacklevel=2,
    )

    if target_mode not in _VALID_TARGET_MODES:
        raise ValueError(
            f"Unsupported target_mode: {target_mode!r}. "
            f"Choose one of {sorted(_VALID_TARGET_MODES)}."
        )

    package_dir = _resolve_training_package_dir(training_package_id)
    if not (package_dir / "events.parquet").exists():
        raise FileNotFoundError(
            f"events.parquet missing from training package: {package_dir}"
        )
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
    # See the matching note on `_load_package_sequences_with_metadata`:
    # the mp_surprise lookup feeds both rich-feature input columns
    # (gated) and the fomc-attributable rates-target projection (always
    # computed). Load unconditionally so `--no-rich-features` does not
    # silently null every projection.
    mp_surprise_lookup: dict[str, dict[str, float]] = _read_mp_surprise_lookup(
        package_dir
    )
    # #215 SEP lookup (legacy loader path mirror -- see the matched site
    # in ``_load_package_sequences_with_metadata``).
    sep_lookup: dict[str, dict[str, Any]] = (
        _read_sep_projections_lookup(package_dir) if use_sep else {}
    )
    # #214 press-conf Q&A lookup mirror on the legacy loader path. Same
    # contract as the metadata loader: empty dict when the parquet is
    # absent on disk, so the composer collapses every event to
    # ``has_press_conf=0``.
    press_conf_lookup: dict[str, dict[str, Any]] = (
        _read_press_conf_qa_lookup(package_dir) if use_press_conf else {}
    )
    llm_lookup: dict[str, list[float]] = {}
    if rich_features:
        linguistic_lookup = _read_linguistic_lookup(package_dir)
        if use_llm_features:
            llm_lookup = _load_llm_feature_lookup(training_package_id)

    # Text-embedding lookup. Loaded once per package; the per-event
    # softmax-weighted pool runs against this dict. When the encoder
    # is unset or the parquet is missing, the lookup stays empty and
    # ``_compute_prior4_pooled_embedding`` returns None for every
    # event so the model sees the zero + missing-flag pair.
    embedding_lookup: dict[str, Any] = {}
    embedding_event_dates: dict[str, str] = {}
    use_text_path = bool(text_encoder) and bool(use_text_embeddings)
    text_adapter_dim_int = int(text_adapter_dim)
    if use_text_path:
        if text_adapter_dim_int <= 0:
            raise ValueError(
                f"text_adapter_dim must be a positive integer; got {text_adapter_dim}"
            )
        cache_dir = (
            Path(text_embedding_cache_dir)
            if text_embedding_cache_dir is not None
            else None
        )
        # The embedding cache builder keys rows on ``record_id``;
        # events.parquet joins on ``text_hash``. The registry parquet
        # under the training package carries both columns, so the
        # lookup walks it once to materialise the record_id ->
        # text_hash join. When the registry is absent the lookup
        # falls back to record_id keys and the per-event join
        # silently misses every row -- that's logged via the
        # missing-flag count downstream.
        registry_parquet = package_dir / "registry_normalized.parquet"
        registry_jsonl = package_dir / "registry_normalized.jsonl"
        registry_for_lookup: Path | None
        if registry_jsonl.exists():
            registry_for_lookup = registry_jsonl
        elif registry_parquet.exists():
            # When only the parquet is available, materialise a
            # registry_normalized.jsonl-style view of the
            # ``record_id`` / ``text_hash`` columns in-memory and
            # ship that to the lookup helper via a tmp file. The
            # parquet schema carries the same columns so the join
            # remains exact.
            import pandas as pd

            reg_frame = pd.read_parquet(registry_parquet)
            reg_subset = reg_frame[[c for c in ("record_id", "text_hash") if c in reg_frame.columns]]
            if not reg_subset.empty and "record_id" in reg_subset.columns and "text_hash" in reg_subset.columns:
                tmp_path = package_dir / "_registry_record_to_text_hash.jsonl"
                with tmp_path.open("w", encoding="utf-8") as handle:
                    for record in reg_subset.to_dict("records"):
                        handle.write(
                            json.dumps(
                                {
                                    "record_id": str(record.get("record_id") or ""),
                                    "text_hash": str(record.get("text_hash") or ""),
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                registry_for_lookup = tmp_path
            else:
                registry_for_lookup = None
        else:
            registry_for_lookup = None
        assert text_encoder is not None  # narrowed by use_text_path
        embedding_lookup, embedding_event_dates = _read_chunk_embedding_lookup(
            text_encoder,
            cache_dir=cache_dir,
            registry_path=registry_for_lookup,
        )
    # Captured for symmetry; the prior-4 pooler reads dates off
    # ``prior_chronology`` so the dict is not consumed here.
    _ = embedding_event_dates

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
    # Track every text_hash + event_date seen in the package -- INCLUDING
    # rows excluded from the training loss -- so the prior-4 statement
    # pool sees the full historical chronology when scoring a training
    # event. A val / test statement that chronologically precedes a
    # train statement is still a "prior" for that train statement and
    # contributes to the pooled embedding.
    prior_chronology_set: set[tuple[datetime.date, str]] = set()
    for row in records:
        candidate_hash = str(row.get("text_hash", ""))
        if not candidate_hash:
            continue
        candidate_date_raw = str(row.get("event_date", ""))[:10]
        if not candidate_date_raw:
            continue
        try:
            candidate_date = datetime.date.fromisoformat(candidate_date_raw)
        except ValueError:
            continue
        prior_chronology_set.add((candidate_date, candidate_hash))
        if candidate_hash in excluded_text_hashes:
            continue
        if candidate_hash in seen:
            continue
        seen.add(candidate_hash)
        by_text_hash[candidate_hash] = row

    prior_chronology: list[tuple[datetime.date, str]] = sorted(prior_chronology_set)

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
        # Phase 9 V2 (#195) classification target rides on every supervised
        # row regardless of rich-features being on or off -- it is the y
        # axis, not an input feature. Tier 1 (Market-Only) needs it just
        # like tier 3 (Market+Rich+NLP); a missing target would crash the
        # per-fold quantile fit with "0 valid rows for n_classes=3".
        forward_vol_value = _coerce_finite_float(
            row.get("forward_realized_vol_10d")
        )
        # #236 GARCH(1,1) baseline + residual (see matched walk-forward
        # site above for the contract).
        garch_baseline_value = _coerce_finite_float(
            row.get("forward_realized_vol_10d_garch_baseline")
        )
        garch_residual_value = _coerce_finite_float(
            row.get("forward_realized_vol_10d_garch_residual")
        )
        # #292 rates-complex strict-forward 5d yield change targets.
        # Each value rides on the same event row alongside
        # forward_realized_vol_10d; the per-fold target builder
        # (:func:`app.training.rates_targets.build_partition_rates_targets`)
        # reads them off the target row of each supervised sequence.
        rates_2y_value = _coerce_finite_float(row.get("yield_2y_change_5d"))
        rates_5y_value = _coerce_finite_float(row.get("yield_5y_change_5d"))
        rates_terminal_value = _coerce_finite_float(
            row.get("terminal_rate_change_5d")
        )
        # #305 FOMC-attributable projections (see ADR 0027 + the matched
        # block above on the walk-forward loader path).
        from app.training.rates_targets import fomc_attributable_projection

        mp_level_lookup = mp_surprise_lookup.get(event_date_str[:10], {})
        mp_level_for_projection = _coerce_finite_float(
            mp_level_lookup.get("mp_surprise_level")
        )
        rates_2y_attributable = fomc_attributable_projection(
            rates_2y_value, mp_level_for_projection
        )
        rates_5y_attributable = fomc_attributable_projection(
            rates_5y_value, mp_level_for_projection
        )
        rates_terminal_attributable = fomc_attributable_projection(
            rates_terminal_value, mp_level_for_projection
        )
        # B1 (#212) LLM-features lookup -- one-hot block + missing flag
        # per event row. Lookup is built once per package outside the
        # loop. Hashes absent from the lookup (failed extraction or
        # below-min-chars document) leave the row in the all-zeros +
        # missing=1.0 state so the rich-feature input shape stays
        # constant regardless of extraction coverage.
        llm_vector = llm_lookup.get(row_text_hash)
        # #306 per-event retrieval (see matched block in
        # `_load_package_sequences_with_metadata`). Gated by
        # ``use_retrieval_analogs``; absent bundle returns None and
        # collapses to all-zeros + missing-flag-1.0 (graceful degrade).
        if use_retrieval_analogs:
            event_text_legacy = str(row.get("text", "") or "")
            analog_features_list = _compute_analog_features_for_event(
                event_text=event_text_legacy,
                event_date=event_date,
                event_stance=row.get("axis_stance"),
            )
        else:
            analog_features_list = None
        # #307 macro-regime block (matched site to the walk-forward
        # loader path). Default off keeps the legacy path byte-identical.
        if use_regime_conditioning:
            regime_block_list = _compute_macro_regime_features_for_event(
                event_date=event_date,
                bars=bars,
                mp_surprise_lookup=mp_surprise_lookup,
            )
        else:
            regime_block_list = None
        # #215 SEP block (matched site to the walk-forward loader path).
        if use_sep:
            sep_block_list = _compute_sep_features_for_event(
                event_date=event_date,
                sep_lookup=sep_lookup,
            )
        else:
            sep_block_list = None
        # #214 FOMC press-conf Q&A block. Composed unconditionally when
        # the flag is on so pre-2011 events land with ``has_press_conf=0``
        # and post-2011 events with a Q&A transcript land with
        # ``has_press_conf=1`` — the zero-impute covariate-shift handling
        # rejected fragmenting the walk-forward fold protocol for an
        # era-specific subset (see ADR 0037).
        if use_press_conf:
            press_conf_block_list = _compute_press_conf_features_for_event(
                event_date_str=event_date_str,
                press_conf_lookup=press_conf_lookup,
            )
        else:
            press_conf_block_list = None
        # #443 statement-delta embedding. Gated by ``use_statement_delta``
        # so the legacy path stays byte-identical when the flag is off.
        # Cold-start rows (no strict-prior statement available) and
        # non-statement event kinds carry ``None`` on the events.parquet
        # column; the loader collapses to the missing-1.0 slot.
        if use_statement_delta:
            statement_delta_list = _read_statement_delta_embedding(row)
        else:
            statement_delta_list = None
        # #444 vote-tally feature block. Gated by ``use_vote_features``;
        # missing column / non-statement / unparseable row → None and
        # the missing flag fires.
        if use_vote_features:
            vote_features_list = _compute_vote_features_for_event(row)
        else:
            vote_features_list = None
        for vector in vectors:
            vector.forward_realized_vol_10d = forward_vol_value
            vector.forward_realized_vol_10d_garch_baseline = garch_baseline_value
            vector.forward_realized_vol_10d_garch_residual = garch_residual_value
            vector.target_yield_2y_change_5d = rates_2y_value
            vector.target_yield_5y_change_5d = rates_5y_value
            vector.target_terminal_rate_change_5d = rates_terminal_value
            vector.target_yield_2y_change_5d_fomc_attributable = rates_2y_attributable
            vector.target_yield_5y_change_5d_fomc_attributable = rates_5y_attributable
            vector.target_terminal_rate_change_5d_fomc_attributable = (
                rates_terminal_attributable
            )
            if llm_vector is not None:
                vector.llm_features = list(llm_vector)
                vector.llm_features_missing = 0.0
            else:
                vector.llm_features = None
                vector.llm_features_missing = 1.0
            if analog_features_list is not None:
                vector.analog_features = list(analog_features_list)
                vector.analog_features_missing = 0.0
            else:
                vector.analog_features = None
                vector.analog_features_missing = 1.0
            # #307 macro-regime block broadcast onto every bar (see the
            # matched walk-forward loader site for the full contract).
            if regime_block_list is not None:
                vector.macro_regime_features = list(regime_block_list)
                vector.macro_regime_features_missing = 0.0
            else:
                vector.macro_regime_features = None
                vector.macro_regime_features_missing = 1.0
            # #215 SEP block broadcast onto every bar (matched site to
            # the walk-forward loader path). Same conditional-emission
            # contract; flag off keeps the byte-identical pre-#215
            # per-bar feature size.
            if sep_block_list is not None:
                vector.sep_features = list(sep_block_list)
                vector.sep_features_missing = 0.0
            else:
                vector.sep_features = None
                vector.sep_features_missing = 1.0
            # #214 press-conf Q&A block broadcast onto every bar.
            if press_conf_block_list is not None:
                vector.press_conf_features = list(press_conf_block_list)
            else:
                vector.press_conf_features = None
            # #443 statement-delta embedding broadcast.
            if statement_delta_list is not None:
                vector.statement_delta_embedding = list(statement_delta_list)
                vector.statement_delta_embedding_missing = 0.0
            else:
                vector.statement_delta_embedding = None
                vector.statement_delta_embedding_missing = 1.0
            # #444 vote-tally feature broadcast.
            if vote_features_list is not None:
                vector.vote_features = list(vote_features_list)
                vector.vote_features_missing = 0.0
            else:
                vector.vote_features = None
                vector.vote_features_missing = 1.0
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
        if use_text_path:
            pooled = _compute_prior4_pooled_embedding(
                text_hash=row_text_hash,
                event_row_text_hash=row_text_hash,
                current_event_date=event_date,
                embedding_lookup=embedding_lookup,
                prior_text_hashes=prior_chronology,
                lambda_inv_days=float(text_pool_lambda_inv_days),
                max_prior=4,
            )
            if pooled is None:
                # Either the first event in the corpus (no prior to
                # pool) or the encoder cache is missing/empty. Emit
                # zeros + flip the missing flag so the model sees
                # "no prior signal" rather than a hallucinated mean.
                missing_flag = 1.0
                pooled_list: list[float] = []
            else:
                missing_flag = 0.0
                pooled_list = [float(v) for v in pooled.tolist()]
            for vector in vectors:
                vector.text_embedding_pooled = list(pooled_list)
                vector.text_embedding_missing = missing_flag
        sequences.append(vectors)
    return sequences


def fit_rich_feature_scaler_tensor(
    train_x: "torch.Tensor",
    *,
    epsilon: float = 1e-6,
) -> "RichFeatureScalerParams | None":
    """Fit per-column median + IQR over positions [FEATURE_SIZE:RICH_FEATURE_SIZE].

    Operates on the TRAIN tensor only -- never call on val / test rows.
    The walk-forward and legacy paths in ``app.training.loop`` both
    honour this by fitting before any val / test transform is applied.
    Quantiles are estimated over the full (n_windows x sequence_length)
    population so each feature gets the maximum statistical mass; at
    the small corpus size the autocorrelation tax is well under the
    sparsity tax on the linguistic right tail.

    Returns ``None`` when the tensor isn't rich-payload (input_dim
    != ``RICH_FEATURE_SIZE``). Keeps the legacy 6-feature training
    path byte-identical for the existing regression contract at
    ``tests/regression/test_forecaster_determinism.py``.

    Constant columns (IQR < ``epsilon`` on the train slice) get their
    IQR coerced to ``1.0`` so the transform reduces to a pure
    centering step. Catches the placeholder
    ``credibility_market_implied_gap`` (always 0.0 by contract today)
    and any per-family ablation that zeros a slot before fit time.
    """

    import numpy as np
    from datetime import datetime, timezone

    if train_x is None or train_x.numel() == 0:
        return None
    # #307 widens the per-bar tensor past ``RICH_FEATURE_SIZE`` when the
    # macro-regime block is attached. The scaler still fits + transforms
    # only the legacy ``[FEATURE_SIZE:RICH_FEATURE_SIZE]`` slice -- the
    # regime block (three signed scalars in {-1, 0, +1}) is already in a
    # tight range and rides the model gate at its raw scale. Any tensor
    # narrower than ``RICH_FEATURE_SIZE`` is the legacy 6-feature path
    # the determinism regression pins, so the scaler returns ``None`` and
    # the legacy training path stays byte-identical.
    if train_x.shape[-1] < RICH_FEATURE_SIZE:
        return None

    input_dim = int(train_x.shape[-1])
    flat = train_x.reshape(-1, input_dim)
    rich_block = flat[:, FEATURE_SIZE:RICH_FEATURE_SIZE].detach().cpu().numpy()
    medians = np.median(rich_block, axis=0)
    q1 = np.quantile(rich_block, 0.25, axis=0)
    q3 = np.quantile(rich_block, 0.75, axis=0)
    iqrs = q3 - q1
    iqrs = np.where(iqrs < epsilon, 1.0, iqrs)
    return RichFeatureScalerParams(
        medians=tuple(float(v) for v in medians.tolist()),
        iqrs=tuple(float(v) for v in iqrs.tolist()),
        epsilon=float(epsilon),
        fitted_at_utc=datetime.now(timezone.utc).isoformat(),
        n_train_observations=int(rich_block.shape[0]),
    )


def apply_rich_feature_scaler_tensor(
    x: "torch.Tensor",
    scaler: "RichFeatureScalerParams | None",
) -> "torch.Tensor":
    """Apply ``(x - median) / iqr`` to positions [FEATURE_SIZE:RICH_FEATURE_SIZE].

    No-op when ``scaler`` is ``None`` or when the tensor isn't
    rich-payload (input_dim != ``RICH_FEATURE_SIZE``). The market
    block [0:FEATURE_SIZE] passes through untouched -- the
    ``close_scale`` fitted in :func:`fit_close_scale` is the only
    transform on the legacy six positions.
    """

    if scaler is None or x is None or x.numel() == 0:
        return x
    # See ``fit_rich_feature_scaler_tensor``: the per-bar tensor widens
    # past ``RICH_FEATURE_SIZE`` when the #307 macro-regime block is
    # attached, but the scaler still transforms the legacy slice only.
    # Tensors narrower than ``RICH_FEATURE_SIZE`` are the legacy
    # 6-feature path the determinism regression pins.
    if x.shape[-1] < RICH_FEATURE_SIZE:
        return x

    medians = torch.tensor(scaler.medians, dtype=x.dtype, device=x.device)
    iqrs = torch.tensor(scaler.iqrs, dtype=x.dtype, device=x.device)
    out = x.clone()
    out[..., FEATURE_SIZE:RICH_FEATURE_SIZE] = (
        x[..., FEATURE_SIZE:RICH_FEATURE_SIZE] - medians
    ) / iqrs
    return out


def fit_vol_regime_quantiles(
    forward_vols: Sequence[float],
    *,
    n_classes: int = 3,
) -> tuple[float, ...]:
    """Fit per-fold quantile cutoffs for the vol-regime classifier (#195).

    Takes a list of forward-realised-vol values from the TRAIN slice
    only and returns the (``n_classes - 1``) interior quantile cutoffs
    that map a continuous vol value to a class index. For the default
    3-class plan: returns the (33%, 67%) cutoffs so a continuous vol
    ``v`` lands in class 0 (calm) when ``v < q33``, class 1 (normal)
    when ``q33 <= v < q67``, class 2 (high) when ``v >= q67``.

    Train-only fit: never call this on val / test rows. The cutoffs
    persist into the model checkpoint via
    ``ModelConfig.vol_regime_quantiles`` so inference + eval apply
    the same boundaries.

    Returns an empty tuple when ``forward_vols`` carries fewer than
    ``n_classes`` non-NaN values (no defensible split possible).
    """

    import numpy as np

    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2; got {n_classes}")
    arr = np.asarray(
        [v for v in forward_vols if v is not None and v == v],  # filter None + NaN
        dtype=np.float64,
    )
    if arr.size < n_classes:
        return ()
    # Interior quantile boundaries: for 3-class -> (1/3, 2/3); for
    # 5-class -> (0.2, 0.4, 0.6, 0.8). Excludes 0% and 100%.
    qs = [(i + 1) / n_classes for i in range(n_classes - 1)]
    cutoffs = np.quantile(arr, qs)
    return tuple(float(c) for c in cutoffs)


def fit_class_weights(
    forward_vols: Sequence[float],
    quantiles: Sequence[float],
    *,
    n_classes: int,
    smoothing: float = 1.0,
    power: float = 1.0,
) -> tuple[float, ...]:
    """Return inverse-frequency class weights fitted on a train slice (#206).

    Computes per-class counts under the supplied quantile cutoffs, then
    returns weights proportional to ``1 / (count + smoothing) ** power``
    so a class with no events still receives finite weight rather than
    blowing up the loss. The weights are normalised so they sum to
    ``n_classes`` -- a class with the dataset's average frequency gets
    weight ~1, a rare class gets > 1, a dominant class gets < 1.

    ``power=1.0`` (default) is the standard inverse-frequency formula
    and preserves the pre-Bundle-A.2 behaviour byte-identically. Larger
    powers (e.g. ``2.0``) steepen the relative weight of rare classes,
    forcing the gradient to chase the middle-vol class that single-seed
    runs collapse on under uniform inverse-frequency weighting. Smaller
    powers (e.g. ``0.5``) flatten the weighting toward uniform.

    Returns ``()`` (empty) when no quantile cutoffs are available; the
    caller then falls back to uniform weighting via the standard
    ``CrossEntropyLoss`` default.

    The motivation is the Tier 1 collapse in §6.3 of the wiki: the
    optimiser learns the training-distribution prior because the loss
    path of least resistance is "predict the majority class". Inverse-
    frequency weights flatten that prior so the gradient is forced to
    discriminate the minority classes.
    """

    if not quantiles or n_classes < 2:
        return ()
    counts = [0] * n_classes
    for v in forward_vols:
        if v is None or v != v:
            continue
        cls = vol_regime_class_for(v, quantiles)
        if 0 <= cls < n_classes:
            counts[cls] += 1
    if sum(counts) == 0:
        return ()
    raw = [1.0 / ((c + smoothing) ** power) for c in counts]
    total = sum(raw)
    return tuple((w / total) * n_classes for w in raw)


def vol_regime_class_for(value: float | None, quantiles: Sequence[float]) -> int:
    """Map a forward-vol value to a class index using fitted quantiles.

    Returns ``-1`` when ``value`` is missing (``None`` / NaN) so the
    caller can drop the row from the classification training set
    rather than silently coercing it to class 0 (calm).
    """

    if value is None or value != value:
        return -1
    for cls_idx, cutoff in enumerate(quantiles):
        if value < cutoff:
            return cls_idx
    return len(quantiles)


def collect_forward_vols(
    sequence_groups: Sequence[Sequence[FeatureVector]],
) -> list[float]:
    """Pull the forward-vol target off every supervised target row.

    A "target" row is the bar at index ``SEQUENCE_LENGTH`` in a sequence
    group (the event-day bar appended by ``_append_event_day_target``).
    Non-target bars are skipped because their ``forward_realized_vol_10d``
    is irrelevant to the y axis. Rows whose target column is null /
    NaN are dropped so the caller (per-fold quantile fit) only sees
    valid floats.
    """

    out: list[float] = []
    for sequence_group in sequence_groups:
        if len(sequence_group) < SEQUENCE_LENGTH + 1:
            continue
        for idx in range(SEQUENCE_LENGTH, len(sequence_group)):
            value = getattr(sequence_group[idx], "forward_realized_vol_10d", None)
            if value is None:
                continue
            if value != value:  # NaN
                continue
            out.append(float(value))
    return out


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
    *,
    output_mode: str = "regression",
    vol_regime_quantiles: Sequence[float] = (),
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
    class_indices: list[int] = []
    is_classification = output_mode == "classification"

    for sequence_group in sequence_groups:
        if len(sequence_group) < SEQUENCE_LENGTH + 1:
            continue
        for idx in range(SEQUENCE_LENGTH, len(sequence_group)):
            window = sequence_group[idx - SEQUENCE_LENGTH : idx]
            target = sequence_group[idx]
            if is_classification:
                cls_idx = vol_regime_class_for(
                    getattr(target, "forward_realized_vol_10d", None),
                    vol_regime_quantiles,
                )
                # Drop rows whose forward-vol target is missing instead
                # of silently coercing them to class 0 (calm). This
                # matches the regression contract: rows without a
                # defensible target never see the optimiser.
                if cls_idx < 0:
                    continue
                class_indices.append(cls_idx)
            else:
                targets.append(
                    [
                        target.market_close / fitted_scale,
                        max(target.market_volatility, 0.0),
                    ]
                )
            if use_rich:
                row_list = [item.as_rich_list() for item in window]
            else:
                row_list = [item.as_list() for item in window]
            sequences.append(row_list)

    if not sequences:
        return None, None, fitted_scale

    x = torch.tensor(sequences, dtype=torch.float32)
    if is_classification:
        y = torch.tensor(class_indices, dtype=torch.long)
    else:
        y = torch.tensor(targets, dtype=torch.float32)
    return x, y, fitted_scale


def _build_multi_task_target_tensors(
    sequence_groups: Sequence[Sequence[FeatureVector]],
    *,
    vol_regime_quantiles: Sequence[float],
) -> dict[str, torch.Tensor] | None:
    """Materialise per-axis targets aligned with ``_build_training_tensors``.

    Returns a dict of 8 1-D tensors: 4 target tensors (stance / factor /
    certainty / topic) and 4 corresponding mask tensors. Row alignment
    matches the classification rows ``_build_training_tensors`` emits —
    same filter (drop rows whose ``forward_realized_vol_10d`` is missing
    so ``vol_regime_class_for`` returns -1), same iteration order. Mask
    tensors are True iff the underlying axis label was populated on the
    target-row event; False rows do not contribute to that axis's loss.

    Returns ``None`` when no rows survive the filter (matches the
    None / None return contract of ``_build_training_tensors``).
    """

    stance_targets: list[int] = []
    stance_masks: list[bool] = []
    factor_targets: list[float] = []
    factor_masks: list[bool] = []
    certainty_targets: list[int] = []
    certainty_masks: list[bool] = []
    topic_targets: list[int] = []
    topic_masks: list[bool] = []

    for sequence_group in sequence_groups:
        if len(sequence_group) < SEQUENCE_LENGTH + 1:
            continue
        for idx in range(SEQUENCE_LENGTH, len(sequence_group)):
            target_row = sequence_group[idx]
            cls_idx = vol_regime_class_for(
                getattr(target_row, "forward_realized_vol_10d", None),
                vol_regime_quantiles,
            )
            if cls_idx < 0:
                continue
            stance_idx = int(getattr(target_row, "target_stance_idx", -1) or 0)
            stance_present = bool(getattr(target_row, "target_stance_present", False))
            stance_targets.append(max(stance_idx, 0))
            stance_masks.append(stance_present)

            factor_val = float(getattr(target_row, "target_factor", 0.0) or 0.0)
            factor_present = bool(getattr(target_row, "target_factor_present", False))
            factor_targets.append(factor_val)
            factor_masks.append(factor_present)

            certainty_idx = int(getattr(target_row, "target_certainty_idx", -1) or 0)
            certainty_present = bool(
                getattr(target_row, "target_certainty_present", False)
            )
            certainty_targets.append(max(certainty_idx, 0))
            certainty_masks.append(certainty_present)

            topic_idx = int(getattr(target_row, "target_topic_idx", -1) or 0)
            topic_present = bool(getattr(target_row, "target_topic_present", False))
            topic_targets.append(max(topic_idx, 0))
            topic_masks.append(topic_present)

    if not stance_targets:
        return None
    return {
        "stance": torch.tensor(stance_targets, dtype=torch.long),
        "stance_mask": torch.tensor(stance_masks, dtype=torch.bool),
        "factor": torch.tensor(factor_targets, dtype=torch.float32),
        "factor_mask": torch.tensor(factor_masks, dtype=torch.bool),
        "certainty": torch.tensor(certainty_targets, dtype=torch.long),
        "certainty_mask": torch.tensor(certainty_masks, dtype=torch.bool),
        "topic": torch.tensor(topic_targets, dtype=torch.long),
        "topic_mask": torch.tensor(topic_masks, dtype=torch.bool),
    }


def _build_text_embedding_tensors(
    sequence_groups: Sequence[Sequence[FeatureVector]],
    *,
    fallback_in_dim: int = 0,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    """Materialise the (pooled, missing_flag, in_dim) triple per training window.

    For each window the per-event pooled embedding lives on every
    ``FeatureVector`` in the group; the helper reads the embedding
    off the LAST prior bar (index ``SEQUENCE_LENGTH - 1``) so the
    chosen vector matches the supervised window the trainer sees.
    Missing rows materialise a zero ``in_dim``-vector + a ``1.0``
    missing flag so the model's adapter projects the same zero slot
    it would in the encoder-disabled path.

    ``fallback_in_dim`` lets the caller pin the expected encoder dim
    when every sequence in the batch is missing (e.g. the prefix of
    the corpus before any prior statement is available). Without
    that fallback the helper would return ``(None, None, 0)`` and
    the training loop would skip the text path entirely; with the
    fallback it materialises a zero-payload tensor of the requested
    width so the model's adapter still runs and the missing flag
    drives the output to zero.

    Returns ``(None, None, 0)`` when none of the sequences carry a
    pooled embedding AND no ``fallback_in_dim`` is supplied. Callers
    that hit the no-text path then skip the text-embedding kwargs
    on the model forward entirely.
    """

    in_dim = 0
    for sequence_group in sequence_groups:
        for item in sequence_group:
            pooled = getattr(item, "text_embedding_pooled", None) or []
            if pooled:
                in_dim = len(pooled)
                break
        if in_dim:
            break
    if in_dim == 0:
        if fallback_in_dim <= 0:
            return None, None, 0
        in_dim = int(fallback_in_dim)

    pooled_rows: list[list[float]] = []
    missing_rows: list[list[float]] = []
    for sequence_group in sequence_groups:
        if len(sequence_group) < SEQUENCE_LENGTH + 1:
            continue
        for idx in range(SEQUENCE_LENGTH, len(sequence_group)):
            anchor = sequence_group[idx - 1]
            pooled_payload = list(getattr(anchor, "text_embedding_pooled", []) or [])
            missing_flag = float(getattr(anchor, "text_embedding_missing", 1.0))
            if not pooled_payload or len(pooled_payload) != in_dim:
                pooled_payload = [0.0] * in_dim
                missing_flag = 1.0
            pooled_rows.append(pooled_payload)
            missing_rows.append([missing_flag])

    if not pooled_rows:
        return None, None, in_dim

    pooled_tensor = torch.tensor(pooled_rows, dtype=torch.float32)
    missing_tensor = torch.tensor(missing_rows, dtype=torch.float32)
    return pooled_tensor, missing_tensor, in_dim


def build_per_bar_text_tensor(
    sequence_groups: Sequence[Sequence[FeatureVector]],
    *,
    fallback_in_dim: int = 0,
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    """Materialise the per-bar text payload triple for issue #327 Arm A.

    For each supervised window the helper emits a ``(num_windows, T,
    in_dim)`` pooled tensor + a ``(num_windows, T)`` missing-flag
    tensor where ``T = sequence_length``. The per-bar payload is read
    off ``FeatureVector.text_per_bar`` when the loader populated it
    upstream; otherwise the helper falls back to tile-replicating the
    last prior bar's ``text_embedding_pooled`` across every bar in the
    lookback so the broadcast-static path stays bit-equivalent to the
    Arm A wiring (the parity test in
    ``tests/unit/test_text_path_arms.py`` exercises this branch).

    Returns ``(None, None, 0)`` when no sequence carries a pooled
    embedding AND no ``fallback_in_dim`` is supplied; callers then
    skip the per-bar kwarg on the model forward.
    """

    in_dim = 0
    for sequence_group in sequence_groups:
        for item in sequence_group:
            per_bar = getattr(item, "text_per_bar", None)
            if per_bar:
                first_row = next((row for row in per_bar if row), None)
                if first_row:
                    in_dim = len(first_row)
                    break
            pooled = getattr(item, "text_embedding_pooled", None) or []
            if pooled:
                in_dim = len(pooled)
                break
        if in_dim:
            break
    if in_dim == 0:
        if fallback_in_dim <= 0:
            return None, None, 0
        in_dim = int(fallback_in_dim)

    per_bar_rows: list[list[list[float]]] = []
    missing_rows: list[list[float]] = []
    for sequence_group in sequence_groups:
        if len(sequence_group) < sequence_length + 1:
            continue
        for idx in range(sequence_length, len(sequence_group)):
            window = list(sequence_group[idx - sequence_length : idx])
            per_bar_payload: list[list[float]] = []
            missing_flags: list[float] = []
            # Anchor the broadcast fallback to the last prior bar so the
            # tile-replicate matches the broadcast-static path the
            # baseline arm consumes.
            anchor = sequence_group[idx - 1]
            anchor_pooled = list(
                getattr(anchor, "text_embedding_pooled", []) or []
            )
            anchor_missing = float(
                getattr(anchor, "text_embedding_missing", 1.0)
            )
            if not anchor_pooled or len(anchor_pooled) != in_dim:
                anchor_pooled = [0.0] * in_dim
                anchor_missing = 1.0
            window_per_bar = getattr(anchor, "text_per_bar", None)
            for bar_idx, bar in enumerate(window):
                bar_pooled: list[float] | None = None
                bar_missing: float | None = None
                if window_per_bar and bar_idx < len(window_per_bar):
                    candidate = window_per_bar[bar_idx]
                    if (
                        isinstance(candidate, list)
                        and len(candidate) == in_dim
                    ):
                        bar_pooled = [float(v) for v in candidate]
                        bar_missing = 0.0
                if bar_pooled is None:
                    bar_pooled_attr = getattr(bar, "text_embedding_pooled", None)
                    if (
                        isinstance(bar_pooled_attr, list)
                        and len(bar_pooled_attr) == in_dim
                    ):
                        bar_pooled = [float(v) for v in bar_pooled_attr]
                        bar_missing = float(
                            getattr(bar, "text_embedding_missing", 1.0)
                        )
                if bar_pooled is None:
                    bar_pooled = list(anchor_pooled)
                    bar_missing = anchor_missing
                per_bar_payload.append(bar_pooled)
                missing_flags.append(float(bar_missing or 0.0))
            per_bar_rows.append(per_bar_payload)
            missing_rows.append(missing_flags)

    if not per_bar_rows:
        return None, None, in_dim

    per_bar_tensor = torch.tensor(per_bar_rows, dtype=torch.float32)
    missing_tensor = torch.tensor(missing_rows, dtype=torch.float32)
    return per_bar_tensor, missing_tensor, in_dim


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
