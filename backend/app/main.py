import asyncio
import fcntl
import io
import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import DATA_DIR
from app.db import (
    AnalysisRun,
    delete_run,
    get_engine,
    get_run,
    get_session,
    list_runs,
    persist_analysis_run,
    session_scope,
)
from app.logging import configure_logging, get_logger
from app.middleware.errors import RunIdMiddleware, register_error_handlers
from app.schemas import (
    AnalogCard,
    AnalogsRequest,
    AnalogsResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    ArtifactFile,
    ClassificationBreakdownClass,
    ClassificationBreakdownResponse,
    ClassificationBreakdownSource,
    DocumentParseResponse,
    DocumentParseUrlRequest,
    EvaluationCoverageResponse,
    FomcCalendarResponse,
    HistoryDetail,
    HistoryEntry,
    HistoryEventStudyResponse,
    HistoryList,
    HistoryRealizedBatchResponse,
    HistoryRealizedResponse,
    MarketReactionPanel,
    NextFomcForecastResponse,
    RatesReactionCard,
    ResearchArtifactsResponse,
    BacktestRequest,
    BacktestResponse,
    ResearchRegistryResponse,
    SettingsCheckpoint,
    SettingsCheckpointsResponse,
    SymbolDescriptor,
    SymbolListResponse,
    TrajectoryMarker,
    TrajectoryProjection,
    TrajectoryRequest,
    TrajectoryResponse,
    VolRegimeReactionCard,
)
from app.evaluation.xai import attribute_text, split_sentences, to_response as xai_to_response
from app.services.document_parser import (
    parse_docx_stream,
    parse_paste,
    parse_pdf_stream,
    parse_url,
)
from app.services.decision_forecast import load_next_fomc_artifacts
from app.services.classification_breakdown_loader import load_latest as load_classification_breakdown
from app.services.fomc_calendar import get_calendar
from app.services.research_artifacts import (
    SECTIONS as RESEARCH_SECTIONS,
    list_section_files,
    load_cross_bank_transfer,
    load_encoder_bakeoff,
    load_research_registry,
)
from app.services.forecaster import (
    bootstrap_checkpoint,
    bucket_realized_regime,
    build_feature_vectors,
    build_market_reaction_panel,
    build_panel_attributions,
    build_regime_classification_card,
    checkpoint_exists,
    forecast_quantitative_series,
    parse_horizon_steps,
)
from app.services.market_data import (
    fetch_event_study_window,
    fetch_forward_trading_dates,
    fetch_market_history,
    fetch_market_snapshot,
    fetch_realized_forward,
)
from app.services.policy_action_extractor import extract_policy_action
from app.services.text_encoder import analyze_text
from app.services.forecaster_text_embedding import encode_text_pooled

logger = logging.getLogger(__name__)

# 252 trading days ≈ one year of context. Used as the cold-start
# bootstrap history when /analyze fires against a host that has no
# checkpoint on disk yet (first run after a fresh clone).
COLD_START_HISTORY_LENGTH = 252

# #379: the frontend fans out /analyze, /analyze/market and /analyze/analogs
# in parallel via Promise.allSettled. On a fresh boot all three race into
# ``_bootstrap_cold_start`` — two concurrent writers on the same checkpoint
# file can produce a half-written artefact, and the second loader then
# raises an opaque deserialisation error that escapes the route's narrow
# ``RuntimeError`` catch. ``ServerErrorMiddleware`` then returns a bare
# 500 outside the CORS layer, surfacing in the browser as ERR_FAILED /
# "no Access-Control-Allow-Origin". Serialising the cold-start with an
# asyncio lock + a checkpoint re-check inside the critical section keeps
# only one bootstrap in flight per process.
#
# #383: the asyncio lock only serialises within one process. Prod runs
# uvicorn with multiple workers, so we also wrap the bootstrap in an
# OS-level ``fcntl.flock`` (see ``_bootstrap_cold_start``). The asyncio
# lock stays as the cheap in-process guard; the file lock is the
# cross-process safety net.
_cold_start_lock = asyncio.Lock()


def _bootstrap_lock_path() -> Path:
    """Sentinel-file path for the cross-process bootstrap lock.

    Resolved lazily because ``BEST_MODEL_PATH`` is imported from a
    config module that some test paths monkeypatch.
    """

    from app.services.forecaster import BEST_MODEL_PATH

    return Path(str(BEST_MODEL_PATH) + ".bootstrap.lock")


@contextmanager
def _bootstrap_file_lock(lock_path: Path):
    """Acquire an exclusive ``fcntl.flock`` on a sentinel file.

    Contract:
      * Only one process at a time holds the lock; runners-up block
        until the leader releases it, then observe the checkpoint and
        skip the bootstrap on the re-check.
      * The lock file is a dedicated sentinel — not the checkpoint
        itself — so a half-written checkpoint can never masquerade as
        "already bootstrapped".
      * ``try/finally`` releases the lock on every path, including
        exceptions raised inside the bootstrap.
      * POSIX-only (``fcntl`` exists on macOS and Linux). The s6
        deploy is POSIX, so this matches the prod runtime.
    """

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


async def _ensure_cold_start(payload: "AnalyzeRequest") -> None:
    """Run ``_bootstrap_cold_start`` at most once across concurrent callers.

    The lock is acquired only when the checkpoint is missing; the
    fast-path (warm process) takes zero locks. Inside the critical
    section we re-check ``checkpoint_exists`` so the runner-up does not
    re-train against the artefact the leader just wrote.
    """

    if checkpoint_exists():
        return
    async with _cold_start_lock:
        if checkpoint_exists():
            return
        await run_in_threadpool(_bootstrap_cold_start, payload)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    configure_logging()
    get_engine()
    # Pre-load the HF classifier so the first /analyze request doesn't pay
    # the cold-start cost. The lazy module-level cache keeps this idempotent.
    try:
        from app.services.text_encoder import warmup_classifier

        warmup_classifier()
    except Exception:  # pragma: no cover — never let model warmup block startup
        get_logger("fed_pulse").warning("classifier_warmup_failed", exc_info=True)
    # Pre-load the multi-axis classifier when a checkpoint is present.
    # The service returns None on missing / malformed checkpoints, so
    # this is a no-op until the trainer ships a real model.
    try:
        from app.services.multi_axis_classifier import (
            checkpoint_exists as multi_axis_checkpoint_exists,
            get_classifier as get_multi_axis_classifier,
        )

        if multi_axis_checkpoint_exists():
            get_multi_axis_classifier()
    except Exception:  # pragma: no cover — never let warmup block startup
        get_logger("fed_pulse").warning(
            "multi_axis_classifier_warmup_failed", exc_info=True
        )
    get_logger("fed_pulse").info("startup", service="fomc-api")
    try:
        yield
    finally:
        get_logger("fed_pulse").info("shutdown", service="fomc-api")


app = FastAPI(title="FOMC Sentiment API", version="0.1.0", lifespan=_lifespan)

app.add_middleware(RunIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
register_error_handlers(app)


@app.get("/health")
def health_check():
    """Liveness probe + serving-contract status (#341, extended in #393).

    The probe always returns ``status: "ok"`` (uvicorn is up and
    serving). Each serving artefact (forecaster + multi-axis classifier
    + trajectory bundle) surfaces a structured contract block: ``ok``
    when the sidecar matches the serving signature, ``sidecar_absent``
    on a legacy artefact, and a structured failure code otherwise.
    Forecaster counters track structured-error increments so an
    operator can spot a stuck contract without parsing logs.
    """

    try:
        from app.services.forecaster import (
            get_contract_counters,
            get_serving_contract_status,
        )

        contract = get_serving_contract_status()
        counters = get_contract_counters()
    except Exception:  # pragma: no cover -- defensive
        contract = {"status": "unknown"}
        counters = {}
    try:
        from app.services.multi_axis_classifier import (
            get_serving_contract_status as _multi_axis_contract,
        )

        multi_axis_contract = _multi_axis_contract()
    except Exception:  # pragma: no cover -- defensive
        multi_axis_contract = {"status": "unknown"}
    try:
        from app.services.trajectory import (
            get_serving_contract_status as _trajectory_contract,
        )

        trajectory_contract = _trajectory_contract()
    except Exception:  # pragma: no cover -- defensive
        trajectory_contract = {"status": "unknown"}
    return {
        "status": "ok",
        "inference_contract": contract,
        "inference_contract_counters": counters,
        "multi_axis_classifier_contract": multi_axis_contract,
        "trajectory_contract": trajectory_contract,
    }


_SYMBOLS_FALLBACK: list[dict[str, str]] = [
    {"symbol": "^GSPC", "name": "S&P 500", "category": "Equity index", "default_horizon": "10d"},
]


def _checkpoint_role(name: str) -> str:
    """Cheap filename-based role inference for the settings inventory."""

    lower = name.lower()
    if "multi_axis" in lower:
        return "multi_axis"
    if "lora" in lower:
        return "lora_adapter"
    if "calibration" in lower:
        return "calibration"
    if "forecaster" in lower:
        return "forecaster"
    return "other"


@app.get("/settings/checkpoints", response_model=SettingsCheckpointsResponse)
def list_settings_checkpoints() -> SettingsCheckpointsResponse:
    """Read-only inventory of model files under ``backend/models/``.

    Surfaces filename, size, mtime, inferred role, and an ``is_active``
    flag pointing at the file each live service is currently loaded
    from. Diagnostic fields (``output_mode``, ``encoder_alias``,
    ``conformal_sidecar_present``) only populate on the active
    forecaster and multi-axis entries — everything else stays None so
    the response stays serialisable on a fresh checkout.
    """

    from app.models.config import MODELS_DIR
    from app.services.forecaster import BEST_MODEL_PATH

    items: list[SettingsCheckpoint] = []
    if not MODELS_DIR.exists():
        return SettingsCheckpointsResponse(models_dir=str(MODELS_DIR), checkpoints=items)

    active_forecaster = BEST_MODEL_PATH.resolve()
    active_forecaster_meta: dict[str, Any] = {}
    try:
        from app.services.forecaster import _get_model, _model_artifact_metadata  # type: ignore

        model = _get_model()
        active_forecaster_meta = {
            "output_mode": str(getattr(model, "output_mode", "regression") or "regression"),
        }
        encoder = (_model_artifact_metadata or {}).get("encoder_key")
        if isinstance(encoder, str):
            active_forecaster_meta["encoder_alias"] = encoder
    except Exception:  # pragma: no cover — diagnostics never block inventory
        logger.warning("settings_checkpoints_forecaster_probe_failed", exc_info=True)

    active_multi_axis: Path | None = None
    active_multi_axis_alias: str | None = None
    try:
        from app.services.multi_axis_classifier import (
            _resolve_checkpoint_path as multi_axis_path,
            get_loaded_encoder_alias,
        )

        active_multi_axis = multi_axis_path().resolve()
        active_multi_axis_alias = get_loaded_encoder_alias()
    except Exception:  # pragma: no cover
        logger.warning("settings_checkpoints_multi_axis_probe_failed", exc_info=True)

    # #342: snapshot the live serving forward signature once per request
    # so each row can mark each declared kwarg as supplied / not supplied
    # without re-introspecting the model class on every iteration. The
    # source of truth is the model class the loader binds — falls back to
    # the static SERVING_FORWARD_KWARGS constant when the import path is
    # not available (defensive; the import should always succeed).
    try:
        from app.models.serving_model import ForecasterServingModel
        from app.training.inference_contract import (
            SERVING_FORWARD_KWARGS,
            collect_serving_forward_kwargs,
            read_sidecar,
        )
    except Exception:  # pragma: no cover -- defensive
        # Hard import failure (genuinely broken env). Nothing to
        # display; render every checkpoint row without a contract
        # surface rather than mislabel the world.
        logger.warning("settings_checkpoints_serving_kwargs_import_failed", exc_info=True)
        serving_kwargs: frozenset[str] = frozenset()
        read_sidecar = None  # type: ignore[assignment]
    else:
        try:
            serving_kwargs = (
                collect_serving_forward_kwargs(ForecasterServingModel)
                or SERVING_FORWARD_KWARGS
            )
        except Exception:  # pragma: no cover -- defensive
            # Live-signature introspection failed but the imports landed.
            # Fall back to the static constant per ADR 0025 so the
            # settings page still renders meaningful supplied/required
            # badges rather than painting every kwarg red.
            logger.warning(
                "settings_checkpoints_serving_kwargs_probe_failed", exc_info=True
            )
            serving_kwargs = SERVING_FORWARD_KWARGS

    for entry in sorted(MODELS_DIR.glob("*.pt"), key=lambda p: p.name):
        try:
            stat = entry.stat()
        except OSError:
            continue
        resolved = entry.resolve()
        role = _checkpoint_role(entry.name)
        is_active_forecaster = role == "forecaster" and resolved == active_forecaster
        is_active_multi_axis = role == "multi_axis" and active_multi_axis is not None and resolved == active_multi_axis
        sidecar_present: bool | None = None
        if role == "forecaster":
            sidecar_present = entry.with_suffix(".conformal.json").exists()

        required_kwargs: list[str] = []
        supplied: dict[str, bool] = {}
        contract_status: str | None = None
        if role == "forecaster" and read_sidecar is not None:
            try:
                contract = read_sidecar(entry)
            except Exception:  # pragma: no cover -- defensive
                contract = None
            if contract is None:
                contract_status = "sidecar_absent"
            else:
                contract_status = "present"
                required_kwargs = [str(k) for k in contract.required_kwargs]
                supplied = {
                    name: (name in serving_kwargs) for name in required_kwargs
                }

        items.append(
            SettingsCheckpoint(
                filename=entry.name,
                relative_path=str(entry.relative_to(MODELS_DIR)),
                role=role,
                size_bytes=int(stat.st_size),
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                is_active=is_active_forecaster or is_active_multi_axis,
                output_mode=active_forecaster_meta.get("output_mode") if is_active_forecaster else None,
                encoder_alias=(
                    active_forecaster_meta.get("encoder_alias")
                    if is_active_forecaster
                    else active_multi_axis_alias if is_active_multi_axis else None
                ),
                conformal_sidecar_present=sidecar_present,
                required_kwargs=required_kwargs,
                supplied_at_inference=supplied,
                inference_contract_status=contract_status,
            )
        )

    return SettingsCheckpointsResponse(models_dir=str(MODELS_DIR), checkpoints=items)


@app.get("/symbols", response_model=SymbolListResponse)
def list_symbols() -> SymbolListResponse:
    """Symbol universe the workspace asset picker reads.

    Loads ``backend/app/data/symbols.json`` from the package directory
    (resolved relative to this module so the path works regardless of
    the Compose volume layout). Falls back to a single S&P 500 entry so
    the endpoint never 500s on a fresh checkout where the data file is
    missing.
    """

    package_path = Path(__file__).parent / "data" / "symbols.json"
    data_dir_path = DATA_DIR / "symbols.json"
    candidates = [package_path]
    if data_dir_path != package_path:
        candidates.append(data_dir_path)
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            entries = payload.get("symbols") if isinstance(payload, dict) else None
            if isinstance(entries, list):
                items = [SymbolDescriptor(**entry) for entry in entries]
                return SymbolListResponse(symbols=items)
        except Exception:
            logger.warning("symbols_load_failed path=%s", path, exc_info=True)
            continue
    return SymbolListResponse(
        symbols=[SymbolDescriptor(**entry) for entry in _SYMBOLS_FALLBACK]
    )


@app.get("/documents")
def list_documents():
    sources = [
        ("fomc_statements.json", "Statement"),
        ("fomc_minutes.json", "Minutes"),
    ]
    documents: list[dict[str, str]] = []

    for filename, default_type in sources:
        path = DATA_DIR / filename
        if not path.exists():
            continue

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read {filename}: {exc}") from exc

        if not isinstance(payload, list):
            continue

        for item in payload:
            if not isinstance(item, dict):
                continue
            documents.append(
                {
                    "title": str(item.get("title", "")),
                    "date": str(item.get("date", "")),
                    "document_type": str(item.get("document_type", default_type)),
                }
            )

    documents.sort(key=lambda doc: doc.get("date", ""), reverse=True)
    return {"count": len(documents), "documents": documents}


@app.get("/documents/by-date")
def get_document_by_date(date: str, kind: str = "auto"):
    """Look up an FOMC statement or minutes by event date so the calendar
    page can prefill the analyze textarea on click. ``kind`` is one of
    ``auto`` (try statement first, then minutes), ``statement``, or
    ``minutes``.
    """

    allowed_kinds = {"auto", "statement", "minutes"}
    if kind not in allowed_kinds:
        raise HTTPException(
            status_code=422,
            detail=f"kind must be one of {sorted(allowed_kinds)}; got {kind!r}",
        )

    sources_in_order: list[tuple[str, str]]
    if kind == "statement":
        sources_in_order = [("fomc_statements.json", "Statement")]
    elif kind == "minutes":
        sources_in_order = [("fomc_minutes.json", "Minutes")]
    else:
        sources_in_order = [
            ("fomc_statements.json", "Statement"),
            ("fomc_minutes.json", "Minutes"),
        ]

    for filename, document_type in sources_in_order:
        path = DATA_DIR / filename
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to read {filename}: {exc}"
            ) from exc
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            if str(item.get("date", "")) != date:
                continue
            text = str(item.get("text") or item.get("content") or "")
            if not text:
                continue
            return {
                "date": date,
                "kind": document_type.lower(),
                "title": str(item.get("title", "")),
                "text": text,
                "source_file": filename,
            }
    raise HTTPException(
        status_code=404,
        detail=f"No FOMC document found for date {date!r} (kind={kind}).",
    )


def _build_analyze_response(
    payload: AnalyzeRequest,
    *,
    mode: str,
    history_length: int,
) -> dict[str, Any]:
    sentiment = analyze_text(payload.text)
    market = fetch_market_snapshot(target_date=payload.date, symbol=payload.symbol)
    market_history = fetch_market_history(
        target_date=payload.date,
        symbol=payload.symbol,
        history_length=history_length,
    )
    horizon_steps = parse_horizon_steps(payload.horizon)
    forecast_dates = fetch_forward_trading_dates(
        target_date=payload.date,
        symbol=payload.symbol,
        steps=horizon_steps,
    )

    # Encode the pasted statement once so the forecaster's text channel is
    # populated at inference. Without it the model degenerates to a market-
    # only predictor. None on load failure -> loader emits the missing-flag
    # and the text slot zero-pads.
    pooled_text_embedding = encode_text_pooled(payload.text)
    history_vectors = build_feature_vectors(
        market_history,
        sentiment_score=float(sentiment["score"]),
        document_date=payload.date,
        text_embedding=pooled_text_embedding,
    )
    forecast = forecast_quantitative_series(
        vectors=history_vectors,
        forecast_mode=mode,
        horizon=payload.horizon,
        forecast_dates=forecast_dates,
    )

    if payload.include_realized:
        realized = fetch_realized_forward(
            target_date=payload.date,
            symbol=payload.symbol,
            steps=horizon_steps,
        )
        if realized:
            forecast["series"]["realized_timestamps"] = [str(point["date"]) for point in realized]
            forecast["series"]["realized_close"] = [float(point["close"]) for point in realized]
            forecast["series"]["realized_volatility"] = [float(point["volatility_5d"]) for point in realized]

    regime_payload = _safe_regime_classification(history_vectors)
    # #341: structured-status payloads carry a ``status`` key and never
    # the full card shape. Split into the legacy card slot (None when
    # degraded) + the sibling status surface so the response stays
    # serialisable against the AnalyzeResponse schema.
    regime_card: dict[str, Any] | None = None
    regime_status: dict[str, Any] | None = None
    if isinstance(regime_payload, dict) and regime_payload.get("status") and (
        "predicted_set" not in regime_payload
    ):
        regime_status = regime_payload
    else:
        regime_card = regime_payload
    response: dict[str, Any] = {
        "sentiment": sentiment,
        "prediction": forecast["prediction"],
        "market": market,
        "model": forecast["model"],
        "series": forecast["series"],
        "multi_axis": _build_multi_axis_block(payload.text, sentiment),
        "regime_classification": regime_card,
        "regime_regression": _build_regime_regression_block(regime_card),
        "regime_classification_status": regime_status,
        "rates_reaction": _safe_rates_reaction(history_vectors),
        "policy_action": _build_policy_action_card(payload),
    }
    if getattr(payload, "include_xai", False):
        attributions = attribute_text(payload.text)
        xai_block = xai_to_response(attributions)
        # #297: layer per-panel integrated-gradients attribution on top
        # of the existing sentence-level surface. Any panel that cannot
        # be explained surfaces an ``unavailable`` payload; the helper
        # itself never raises (every internal call is wrapped in a
        # structured-degrade try/except). Guard the dispatch defensively
        # so an IG runtime failure cannot break /analyze.
        try:
            panel_attributions = build_panel_attributions(
                history_vectors, as_of_date=payload.date
            )
        except Exception:  # noqa: BLE001 -- defensive: never break /analyze
            logger.warning("xai_panel_attribution_failed", exc_info=True)
            panel_attributions = []
        if panel_attributions:
            xai_block["panels"] = panel_attributions
        response["xai"] = xai_block
    return response


def _build_regime_regression_block(
    regime_card: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Derive the sibling ``regime_regression`` block from the classification card (#304).

    The dual-head retrofit keeps the classification card as the
    product-facing surface and exposes the regression head's point
    estimate + 90% conformal interval as a standalone sibling block so
    a frontend can render it behind a "show details" toggle without
    re-deriving the read out of the classification card. The block is
    emitted only when the classification card carries a non-null
    ``log_rv_point`` (i.e. ``head_mode`` in ``regression`` / ``dual``);
    otherwise the field stays ``None`` on the response and the legacy
    classification-only payload shape is byte-identical.
    """

    if not isinstance(regime_card, dict):
        return None
    log_rv_point = regime_card.get("log_rv_point")
    if log_rv_point is None:
        return None
    block: dict[str, Any] = {
        "log_rv_point": float(log_rv_point),
        "log_rv_lower": regime_card.get("log_rv_lower"),
        "log_rv_upper": regime_card.get("log_rv_upper"),
    }
    coverage = regime_card.get("coverage")
    # Coverage on the classification card maps to the calibrated APS
    # set's nominal coverage. The regression interval rides off the
    # same conformal manifest (residual_quantile_volatility) so it
    # inherits the same nominal coverage for now; when the regression
    # head gets a dedicated quantile sidecar this read switches over.
    if isinstance(coverage, int | float) and coverage > 0:
        block["coverage"] = float(coverage)
    return block


def _safe_rates_reaction(
    history_vectors: list[Any],
) -> list[dict[str, Any]] | None:
    """Build the rates-reaction list off the active checkpoint (#292).

    Reuses :func:`build_market_reaction_panel` to populate the per-head
    rates cards for #293's panel. Returns ``None`` when the checkpoint
    mounts no rates heads or the panel builder degrades to a structured
    status payload (no card data to surface). An empty list rides when
    the heads exist but the per-event forward produced no rows -- the
    schema layer treats that as "active, no read" rather than "absent".

    Wrapped end-to-end in try/except so an inference path crash never
    breaks /analyze.
    """

    from app.services.forecaster import build_market_reaction_panel

    try:
        panel = build_market_reaction_panel(history_vectors)
    except Exception:  # pragma: no cover -- defensive: never break /analyze
        logger.warning("rates_reaction_failed", exc_info=True)
        return None
    if not isinstance(panel, dict):
        return None
    rates_block = panel.get("rates")
    if not isinstance(rates_block, list):
        return None
    if not rates_block:
        # Heads mounted but produced no rows; surface an empty list so
        # the frontend can render an "active, no read" badge instead of
        # falling back to the "no rates heads" empty state.
        return []
    return rates_block


def _build_policy_action_card(
    payload: AnalyzeRequest,
) -> dict[str, Any] | None:
    """Extract the mechanical policy decision off the request text (#446).

    Pure regex / keyword pass over ``payload.text`` via
    :func:`extract_policy_action`. Short-circuits to ``None`` when the
    request body carries no text (``AnalyzeRequest.text`` is required
    by the schema but the guard stays so a hand-crafted payload with
    whitespace-only text still serialises cleanly). Wrapped end-to-end
    in try/except so a regex misfire or an unexpected dataclass shape
    can never break /analyze.

    ``prior_target_range_mid_bp`` is left ``None`` here — the request
    schema carries no prior-statement context and we deliberately do
    not reach into the persisted history layer for it. The card still
    surfaces a populated ``change_direction`` whenever the policy verb
    is named in the prose (``decided to raise`` / ``decided to lower``
    / ``decided to maintain``); only the prior-midpoint fallback is
    deferred to a follow-up.
    """

    text = getattr(payload, "text", None)
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        action = extract_policy_action(text)
    except Exception:  # pragma: no cover -- defensive: never break /analyze
        logger.warning("policy_action_extraction_failed", exc_info=True)
        return None
    return {
        "target_range_low_bp": action.target_range_low_bp,
        "target_range_high_bp": action.target_range_high_bp,
        "change_direction": action.change_direction,
        "change_magnitude_bp": action.change_magnitude_bp,
        "balance_sheet_state": action.balance_sheet_state,
    }


def _safe_regime_classification(history_vectors: list[Any]) -> dict[str, Any] | None:
    """Wrap ``build_regime_classification_card`` so a failure never breaks /analyze.

    The card is opt-in by checkpoint flavour and manifest presence; any
    exception inside the inference + calibrated-set path degrades to a
    surfaced structured payload rather than 500ing the whole response.

    #341 structured surface. Three branches:

    1. **Not classification mode.** The active checkpoint emits no
       regime card by contract -- legitimate ``None`` with
       ``status="not_classification_mode"`` so the operator can
       distinguish "model deliberately mute" from "model crashed
       silently".
    2. **Inference kwarg missing.** The forward path raised
       :class:`TypeError` (e.g. the call site passed a kwarg the
       checkpoint did not declare in its inference contract sidecar,
       or omitted one the model requires). Surfaces
       ``status="inference_kwarg_missing"`` + the missing kwarg name
       parsed from the exception, increments the module-level counter,
       and logs at WARNING.
    3. **Unexpected exception.** Anything else: structured payload
       with the exception class name + WARNING log + counter
       increment. The /analyze response stays serialisable so the
       frontend can render an "evidence unavailable" badge.
    """

    from app.services import forecaster as _forecaster_service

    try:
        result = build_regime_classification_card(history_vectors)
    except TypeError as exc:
        _forecaster_service._contract_counters[
            "regime_classification_inference_kwarg_missing"
        ] += 1
        missing = _extract_missing_kwarg_from_typeerror(exc)
        logger.warning(
            "regime_classification_inference_kwarg_missing kwarg=%s detail=%s",
            missing,
            str(exc),
        )
        return {
            "status": "inference_kwarg_missing",
            "missing_kwarg": missing,
        }
    except Exception as exc:  # pragma: no cover — defensive, see #216 follow-up
        _forecaster_service._contract_counters[
            "regime_classification_unexpected_exception"
        ] += 1
        logger.warning(
            "regime_classification_card_failed exception_class=%s detail=%s",
            type(exc).__name__,
            str(exc),
            exc_info=True,
        )
        return {
            "status": "unexpected_exception",
            "exception_class": type(exc).__name__,
        }
    if result is None:
        return {"status": "not_classification_mode"}
    return result


def _extract_missing_kwarg_from_typeerror(exc: TypeError) -> str | None:
    """Parse a python ``TypeError`` for the offending kwarg name.

    Python emits messages like ``forward_multi_task() missing 1
    required keyword-only argument: 'text_embedding'`` or
    ``forward_multi_task() got an unexpected keyword argument
    'foo'``. We pull the quoted name out; on no-match we return
    ``None`` rather than guess.
    """

    import re

    message = str(exc)
    match = re.search(r"keyword[- ]?(?:only )?argument[s]?:?\s*['\"]([^'\"]+)['\"]", message)
    if match:
        return match.group(1)
    match = re.search(r"unexpected keyword argument\s*['\"]([^'\"]+)['\"]", message)
    if match:
        return match.group(1)
    return None


def _build_multi_axis_block(
    text: str, sentiment: dict[str, Any]
) -> dict[str, Any] | None:
    """Build the multi-axis card block from the available text signals (#78).

    Two paths:

    1. **Trained multi-axis classifier present.** Run the
       ``TextMultiAxisClassifier`` checkpoint via
       ``app.services.multi_axis_classifier.score_text``; populate the
       three cards (stance / factor / certainty) from the per-axis
       predictions. (The topic axis was retired in ADR 0044 — no
       upstream corpus shipped topic labels.)

    2. **No checkpoint.** Fall back to populating the stance card from
       the existing sentiment classifier output and leave the other
       two axes at ``None``. The frontend renders ``None`` cards as
       absent so the user sees honest absence rather than a
       placeholder value.
    """

    try:
        from app.services.multi_axis_classifier import score_text as multi_axis_score
    except Exception:  # pragma: no cover — import-time failures fall through
        multi_axis_score = None  # type: ignore[assignment]

    if multi_axis_score is not None:
        try:
            classifier_block = multi_axis_score(text)
        except Exception:  # pragma: no cover — never let the classifier crash /analyze
            logger.warning("multi_axis_classifier_failed", exc_info=True)
            classifier_block = None
        if classifier_block is not None:
            return classifier_block

    label_raw = str(sentiment.get("label", "")).strip().lower()
    canonical_labels = ("hawkish", "dovish", "neutral")
    label = label_raw if label_raw in canonical_labels else "neutral"

    distribution: dict[str, float] = dict.fromkeys(canonical_labels, 0.0)
    for entry in sentiment.get("raw", []) or []:
        raw_label = str(entry.get("label", "")).strip().lower()
        if raw_label in distribution:
            distribution[raw_label] = float(entry.get("score", 0.0) or 0.0)
    confidence = float(sentiment.get("score", distribution.get(label, 0.0)) or 0.0)
    confidence = max(0.0, min(1.0, confidence))

    return {
        "stance": {
            "label": label,
            "confidence": confidence,
            "distribution": distribution,
        },
        "factor": None,
        "certainty": None,
    }


def _apply_sentence_mask(text: str, mask: list[int]) -> str:
    """Drop the masked sentence indices and rejoin the remainder.

    Indices reference the same tokenization that produces ``xai.sentences``.
    Out-of-range / duplicate indices are silently ignored; striking every
    sentence falls back to the original text so the classifier still has
    something to score.
    """

    if not mask:
        return text
    sentences = split_sentences(text)
    if not sentences:
        return text
    masked: set[int] = set()
    for raw in mask:
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            # Defensive: schema typing already guarantees ints, but a
            # test harness or future caller passing non-numeric values
            # should be ignored per the docstring rather than 500'ing.
            continue
        if 0 <= idx < len(sentences):
            masked.add(idx)
    if not masked:
        return text
    kept = [sent for idx, sent in enumerate(sentences) if idx not in masked]
    return " ".join(kept).strip() or text


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest):
    """Async handler — heavy sync work (transformers, yfinance, torch) runs in
    the thread pool so the event loop stays responsive under load."""

    try:
        # Cold-start bootstrap: a fresh clone has no checkpoint on disk yet,
        # so seed one against a 252-day window before the first inference.
        # Serialised across the /analyze, /analyze/market and /analyze/analogs
        # fan-out via ``_ensure_cold_start`` -- #379.
        await _ensure_cold_start(payload)

        masked_text = _apply_sentence_mask(payload.text, payload.mask_sentence_indices)
        run_payload = (
            payload.model_copy(update={"text": masked_text})
            if masked_text != payload.text
            else payload
        )

        response = await run_in_threadpool(
            _build_analyze_response, run_payload, mode="fast", history_length=30
        )
        # Counterfactual runs (any non-empty mask) are synthetic — the
        # workspace fires one per sentence-strike and the user does not
        # expect each click to land in the persistent history. Skip
        # persistence so the history list only carries baseline runs.
        if not payload.mask_sentence_indices:
            await run_in_threadpool(_record_history, run_payload, response)
        return response
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:  # pragma: no cover
        # #342: contract-revalidation failures from _bootstrap_cold_start
        # land here. Log the raw message but ship a structured detail
        # carrying only the exception class -- the #341 review rule
        # keeps raw str(exc) out of client-facing payloads.
        logger.warning(
            "analyze_runtime_error exception_class=%s detail=%s",
            type(exc).__name__,
            str(exc),
        )
        raise HTTPException(
            status_code=500,
            detail="Analyze pipeline failed: serving runtime error",
        ) from None
    except Exception as exc:  # pragma: no cover
        logger.exception(
            "analyze_unexpected_exception exception_class=%s", type(exc).__name__
        )
        raise HTTPException(
            status_code=500, detail="Analyze pipeline failed: unexpected error"
        ) from None


def _bootstrap_cold_start(payload: AnalyzeRequest) -> None:
    """Seed an initial checkpoint when the model file is missing.

    Synchronous — invoked via ``run_in_threadpool`` from the async
    handler so the long-running fit does not block the event loop.

    #383: the body runs under an exclusive ``fcntl.flock`` on a
    sentinel file next to ``BEST_MODEL_PATH``. Composes with the
    process-local ``_cold_start_lock``: the asyncio lock is the cheap
    in-process guard; the file lock is the cross-process safety net so
    uvicorn can run with ``--workers > 1`` without two workers racing
    on the same checkpoint write. We re-check ``checkpoint_exists``
    inside the file lock so the runner-up (which blocked on
    ``LOCK_EX`` until the leader finished) observes the freshly
    written checkpoint and returns without re-training.

    #342: once the bootstrap writes a fresh checkpoint, we drop the
    in-process singleton + re-invoke the same loader the /analyze path
    uses (``_get_model`` -> ``_validate_serving_contract``). That way a
    bootstrap whose freshly written sidecar declares kwargs the serving
    forward does not accept raises ``RuntimeError`` here rather than
    silently binding via ``_set_singleton_after_train``. The cold-start
    boot is then loud-fail: the /analyze caller surfaces a 500 with the
    structured incompatibility reason and ``/health`` picks up the
    contract status. Bypasses the reset for legacy artefacts (no
    sidecar) because the validation degrades to ``sidecar_absent`` and
    binds normally.
    """

    with _bootstrap_file_lock(_bootstrap_lock_path()):
        # Runner-up path: the leader wrote the checkpoint while we
        # waited on ``LOCK_EX``. Skip the retrain entirely.
        if checkpoint_exists():
            return

        warmup_sentiment = analyze_text(payload.text)
        warmup_history = fetch_market_history(
            target_date=payload.date,
            symbol=payload.symbol,
            history_length=COLD_START_HISTORY_LENGTH,
        )
        warmup_vectors = build_feature_vectors(
            warmup_history,
            sentiment_score=float(warmup_sentiment["score"]),
            document_date=payload.date,
        )
        bootstrap_checkpoint(
            vectors=warmup_vectors,
            epochs=60,
            batch_size=64,
            learning_rate=4e-4,
            early_stopping_patience=8,
        )
        # #342: drop the post-train singleton + force a cold load through
        # the canonical loader so the contract validation actually runs on
        # the freshly written sidecar. ``_set_singleton_after_train`` writes
        # ``_model`` directly without crossing ``_validate_serving_contract``
        # — without this reset the cold-start path would silently bind a
        # checkpoint whose sidecar declares an unknown kwarg.
        try:
            from app.services.forecaster import (
                _get_model as _forecaster_get_model,
                reset_singleton_for_revalidation,
            )

            reset_singleton_for_revalidation()
            _forecaster_get_model()
        except RuntimeError:
            # Re-raise so the /analyze caller surfaces the contract failure
            # rather than swallowing it. ``/health`` already exposes the
            # structured reason via ``get_serving_contract_status``.
            raise
        except Exception:  # pragma: no cover -- defensive
            logger.warning("cold_start_contract_revalidation_failed", exc_info=True)
            raise


def _record_history(request: AnalyzeRequest, response: dict[str, Any]) -> None:
    # Persistence must not break /analyze; log so silent failures (disk full,
    # missing table, etc.) still show up in uvicorn output.
    request_payload = request.model_dump()
    # ``forecast_mode`` was retired from AnalyzeRequest in #265 but the
    # analysis_runs.forecast_mode DB column stayed for the legacy
    # history-list rendering. New rows would otherwise land with empty
    # strings — stamp the only runtime mode that still exists so the
    # /history listing keeps a non-empty value per row.
    request_payload.setdefault("forecast_mode", "fast")
    try:
        with session_scope() as session:
            persist_analysis_run(
                session,
                payload=response,
                request=request_payload,
                response=response,
            )
    except Exception:
        logger.warning(
            "history persistence failed for symbol=%s date=%s",
            request.symbol,
            request.date,
            exc_info=True,
        )


@app.get("/history", response_model=HistoryList)
def get_history(
    symbol: str | None = Query(default=None),
    horizon: str | None = Query(default=None),
    stance: str | None = Query(default=None),
    document_date: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0, le=2_147_483_647),
    session: Session = Depends(get_session),
) -> HistoryList:
    rows, total = list_runs(
        session,
        limit=limit,
        offset=offset,
        symbol=symbol,
        horizon=horizon,
        stance=stance,
        document_date=document_date,
    )
    items = [HistoryEntry(**row.to_summary()) for row in rows]
    return HistoryList(items=items, total=total, limit=limit, offset=offset)


@app.get("/history/{run_id}", response_model=HistoryDetail)
def get_history_run(run_id: str, session: Session = Depends(get_session)) -> HistoryDetail:
    row = get_run(session, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return HistoryDetail(**row.to_detail())


@app.delete("/history/{run_id}", status_code=204)
def delete_history_run(run_id: str, session: Session = Depends(get_session)) -> None:
    if not delete_run(session, run_id):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")


def _build_realized_payload(row) -> HistoryRealizedResponse:
    steps = parse_horizon_steps(row.horizon)
    realized = fetch_realized_forward(
        target_date=row.document_date,
        symbol=row.symbol,
        steps=steps,
    )
    return HistoryRealizedResponse(
        run_id=row.id,
        symbol=row.symbol,
        document_date=row.document_date,
        horizon=row.horizon,
        timestamps=[str(point["date"]) for point in realized],
        close=[float(point["close"]) for point in realized],
        volatility=[float(point["volatility_5d"]) for point in realized],
        realized_regime=bucket_realized_regime(
            float(realized[-1]["volatility_5d"]) if realized else None
        ),
    )


@app.get("/history/{run_id}/realized", response_model=HistoryRealizedResponse)
def get_history_run_realized(
    run_id: str, session: Session = Depends(get_session)
) -> HistoryRealizedResponse:
    row = get_run(session, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    try:
        return _build_realized_payload(row)
    except Exception as exc:  # pragma: no cover — yfinance failures bubble as 502
        raise HTTPException(status_code=502, detail=f"Market lookup failed: {exc}") from exc


def _predicted_regime_from_payload(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    regime = payload.get("regime_classification")
    if isinstance(regime, dict):
        argmax = regime.get("argmax_class")
        if isinstance(argmax, str) and argmax:
            return argmax
    return None


def _realized_vol_from_log_returns(log_returns: list[float]) -> float | None:
    if len(log_returns) < 2:
        return None
    mean = sum(log_returns) / len(log_returns)
    var = sum((value - mean) ** 2 for value in log_returns) / (len(log_returns) - 1)
    return math.sqrt(max(var, 0.0))


@app.get("/history/{run_id}/event-study", response_model=HistoryEventStudyResponse)
def get_history_run_event_study(
    run_id: str, session: Session = Depends(get_session)
) -> HistoryEventStudyResponse:
    """Forward 10-trading-day close path + bucketed realised regime.

    Powers the event-study chart on /history/[id]. Pulls the next 10
    trading bars after the stored event date from yfinance, computes
    log-returns and the realised-vol bucket, and surfaces both predicted
    and realised regime labels for the headline.
    """

    row = get_run(session, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    try:
        bars = fetch_event_study_window(
            event_date=row.document_date,
            symbol=row.symbol,
            steps=10,
            window_days=30,
        )
    except Exception as exc:  # pragma: no cover — yfinance failures bubble as 502
        raise HTTPException(status_code=502, detail=f"Market lookup failed: {exc}") from exc

    forward_dates = [str(bar["date"]) for bar in bars]
    forward_close = [float(bar["close"]) for bar in bars]
    forward_log_returns = [float(bar["log_return"]) for bar in bars]
    realized_vol_10d = _realized_vol_from_log_returns(forward_log_returns)
    return HistoryEventStudyResponse(
        event_date=row.document_date,
        symbol=row.symbol,
        forward_dates=forward_dates,
        forward_close=forward_close,
        forward_log_returns=forward_log_returns,
        realized_vol_10d=realized_vol_10d,
        predicted_regime=_predicted_regime_from_payload(row.payload),
        realized_regime=bucket_realized_regime(realized_vol_10d),
    )


@app.get("/history-realized", response_model=HistoryRealizedBatchResponse)
def get_history_realized_batch(
    ids: str = Query(..., description="Comma-separated run IDs (max 50)"),
    session: Session = Depends(get_session),
) -> HistoryRealizedBatchResponse:
    """Batch the per-row realized fetch the /history list page used to do
    one row at a time. ``ids`` is the comma-joined run-id list; deleted
    rows and yfinance failures land under ``missing`` so a single broken
    row does not nuke the table render."""

    # Order-preserving dedupe so a caller passing the same id twice
    # only triggers one yfinance lookup, and the response order matches
    # the request order on the way back.
    seen: set[str] = set()
    id_list: list[str] = []
    for chunk in ids.split(","):
        chunk = chunk.strip()
        if not chunk or chunk in seen:
            continue
        seen.add(chunk)
        id_list.append(chunk)

    if not id_list:
        raise HTTPException(
            status_code=422, detail="ids must contain at least one run id"
        )
    if len(id_list) > 50:
        raise HTTPException(
            status_code=422,
            detail=f"ids must contain at most 50 run ids; got {len(id_list)}",
        )

    items: dict[str, HistoryRealizedResponse] = {}
    missing: list[str] = []
    for run_id in id_list:
        row = get_run(session, run_id)
        if row is None:
            missing.append(run_id)
            continue
        try:
            items[run_id] = _build_realized_payload(row)
        except Exception:  # pragma: no cover — partial failures degrade silently
            missing.append(run_id)
    return HistoryRealizedBatchResponse(items=items, missing=missing)


# In-memory cache for /evaluation/coverage. Aggregation walks up to
# ``lookback_runs`` history rows and triggers one yfinance call per row,
# so cold-cache latency can climb into tens of seconds. The default is
# tightened to 25 and the hard cap to 100 to keep the workspace
# headline chip from pinning a worker on first hit; a 5-minute TTL
# absorbs repeated visits. A persisted realized-regime column on the
# history row would let us drop the yfinance hop entirely — tracked as
# the next-step polish item.
_COVERAGE_CACHE_TTL_SECONDS = 5 * 60
_coverage_cache: dict[str, tuple[float, "EvaluationCoverageResponse"]] = {}


def _reset_coverage_cache() -> None:
    _coverage_cache.clear()


@app.get("/evaluation/coverage", response_model=EvaluationCoverageResponse)
def evaluation_coverage(
    symbol: str | None = Query(default=None),
    lookback_runs: int = Query(default=25, ge=1, le=100),
    session: Session = Depends(get_session),
) -> EvaluationCoverageResponse:
    """Aggregate empirical conformal coverage across recent history runs.

    Empirical = fraction of runs where the realized regime label fell
    inside that run's predicted set. Nominal is the conformal target the
    active model was calibrated to (read off the most-recent run that
    carries ``series.forecast_confidence_level``). Rows without a
    ``regime_classification.predicted_set`` or without a fetchable
    realized regime are skipped. Results cached for 5 minutes."""

    cache_key = f"{symbol or '*'}:{lookback_runs}"
    cached = _coverage_cache.get(cache_key)
    now = time.monotonic()
    if cached and now - cached[0] < _COVERAGE_CACHE_TTL_SECONDS:
        return cached[1]

    query = session.query(AnalysisRun).order_by(AnalysisRun.created_at.desc())
    if symbol:
        query = query.filter(AnalysisRun.symbol == symbol)
    rows = query.limit(lookback_runs).all()

    nominal: float | None = None
    inside = 0
    sample_size = 0
    for row in rows:
        payload = row.payload if isinstance(row.payload, dict) else {}
        regime = payload.get("regime_classification") or {}
        predicted_set = regime.get("predicted_set")
        if not isinstance(predicted_set, list) or not predicted_set:
            continue
        if nominal is None:
            series = payload.get("series") or {}
            level = series.get("forecast_confidence_level")
            if isinstance(level, int | float):
                nominal = float(level)
        try:
            realized = _build_realized_payload(row)
        except Exception:  # pragma: no cover — yfinance failures skip the row
            continue
        if realized.realized_regime is None:
            continue
        sample_size += 1
        if realized.realized_regime in predicted_set:
            inside += 1

    response = EvaluationCoverageResponse(
        nominal=nominal,
        empirical=(inside / sample_size) if sample_size else None,
        sample_size=sample_size,
        runs_total=len(rows),
        computed_at=datetime.now(timezone.utc).isoformat(),
    )
    _coverage_cache[cache_key] = (now, response)
    return response


@app.get(
    "/evaluation/classification-breakdown",
    response_model=ClassificationBreakdownResponse,
)
def evaluation_classification_breakdown() -> ClassificationBreakdownResponse:
    """Surface the freshest classification breakdown written by the
    regime training scripts under ``data/artifacts/regime_*``. When no
    qualifying artifact exists the response is ``available=False`` and
    the /performance dashboard falls back to its client-side
    aggregation."""

    payload = load_classification_breakdown(DATA_DIR / "artifacts")
    if payload is None:
        return ClassificationBreakdownResponse(available=False)
    return ClassificationBreakdownResponse(
        available=True,
        confusion_matrix=payload.confusion_matrix,
        per_class=[ClassificationBreakdownClass(**row) for row in payload.per_class],
        macro_f1=payload.macro_f1,
        macro_precision=payload.macro_precision,
        macro_recall=payload.macro_recall,
        macro_roc_auc=payload.macro_roc_auc,
        macro_pr_auc=payload.macro_pr_auc,
        weighted_f1=payload.weighted_f1,
        n_classes=payload.n_classes,
        class_labels=payload.class_labels,
        source=ClassificationBreakdownSource(
            relative_path=payload.source_relative,
            training_package_id=payload.training_package_id,
            checkpoint_path=payload.checkpoint_path,
            modified_at=payload.modified_at,
        ),
    )


@app.get("/fomc/calendar", response_model=FomcCalendarResponse)
def fomc_calendar(
    upcoming_limit: int = Query(default=12, ge=1, le=60),
    past_limit: int = Query(default=12, ge=0, le=60),
    as_of: str | None = Query(default=None, description="YYYY-MM-DD; defaults to today"),
) -> FomcCalendarResponse:
    reference: date | None = None
    if as_of:
        try:
            reference = date.fromisoformat(as_of)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="as_of must be YYYY-MM-DD") from exc
    calendar = get_calendar(
        as_of=reference, upcoming_limit=upcoming_limit, past_limit=past_limit
    )
    return FomcCalendarResponse(
        past=[meeting.to_dict() for meeting in calendar["past"]],  # type: ignore[arg-type]
        upcoming=[meeting.to_dict() for meeting in calendar["upcoming"]],  # type: ignore[arg-type]
    )


@app.post("/documents/parse", response_model=DocumentParseResponse)
async def parse_document(
    text: str | None = Form(default=None),
    url: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
) -> DocumentParseResponse:
    """Normalise a document from one of three modes into the same plain-text
    shape the analyze form consumes. Exactly one of `text`, `url`, or `file`
    must be supplied."""

    provided = [name for name, value in (("text", text), ("url", url), ("file", file)) if value]
    if not provided:
        raise HTTPException(status_code=422, detail="Provide one of text, url, or file.")
    if len(provided) > 1:
        raise HTTPException(
            status_code=422,
            detail=f"Provide exactly one of text/url/file; got {provided}.",
        )

    if text is not None:
        parsed = parse_paste(text)
        return DocumentParseResponse(**parsed.to_dict())

    if url is not None:
        try:
            parsed = await parse_url(url)
        except httpx.HTTPError as exc:  # pragma: no cover
            raise HTTPException(status_code=502, detail=f"URL fetch failed: {exc}") from exc
        return DocumentParseResponse(**parsed.to_dict())

    assert file is not None
    content_type = (file.content_type or "").lower()
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=422, detail="Empty upload")
    stream = io.BytesIO(payload)
    if content_type == "application/pdf" or (file.filename or "").lower().endswith(".pdf"):
        parsed = parse_pdf_stream(stream, filename=file.filename)
    elif content_type in {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    } or (file.filename or "").lower().endswith(".docx"):
        parsed = parse_docx_stream(stream, filename=file.filename)
    else:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type {content_type or 'unknown'} (expected PDF or DOCX)",
        )
    return DocumentParseResponse(**parsed.to_dict())


# ---------------------------------------------------------------------------
# Research artifacts (#150 — /research tab)
# ---------------------------------------------------------------------------


@app.get("/research/artifacts", response_model=ResearchArtifactsResponse)
def research_artifacts() -> ResearchArtifactsResponse:
    """Aggregate artefact metadata + parsed shapes for the research tab.

    Missing sections are not an error; the response marks each one
    unavailable so the dashboard can render an empty state without
    issuing a second request.
    """

    artifacts_root = DATA_DIR / "artifacts"
    sections: dict[str, list[ArtifactFile]] = {}
    for name in RESEARCH_SECTIONS:
        infos = list_section_files(artifacts_root, name)
        sections[name] = [
            ArtifactFile(
                relative_path=info.relative_path,
                size_bytes=info.size_bytes,
                modified_at=info.modified_at,
                suffix=info.suffix,
            )
            for info in infos
        ]
    bakeoff = load_encoder_bakeoff(artifacts_root)
    transfer = load_cross_bank_transfer(artifacts_root)
    return ResearchArtifactsResponse(
        artifacts_root=str(artifacts_root),
        sections=sections,
        encoder_bakeoff=bakeoff,  # type: ignore[arg-type]
        cross_bank_transfer=transfer,  # type: ignore[arg-type]
    )


@app.get("/research/registry", response_model=ResearchRegistryResponse)
def research_registry(
    surface: str = "dual",
    include_rejected: bool = False,
) -> ResearchRegistryResponse:
    """Quant-facing encoder bake-off registry from the §6.41 manifest.

    Filtered by default to rows with non-negative Δ on the active
    surface (``dual`` or ``cls``) so the dashboard never surfaces
    negative-lift encoders. Pass ``include_rejected=true`` to see the
    full table including nulls/negatives.
    """

    if surface not in {"dual", "cls"}:
        raise HTTPException(status_code=400, detail="surface must be 'dual' or 'cls'")
    payload = load_research_registry(surface=surface, include_rejected=include_rejected)
    return ResearchRegistryResponse(**payload)


@app.post("/research/backtest", response_model=BacktestResponse)
def research_backtest(request: BacktestRequest) -> BacktestResponse:
    """Run the stance-directional backtest engine on a caller-supplied
    {date, position} series.

    Frontend (or any caller) supplies the positions; this endpoint
    looks up the S&P forward returns per date and aggregates Sharpe,
    hit-rate, max-drawdown, and benchmark deltas. Decouples the
    engine from any specific signal source so the same harness can
    serve oracle backtests, history-driven backtests, and live-
    classifier backtests interchangeably.
    """

    from app.evaluation.backtest import compute_backtest_metrics

    payload = compute_backtest_metrics(
        positions=[entry.model_dump() for entry in request.positions],
        symbol=request.symbol,
        horizon_days=request.horizon_days,
    )
    return BacktestResponse(**payload)


# ---------------------------------------------------------------------------
# Next-FOMC decision forecast (#150 — /decisions tab)
# ---------------------------------------------------------------------------


@app.get("/forecasts/next-fomc", response_model=NextFomcForecastResponse)
def next_fomc_forecast() -> NextFomcForecastResponse:
    """Read ``data/artifacts/next_fomc/`` for the decisions dashboard.

    Returns ``available: False`` when the forecaster has not been run
    against this checkout. The dashboard surfaces the empty-state with
    the documented ``make next-fomc`` instruction in that case.
    """

    artifacts_dir = DATA_DIR / "artifacts" / "next_fomc"
    payload = load_next_fomc_artifacts(artifacts_dir)
    return NextFomcForecastResponse(**payload)


# ---------------------------------------------------------------------------
# Market reaction panel (#293 — /analyze/market)
# ---------------------------------------------------------------------------


@app.post("/analyze/market", response_model=MarketReactionPanel)
async def analyze_market(payload: AnalyzeRequest) -> MarketReactionPanel:
    """Build the four-card market-reaction panel for one document (#293).

    Reuses the trained shared encoder + per-fold scalers persisted on
    the active forecaster checkpoint. Returns an empty payload (empty
    ``rates`` list + ``vol_regime=None``) when the active checkpoint
    has neither rates heads nor a classification surface mounted, so a
    fresh checkout that has not yet run the rates-head sweep does not
    surface a 5xx.
    """

    # #317 finding #12: mirror the /analyze cold-start guard so a
    # fresh deploy hitting /analyze/market first does not surface
    # empty cards. Shared with /analyze via the _bootstrap_cold_start
    # helper.
    # #342: cold-start now invokes the canonical loader for contract
    # revalidation; a sidecar mismatch raises RuntimeError. Surface it
    # symmetrically to /analyze — 503 with the structured-status detail
    # (the message is a deliberate contract code, not raw exception
    # text) so the operator sees the same shape on both endpoints.
    # #379: route every unhandled exception path through HTTPException
    # so the response stays inside the CORS-wrapped ExceptionMiddleware
    # layer; ServerErrorMiddleware (outside CORS) was returning bare
    # 500s without Access-Control-Allow-Origin on cold-start races.
    try:
        await _ensure_cold_start(payload)
    except RuntimeError as exc:
        logger.warning("analyze_market_cold_start_contract_mismatch detail=%s", str(exc))
        raise HTTPException(
            status_code=503,
            detail="Market reaction panel unavailable: serving contract mismatch",
        ) from None
    except Exception as exc:  # pragma: no cover -- cold-start race fallback
        logger.exception(
            "analyze_market_cold_start_unexpected exception_class=%s",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="Market reaction panel unavailable: cold-start failed",
        ) from None

    # Request-shape errors (bad date string from the client, etc.) must
    # surface as 422 via the registered ``_value_error_handler`` so the
    # contract tests + the frontend toast layer keep their existing
    # validation semantics; only true server-side failures collapse to
    # the structured 503 below.
    sentiment = analyze_text(payload.text)
    market_history = await run_in_threadpool(
        fetch_market_history,
        target_date=payload.date,
        symbol=payload.symbol,
        history_length=30,
    )
    pooled_text_embedding = encode_text_pooled(payload.text)
    history_vectors = build_feature_vectors(
        market_history,
        sentiment_score=float(sentiment["score"]),
        document_date=payload.date,
        text_embedding=pooled_text_embedding,
    )
    try:
        result = await run_in_threadpool(
            build_market_reaction_panel, history_vectors
        )
    except Exception:  # pragma: no cover -- defensive
        logger.exception("analyze_market_failed")
        raise HTTPException(
            status_code=503, detail="Market reaction panel unavailable"
        ) from None
    if result is None:
        return MarketReactionPanel(rates=[], vol_regime=None)
    # #341: ``build_market_reaction_panel`` now returns a structured
    # status payload on the soft-error paths instead of bare None. The
    # status field is mutually exclusive with the panel fields -- a
    # status-only payload renders as an empty panel surface on the
    # frontend, so we collapse it to the legacy empty-panel response
    # here. The structured detail is logged at the service layer; the
    # client gets an honest "no evidence" panel without a 5xx.
    if isinstance(result, dict) and "rates" not in result and result.get("status"):
        logger.info(
            "market_reaction_panel_status status=%s detail=%s",
            result.get("status"),
            result.get("detail"),
        )
        return MarketReactionPanel(rates=[], vol_regime=None)
    cards = [RatesReactionCard(**row) for row in result.get("rates", [])]
    vol_regime_payload = result.get("vol_regime")
    vol_regime = (
        VolRegimeReactionCard(**vol_regime_payload)
        if vol_regime_payload is not None
        else None
    )
    return MarketReactionPanel(
        rates=cards,
        vol_regime=vol_regime,
        encoder_alias=result.get("encoder_alias"),
        checkpoint_path=result.get("checkpoint_path"),
    )


# ---------------------------------------------------------------------------
# Historical analog retrieval (#294 — /analyze/analogs)
# ---------------------------------------------------------------------------


_DEFAULT_RETRIEVAL_ENCODER_ALIAS = "finbert_fed_adjacent_xbank_dapt_retrieval"


@app.post("/analyze/analogs", response_model=AnalogsResponse)
async def analyze_analogs(payload: AnalogsRequest) -> AnalogsResponse:
    """Return historical FOMC statements that semantically match ``text``.

    Loads the fine-tuned retrieval encoder + persisted analog index on
    first hit via the singleton at ``app.services.analogs``. When no
    bundle is present (fresh checkout, encoder not yet trained) the
    response is shaped with an empty ``analogs`` list and
    ``index_size=0`` so the frontend can render an empty state rather
    than treating the call as a 5xx. Unexpected failures surface as a
    sanitized 503 — internal paths / exception text never leak to the
    client.
    """

    from app.services import analogs as analogs_service

    # Pre-flight: if the bundle is permanently absent on disk and no
    # state has been installed (e.g. by the test suite), short-circuit
    # rather than paying ``_load_state`` + log a cold-miss on every
    # request.
    if not analogs_service.bundle_available() and analogs_service.get_state() is None:
        return AnalogsResponse(
            analogs=[],
            index_size=0,
            encoder_alias=_DEFAULT_RETRIEVAL_ENCODER_ALIAS,
        )

    try:
        result = await run_in_threadpool(
            analogs_service.find_analogs,
            payload.text,
            k=payload.k,
            as_of_date=payload.as_of_date,
        )
    except ValueError:
        # Surface a client-safe message instead of echoing the raw
        # exception text (file paths, library internals).
        logger.warning("analyze_analogs_validation_failed", exc_info=True)
        raise HTTPException(status_code=422, detail="Invalid analog query") from None
    except Exception:  # never let a downstream failure 500 the whole API
        logger.exception("analyze_analogs_failed")
        raise HTTPException(
            status_code=503, detail="Analog retrieval unavailable"
        ) from None

    if result is None:
        return AnalogsResponse(
            analogs=[],
            index_size=0,
            encoder_alias=_DEFAULT_RETRIEVAL_ENCODER_ALIAS,
        )

    cards = [AnalogCard(**row) for row in analogs_service.render_analog_cards(result["hits"])]
    return AnalogsResponse(
        analogs=cards,
        index_size=int(result["index_size"]),
        encoder_alias=str(result["encoder_alias"]),
    )


# ---------------------------------------------------------------------------
# Hawkish/dovish trajectory model (#296 — /analyze/trajectory)
# ---------------------------------------------------------------------------


_DEFAULT_TRAJECTORY_ENCODER_ALIAS = "finbert_fed_adjacent_xbank_dapt"


@app.post("/analyze/trajectory", response_model=TrajectoryResponse)
async def analyze_trajectory(payload: TrajectoryRequest) -> TrajectoryResponse:
    """Project the FOMC stance trajectory ending at ``as_of_date``.

    Loads the trajectory bundle (LSTM or Transformer arm + per-meeting
    embedding index) on first hit via the singleton at
    :mod:`app.services.trajectory`. When no bundle is present the
    response is shaped with ``available=False`` and an empty
    ``history`` so the frontend renders an empty state rather than
    treating the call as a 5xx. Unexpected failures surface as a
    sanitized 503.
    """

    from app.services import trajectory as trajectory_service

    if not trajectory_service.bundle_available() and trajectory_service.get_state() is None:
        return TrajectoryResponse(
            available=False,
            encoder_alias=_DEFAULT_TRAJECTORY_ENCODER_ALIAS,
            history=[],
            projected_next=None,
            history_length=int(payload.history_length),
            as_of_date=payload.as_of_date.isoformat(),
        )

    state = trajectory_service.get_state()
    if state is None:
        return TrajectoryResponse(
            available=False,
            encoder_alias=_DEFAULT_TRAJECTORY_ENCODER_ALIAS,
            history=[],
            projected_next=None,
            history_length=int(payload.history_length),
            as_of_date=payload.as_of_date.isoformat(),
        )

    try:
        result = await run_in_threadpool(
            trajectory_service.project_trajectory,
            state,
            as_of_date=payload.as_of_date,
            history_length=payload.history_length,
        )
    except ValueError:
        logger.warning("analyze_trajectory_validation_failed", exc_info=True)
        raise HTTPException(status_code=422, detail="Invalid trajectory query") from None
    except Exception:  # never let a downstream failure 500 the whole API
        logger.exception("analyze_trajectory_failed")
        raise HTTPException(
            status_code=503, detail="Trajectory projection unavailable"
        ) from None

    history = [TrajectoryMarker(**marker) for marker in result.get("history", [])]
    projection_payload = result.get("projected_next")
    projection = (
        TrajectoryProjection(**projection_payload)
        if projection_payload is not None
        else None
    )
    return TrajectoryResponse(
        available=bool(result.get("available", False)),
        history=history,
        projected_next=projection,
        architecture=result.get("architecture"),
        encoder_alias=str(result.get("encoder_alias") or _DEFAULT_TRAJECTORY_ENCODER_ALIAS),
        history_length=int(result.get("history_length", payload.history_length)),
        train_end=result.get("train_end"),
        as_of_date=str(result.get("as_of_date") or payload.as_of_date.isoformat()),
        warning=result.get("warning"),
    )
