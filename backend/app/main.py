import io
import json
import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from typing import Any

import httpx
from arq import constants as arq_constants
from arq.connections import ArqRedis, create_pool
from arq.jobs import Job as ArqJob, JobStatus
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.audit import append_audit_entry
from app.config import DATA_DIR
from app.db import (
    delete_run,
    get_engine,
    get_run,
    get_session,
    list_runs,
    persist_analysis_run,
    session_scope,
)
from app.logging import bind_run_id, clear_run_id, configure_logging, get_logger
from app.middleware.errors import RunIdMiddleware, register_error_handlers
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ArtifactFile,
    DocumentParseResponse,
    DocumentParseUrlRequest,
    FomcCalendarResponse,
    HistoryDetail,
    HistoryEntry,
    HistoryList,
    HistoryRealizedResponse,
    NextFomcForecastResponse,
    ResearchArtifactsResponse,
    TrainJobAcceptedResponse,
    TrainJobStatusResponse,
    TrainJobSummary,
    TrainJobsListResponse,
)
from app.evaluation.xai import attribute_text, to_response as xai_to_response
from app.services.document_parser import (
    parse_docx_stream,
    parse_paste,
    parse_pdf_stream,
    parse_url,
)
from app.services.decision_forecast import load_next_fomc_artifacts
from app.services.fomc_calendar import get_calendar
from app.services.research_artifacts import (
    SECTIONS as RESEARCH_SECTIONS,
    list_section_files,
    load_cross_bank_transfer,
    load_encoder_bakeoff,
)
from app.services.forecaster import (
    bootstrap_checkpoint,
    build_feature_vectors,
    checkpoint_exists,
    forecast_quantitative_series,
    parse_horizon_steps,
)
from app.services.market_data import (
    fetch_forward_trading_dates,
    fetch_market_history,
    fetch_market_snapshot,
    fetch_realized_forward,
)
from app.services.sentiment import analyze_text
from app.worker import REAL_TRAIN_HISTORY_LENGTH, get_redis_settings

logger = logging.getLogger(__name__)


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
    # Bring up an arq Redis pool so /analyze can enqueue real_train jobs
    # into a durable queue. A missing or unreachable Redis is not fatal:
    # the endpoint falls back to the in-process daemon thread so dev
    # boxes without docker compose still work. ``DISABLE_REDIS_POOL`` is
    # a test-only escape hatch that skips the connect attempt entirely
    # (the default arq retry loop is otherwise long enough to slow the
    # test suite noticeably).
    pool: ArqRedis | None = None
    if os.environ.get("DISABLE_REDIS_POOL") not in {"1", "true", "TRUE"}:
        try:
            pool = await create_pool(get_redis_settings())
            # Force a round-trip so an unreachable Redis fails fast instead of
            # the first /analyze call discovering it the hard way.
            await pool.ping()
            get_logger("fed_pulse").info("arq_pool_ready", service="fomc-api")
        except Exception:
            get_logger("fed_pulse").warning(
                "arq_pool_unavailable",
                service="fomc-api",
                exc_info=True,
            )
            if pool is not None:  # pragma: no cover — partial init unwinds
                try:
                    await pool.close(close_connection_pool=True)
                except Exception:
                    pass
            pool = None
    app.state.redis_pool = pool
    get_logger("fed_pulse").info("startup", service="fomc-api")
    try:
        yield
    finally:
        if pool is not None:
            try:
                await pool.close(close_connection_pool=True)
            except Exception:  # pragma: no cover
                get_logger("fed_pulse").warning("arq_pool_close_failed", exc_info=True)
        get_logger("fed_pulse").info("shutdown", service="fomc-api")


app = FastAPI(title="FOMC Sentiment API", version="0.1.0", lifespan=_lifespan)

# In-memory fallback only. The primary store is Redis via arq; this dict is
# read/written only when ``app.state.redis_pool`` is None (dev environments
# without a Redis container). It keeps the surface backwards-compatible
# with the pre-#103 daemon-thread path.
_train_jobs: dict[str, dict[str, Any]] = {}
_train_jobs_lock = threading.Lock()


def _redis_pool() -> ArqRedis | None:
    """Return the arq pool stashed on ``app.state`` during lifespan, if any."""

    return getattr(app.state, "redis_pool", None)

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
    return {"status": "ok"}


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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_job_state(job_id: str, **patch: Any) -> None:
    with _train_jobs_lock:
        state = _train_jobs.get(job_id)
        if state is None:
            return
        state.update(patch)


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

    history_vectors = build_feature_vectors(market_history, sentiment_score=float(sentiment["score"]), document_date=payload.date)
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

    response: dict[str, Any] = {
        "sentiment": sentiment,
        "prediction": forecast["prediction"],
        "market": market,
        "model": forecast["model"],
        "series": forecast["series"],
    }
    if getattr(payload, "include_xai", False):
        attributions = attribute_text(payload.text)
        response["xai"] = xai_to_response(attributions)
    return response


def _run_real_train_job(job_id: str, payload: AnalyzeRequest) -> None:
    # Bind run_id on the daemon thread so checkpoint/audit hooks downstream
    # tag every log line and audit row with the same id as the API caller.
    bind_run_id(job_id)
    try:
        _set_job_state(job_id, status="running", started_at=_utc_now_iso())
        _run_real_train_job_body(job_id, payload)
    finally:
        clear_run_id()


def _run_real_train_job_body(job_id: str, payload: AnalyzeRequest) -> None:
    try:
        sentiment = analyze_text(payload.text)
        market_history = fetch_market_history(
            target_date=payload.date,
            symbol=payload.symbol,
            history_length=REAL_TRAIN_HISTORY_LENGTH,
        )
        history_vectors = build_feature_vectors(market_history, sentiment_score=float(sentiment["score"]), document_date=payload.date)

        # Real Train intentionally runs a stronger checkpoint update over 252-day context.
        bootstrap_checkpoint(
            vectors=history_vectors,
            epochs=120,
            batch_size=64,
            learning_rate=3e-4,
            validation_split=0.2,
            early_stopping_patience=12,
        )
        result = _build_analyze_response(payload, mode="real_train", history_length=REAL_TRAIN_HISTORY_LENGTH)
        _record_history(payload, result)
        _set_job_state(
            job_id,
            status="succeeded",
            result=result,
            finished_at=_utc_now_iso(),
        )
        try:
            append_audit_entry(
                "real_train_completed",
                run_id=job_id,
                metadata={"symbol": payload.symbol, "date": payload.date},
            )
        except Exception:  # pragma: no cover
            get_logger("fed_pulse").warning("audit_write_failed", run_id=job_id)
    except Exception as exc:  # pragma: no cover
        _set_job_state(
            job_id,
            status="failed",
            error=f"Real train job failed: {exc}",
            finished_at=_utc_now_iso(),
        )
        try:
            append_audit_entry(
                "real_train_failed",
                run_id=job_id,
                metadata={"symbol": payload.symbol, "date": payload.date, "error": str(exc)},
            )
        except Exception:
            get_logger("fed_pulse").warning("audit_write_failed", run_id=job_id)


async def _enqueue_real_train(payload: AnalyzeRequest) -> dict[str, Any]:
    """Enqueue a Real-Train job through the arq Redis pool when available,
    falling back to the legacy in-process daemon thread otherwise.

    The fallback path keeps dev environments without a Redis container
    working and preserves the pre-#103 response shape so the existing
    frontend polling loop is unaffected.
    """

    job_id = str(uuid.uuid4())
    pool = _redis_pool()
    if pool is not None:
        await pool.enqueue_job(
            "real_train_task",
            payload.model_dump(),
            _job_id=job_id,
        )
        return {
            "status": "queued",
            "job_id": job_id,
            "message": "Real Train started with 252-day history. Poll /train-jobs/{job_id} for progress.",
        }

    # Fallback: in-process daemon thread + in-memory job map. Same response
    # shape as the Redis path so callers cannot tell the two apart.
    job_state: dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "error": None,
        "started_at": None,
        "finished_at": None,
        "result": None,
        "created_at": _utc_now_iso(),
        "history_length": REAL_TRAIN_HISTORY_LENGTH,
        "symbol": payload.symbol,
        "date": payload.date,
    }
    with _train_jobs_lock:
        _train_jobs[job_id] = job_state
    thread = threading.Thread(
        target=_run_real_train_job, args=(job_id, payload), daemon=True
    )
    thread.start()
    return {
        "status": "queued",
        "job_id": job_id,
        "message": "Real Train started with 252-day history. Poll /train-jobs/{job_id} for progress.",
    }


@app.post("/analyze", response_model=AnalyzeResponse | TrainJobAcceptedResponse)
async def analyze(payload: AnalyzeRequest):
    """Async handler — heavy sync work (transformers, yfinance, torch) runs in
    the thread pool so the event loop stays responsive under load."""

    try:
        mode = payload.forecast_mode.strip().lower()
        if mode not in {"fast", "quick_train", "real_train"}:
            raise ValueError("forecast_mode must be 'fast', 'quick_train', or 'real_train'")

        if mode == "real_train":
            return await _enqueue_real_train(payload)

        history_length = 30
        if mode == "fast" and not checkpoint_exists():
            # Bootstrap a first checkpoint so fast-mode inference is not random on cold start.
            await run_in_threadpool(_bootstrap_cold_start, payload)

        response = await run_in_threadpool(
            _build_analyze_response, payload, mode=mode, history_length=history_length
        )
        await run_in_threadpool(_record_history, payload, response)
        return response
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Analyze pipeline failed: {exc}") from exc


def _bootstrap_cold_start(payload: AnalyzeRequest) -> None:
    """Run the one-shot bootstrap training when the checkpoint is missing.

    Synchronous — invoked via `run_in_threadpool` from the async handler so the
    long-running fit doesn't block the event loop.
    """

    warmup_sentiment = analyze_text(payload.text)
    warmup_history = fetch_market_history(
        target_date=payload.date,
        symbol=payload.symbol,
        history_length=REAL_TRAIN_HISTORY_LENGTH,
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


def _iso_or_none(ts: Any) -> str | None:
    """Best-effort ISO-8601 stringifier for arq's datetime fields."""

    if ts is None:
        return None
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat()
    return str(ts)


def _payload_from_args(info: Any) -> dict[str, Any]:
    """Pull the AnalyzeRequest payload out of an arq job def/result."""

    if info is None:
        return {}
    args = getattr(info, "args", None) or ()
    if args and isinstance(args[0], dict):
        return args[0]
    kwargs = getattr(info, "kwargs", None) or {}
    payload = kwargs.get("payload")
    return payload if isinstance(payload, dict) else {}


async def _state_from_redis_job(pool: ArqRedis, job_id: str) -> dict[str, Any] | None:
    """Read job state from Redis and shape it like the legacy ``_train_jobs``
    dict so the response models do not need to change."""

    job = ArqJob(job_id, pool)
    status = await job.status()
    if status == JobStatus.not_found:
        return None
    info = await job.info()
    payload = _payload_from_args(info)
    state: dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "error": None,
        "started_at": None,
        "finished_at": None,
        "result": None,
        "created_at": _iso_or_none(getattr(info, "enqueue_time", None)),
        "history_length": REAL_TRAIN_HISTORY_LENGTH,
        "symbol": payload.get("symbol"),
        "date": payload.get("date"),
    }
    if status == JobStatus.in_progress:
        state["status"] = "running"
        result_info = await job.result_info()
        if result_info is not None:
            state["started_at"] = _iso_or_none(result_info.start_time)
        return state
    if status == JobStatus.complete:
        result_info = await job.result_info()
        if result_info is None:
            state["status"] = "succeeded"
            return state
        state["started_at"] = _iso_or_none(result_info.start_time)
        state["finished_at"] = _iso_or_none(result_info.finish_time)
        if result_info.success:
            state["status"] = "succeeded"
            state["result"] = result_info.result if isinstance(result_info.result, dict) else None
        else:
            state["status"] = "failed"
            err = result_info.result
            state["error"] = (
                f"Real train job failed: {err}" if err is not None else "Real train job failed"
            )
        return state
    # queued / deferred share the same surface state from the API's view.
    return state


async def _list_redis_job_ids(pool: ArqRedis) -> list[str]:
    """Enumerate every job arq knows about: queued/in-progress (``arq:job:*``)
    plus completed (``arq:result:*``). Both prefixes are stripped to recover
    the bare job id."""

    seen: set[str] = set()
    for prefix in (arq_constants.job_key_prefix, arq_constants.result_key_prefix):
        match = f"{prefix}*"
        async for key in pool.scan_iter(match=match):
            key_str = key.decode() if isinstance(key, bytes) else key
            seen.add(key_str[len(prefix):])
    return sorted(seen)


@app.get("/train-jobs/{job_id}", response_model=TrainJobStatusResponse)
async def get_train_job(job_id: str):
    pool = _redis_pool()
    if pool is not None:
        state = await _state_from_redis_job(pool, job_id)
        if state is not None:
            return state
        # Fall through to the in-memory map -- a daemon-thread job submitted
        # while Redis was down still has to be reachable.
    with _train_jobs_lock:
        state = _train_jobs.get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Train job not found: {job_id}")
        return dict(state)


def _record_history(request: AnalyzeRequest, response: dict[str, Any]) -> None:
    # Persistence must not break /analyze; log so silent failures (disk full,
    # missing table, etc.) still show up in uvicorn output.
    try:
        with session_scope() as session:
            persist_analysis_run(
                session,
                payload=response,
                request=request.model_dump(),
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
    offset: int = Query(default=0, ge=0),
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


@app.get("/history/{run_id}/realized", response_model=HistoryRealizedResponse)
def get_history_run_realized(
    run_id: str, session: Session = Depends(get_session)
) -> HistoryRealizedResponse:
    row = get_run(session, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    steps = parse_horizon_steps(row.horizon)
    try:
        realized = fetch_realized_forward(
            target_date=row.document_date,
            symbol=row.symbol,
            steps=steps,
        )
    except Exception as exc:  # pragma: no cover — yfinance failures bubble as 502
        raise HTTPException(status_code=502, detail=f"Market lookup failed: {exc}") from exc
    return HistoryRealizedResponse(
        run_id=row.id,
        symbol=row.symbol,
        document_date=row.document_date,
        horizon=row.horizon,
        timestamps=[str(point["date"]) for point in realized],
        close=[float(point["close"]) for point in realized],
        volatility=[float(point["volatility_5d"]) for point in realized],
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
# Train-jobs listing (multi-page expansion #150)
# ---------------------------------------------------------------------------

_TRAIN_JOB_SORT_RANK: dict[str, int] = {
    "running": 0,
    "queued": 1,
    "failed": 2,
    "succeeded": 3,
}


def _train_job_sort_key(state: dict[str, Any]) -> tuple[int, str]:
    status = str(state.get("status") or "queued").lower()
    rank = _TRAIN_JOB_SORT_RANK.get(status, 99)
    # Within a status bucket sort newest-first by created_at; the
    # fallback empty string keeps the ordering deterministic.
    created = str(state.get("created_at") or "")
    return (rank, _invert_iso(created))


def _invert_iso(value: str) -> str:
    """Cheap descending-sort key for ISO timestamps."""

    return "".join(chr(255 - ord(c)) if ord(c) < 255 else c for c in value)


async def _redis_train_jobs_snapshot(pool: ArqRedis) -> list[dict[str, Any]]:
    """Materialise every arq-tracked real_train job into the legacy state
    shape so :func:`_train_job_sort_key` and the response model can ingest
    them without branching on the storage backend."""

    job_ids = await _list_redis_job_ids(pool)
    states: list[dict[str, Any]] = []
    for jid in job_ids:
        state = await _state_from_redis_job(pool, jid)
        if state is not None:
            states.append(state)
    return states


@app.get("/train-jobs", response_model=TrainJobsListResponse)
async def list_train_jobs(
    status: str | None = Query(default=None, description="Filter by status: queued/running/succeeded/failed."),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> TrainJobsListResponse:
    """List real_train jobs from the arq-backed Redis queue.

    Falls back to the in-memory ``_train_jobs`` dict when Redis is
    unreachable -- dev environments without a worker container still
    surface daemon-thread jobs through the dashboard.
    """

    pool = _redis_pool()
    if pool is not None:
        snapshot = await _redis_train_jobs_snapshot(pool)
    else:
        with _train_jobs_lock:
            snapshot = [dict(state) for state in _train_jobs.values()]
    if status:
        wanted = status.strip().lower()
        snapshot = [s for s in snapshot if str(s.get("status") or "").lower() == wanted]
    snapshot.sort(key=_train_job_sort_key)
    total = len(snapshot)
    sliced = snapshot[offset : offset + limit]
    items = [
        TrainJobSummary(
            job_id=str(state.get("job_id")),
            status=str(state.get("status") or "queued"),
            symbol=state.get("symbol"),
            date=state.get("date"),
            created_at=state.get("created_at"),
            started_at=state.get("started_at"),
            finished_at=state.get("finished_at"),
            history_length=state.get("history_length"),
            error=state.get("error"),
        )
        for state in sliced
    ]
    return TrainJobsListResponse(items=items, total=total, limit=limit, offset=offset)


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
