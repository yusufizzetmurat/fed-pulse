import io
import json
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from typing import Any

import httpx
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
    DocumentParseResponse,
    DocumentParseUrlRequest,
    FomcCalendarResponse,
    HistoryDetail,
    HistoryEntry,
    HistoryList,
    TrainJobAcceptedResponse,
    TrainJobStatusResponse,
)
from app.evaluation.xai import attribute_text, to_response as xai_to_response
from app.services.document_parser import (
    parse_docx_stream,
    parse_paste,
    parse_pdf_stream,
    parse_url,
)
from app.services.fomc_calendar import get_calendar
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

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001
    configure_logging()
    get_engine()
    # Pre-load the HF classifier so the first /analyze request doesn't pay
    # the cold-start cost. The lazy module-level cache keeps this idempotent.
    try:
        from app.services.text_encoder import warmup_classifier

        warmup_classifier()
    except Exception:  # pragma: no cover — never let model warmup block startup
        get_logger("fed_pulse").warning("classifier_warmup_failed", exc_info=True)
    get_logger("fed_pulse").info("startup", service="fomc-api")
    try:
        yield
    finally:
        get_logger("fed_pulse").info("shutdown", service="fomc-api")


app = FastAPI(title="FOMC Sentiment API", version="0.1.0", lifespan=_lifespan)
REAL_TRAIN_HISTORY_LENGTH = 252

_train_jobs: dict[str, dict[str, Any]] = {}
_train_jobs_lock = threading.Lock()

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


@app.post("/analyze", response_model=AnalyzeResponse | TrainJobAcceptedResponse)
async def analyze(payload: AnalyzeRequest):
    """Async handler — heavy sync work (transformers, yfinance, torch) runs in
    the thread pool so the event loop stays responsive under load."""

    try:
        mode = payload.forecast_mode.strip().lower()
        if mode not in {"fast", "quick_train", "real_train"}:
            raise ValueError("forecast_mode must be 'fast', 'quick_train', or 'real_train'")

        if mode == "real_train":
            job_id = str(uuid.uuid4())
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
            thread = threading.Thread(target=_run_real_train_job, args=(job_id, payload), daemon=True)
            thread.start()
            return {
                "status": "queued",
                "job_id": job_id,
                "message": "Real Train started with 252-day history. Poll /train-jobs/{job_id} for progress.",
            }

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


@app.get("/train-jobs/{job_id}", response_model=TrainJobStatusResponse)
def get_train_job(job_id: str):
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
