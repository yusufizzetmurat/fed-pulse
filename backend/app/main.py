import io
import json
import logging
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

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
from app.logging import configure_logging, get_logger
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
    build_regime_classification_card,
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

# 252 trading days ≈ one year of context. Used as the cold-start
# bootstrap history when /analyze fires against a host that has no
# checkpoint on disk yet (first run after a fresh clone).
COLD_START_HISTORY_LENGTH = 252


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
        "multi_axis": _build_multi_axis_block(payload.text, sentiment),
        "regime_classification": _safe_regime_classification(history_vectors),
    }
    if getattr(payload, "include_xai", False):
        attributions = attribute_text(payload.text)
        response["xai"] = xai_to_response(attributions)
    return response


def _safe_regime_classification(history_vectors: list[Any]) -> dict[str, Any] | None:
    """Wrap ``build_regime_classification_card`` so a failure never breaks /analyze.

    The card is opt-in by checkpoint flavour and manifest presence; any
    exception inside the inference + calibrated-set path degrades to
    ``None`` rather than 500ing the whole response.
    """

    try:
        return build_regime_classification_card(history_vectors)
    except Exception:  # pragma: no cover — defensive, see #216 follow-up
        logger.warning("regime_classification_card_failed", exc_info=True)
        return None


def _build_multi_axis_block(
    text: str, sentiment: dict[str, Any]
) -> dict[str, Any] | None:
    """Build the multi-axis card block from the available text signals (#78).

    Two paths:

    1. **Trained multi-axis classifier present.** Run the
       ``TextMultiAxisClassifier`` checkpoint via
       ``app.services.multi_axis_classifier.score_text``; populate all
       four cards (stance / factor / certainty / topic) from the
       per-axis predictions.

    2. **No checkpoint.** Fall back to populating the stance card from
       the existing sentiment classifier output and leave the other
       three axes at ``None``. The frontend renders ``None`` cards as
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

    distribution: dict[str, float] = {key: 0.0 for key in canonical_labels}
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
        "topic": None,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest):
    """Async handler — heavy sync work (transformers, yfinance, torch) runs in
    the thread pool so the event loop stays responsive under load."""

    try:
        if not checkpoint_exists():
            # Cold-start bootstrap: a fresh clone has no checkpoint on disk yet,
            # so seed one against a 252-day window before the first inference.
            await run_in_threadpool(_bootstrap_cold_start, payload)

        response = await run_in_threadpool(
            _build_analyze_response, payload, mode="fast", history_length=30
        )
        await run_in_threadpool(_record_history, payload, response)
        return response
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Analyze pipeline failed: {exc}") from exc


def _bootstrap_cold_start(payload: AnalyzeRequest) -> None:
    """Seed an initial checkpoint when the model file is missing.

    Synchronous — invoked via ``run_in_threadpool`` from the async
    handler so the long-running fit does not block the event loop.
    """

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
