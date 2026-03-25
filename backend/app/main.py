import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import AnalyzeRequest, AnalyzeResponse, TrainJobAcceptedResponse, TrainJobStatusResponse
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

app = FastAPI(title="FOMC Sentiment API", version="0.1.0")
DATA_DIR = Path("/data")
REAL_TRAIN_HISTORY_LENGTH = 252

_train_jobs: dict[str, dict[str, Any]] = {}
_train_jobs_lock = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    history_vectors = build_feature_vectors(market_history, sentiment_score=float(sentiment["score"]))
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

    return {
        "sentiment": sentiment,
        "prediction": forecast["prediction"],
        "market": market,
        "model": forecast["model"],
        "series": forecast["series"],
    }


def _run_real_train_job(job_id: str, payload: AnalyzeRequest) -> None:
    _set_job_state(job_id, status="running", started_at=_utc_now_iso())
    try:
        sentiment = analyze_text(payload.text)
        market_history = fetch_market_history(
            target_date=payload.date,
            symbol=payload.symbol,
            history_length=REAL_TRAIN_HISTORY_LENGTH,
        )
        history_vectors = build_feature_vectors(market_history, sentiment_score=float(sentiment["score"]))

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
        _set_job_state(
            job_id,
            status="succeeded",
            result=result,
            finished_at=_utc_now_iso(),
        )
    except Exception as exc:  # pragma: no cover
        _set_job_state(
            job_id,
            status="failed",
            error=f"Real train job failed: {exc}",
            finished_at=_utc_now_iso(),
        )


@app.post("/analyze", response_model=AnalyzeResponse | TrainJobAcceptedResponse)
def analyze(payload: AnalyzeRequest):
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
            warmup_sentiment = analyze_text(payload.text)
            warmup_history = fetch_market_history(
                target_date=payload.date,
                symbol=payload.symbol,
                history_length=REAL_TRAIN_HISTORY_LENGTH,
            )
            warmup_vectors = build_feature_vectors(
                warmup_history,
                sentiment_score=float(warmup_sentiment["score"]),
            )
            bootstrap_checkpoint(
                vectors=warmup_vectors,
                epochs=60,
                batch_size=64,
                learning_rate=4e-4,
                early_stopping_patience=8,
            )

        return _build_analyze_response(payload, mode=mode, history_length=history_length)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Analyze pipeline failed: {exc}") from exc


@app.get("/train-jobs/{job_id}", response_model=TrainJobStatusResponse)
def get_train_job(job_id: str):
    with _train_jobs_lock:
        state = _train_jobs.get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Train job not found: {job_id}")
        return dict(state)
