import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import AnalyzeRequest, AnalyzeResponse
from app.services.forecaster import FeatureVector, forecast_quantitative_series, parse_horizon_steps
from app.services.market_data import fetch_market_history, fetch_market_snapshot, fetch_realized_forward
from app.services.sentiment import analyze_text

app = FastAPI(title="FOMC Sentiment API", version="0.1.0")
DATA_DIR = Path("/data")

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


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest):
    try:
        mode = payload.forecast_mode.strip().lower()
        if mode not in {"fast", "quick_train"}:
            raise ValueError("forecast_mode must be either 'fast' or 'quick_train'")

        sentiment = analyze_text(payload.text)
        market = fetch_market_snapshot(target_date=payload.date, symbol=payload.symbol)
        market_history = fetch_market_history(target_date=payload.date, symbol=payload.symbol, history_length=30)

        history_vectors = [
            FeatureVector(
                date=str(point["date"]),
                sentiment_score=float(sentiment["score"]),
                market_close=float(point["close"]),
                market_volatility=float(point["volatility_5d"]),
            )
            for point in market_history
        ]
        forecast = forecast_quantitative_series(
            vectors=history_vectors,
            forecast_mode=mode,
            horizon=payload.horizon,
        )

        if payload.include_realized:
            horizon_steps = parse_horizon_steps(payload.horizon)
            realized = fetch_realized_forward(
                target_date=payload.date,
                symbol=payload.symbol,
                steps=horizon_steps,
            )
            if realized:
                forecast["series"]["realized_timestamps"] = [str(point["date"]) for point in realized]
                forecast["series"]["realized_close"] = [float(point["close"]) for point in realized]
                forecast["series"]["realized_volatility"] = [
                    float(point["volatility_5d"]) for point in realized
                ]

        return {
            "sentiment": sentiment,
            "prediction": forecast["prediction"],
            "market": market,
            "series": forecast["series"],
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Analyze pipeline failed: {exc}") from exc
