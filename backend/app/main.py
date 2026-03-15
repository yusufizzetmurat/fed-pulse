import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import AnalyzeRequest, AnalyzeResponse
from app.services.forecaster import FeatureVector, predict_volatility
from app.services.market_data import fetch_market_sequence, fetch_market_snapshot
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
        sentiment = analyze_text(payload.text)
        market = fetch_market_snapshot(target_date=payload.date, symbol=payload.symbol)
        market_sequence = fetch_market_sequence(target_date=payload.date, symbol=payload.symbol)

        sequence_vectors = [
            FeatureVector(
                sentiment_score=float(sentiment["score"]),
                market_close=float(point["close"]),
                market_volatility=float(point["volatility_5d"]),
            )
            for point in market_sequence
        ]
        prediction = predict_volatility(sequence_vectors, horizon=payload.horizon)

        return {"sentiment": sentiment, "prediction": prediction, "market": market}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Analyze pipeline failed: {exc}") from exc
