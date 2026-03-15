# System Design: Multi-Modal MVP

## Objective
Provide a single API workflow that combines text sentiment with market indicators and returns a short-horizon volatility estimate for demo use.

## End-to-End Flow
1. Frontend sends `POST /analyze` with `text`, `date`, and optional `symbol`.
2. Backend `sentiment.py` runs transformer inference and returns normalized sentiment output.
3. Backend `market_data.py` fetches yfinance close series, applies trading-day look-back fallback, and computes a 5-day volatility proxy.
4. Backend `forecaster.py` builds a last-5 sequence feature tensor and runs an LSTM forward pass.
5. API responds with `{ sentiment, prediction, market }`.

## Feature Construction
- **Sentiment features**
  - `score` from sentiment model output (0-1 scale).
- **Market features**
  - `close` price (normalized in forecaster by `/10000`).
  - `volatility_5d` from rolling standard deviation of percentage returns.
- **Sequence policy**
  - Uses up to last 5 trading points.
  - Pads with the earliest available point if fewer than 5 exist.

## Forecaster Assumptions (MVP)
- Model is a skeleton, not a trained production model.
- Architecture:
  - LSTM (`input_size=3`, `hidden_size=16`, `num_layers=1`)
  - MLP head with sigmoid scalar output
- Inference behavior:
  - singleton model instance
  - `eval()` + `torch.no_grad()`
- Output:
  - `prediction.volatility` in `[0, 1]` range for stable demo display
  - `prediction.horizon` currently `3d`

## Error Handling and Fallbacks
- Invalid date format returns `422`.
- Non-trading day requests use nearest prior trading day within look-back window.
- If no valid trading point exists in the look-back range, API returns a `500` pipeline error.
- Sentiment model uses fallback transformer when target model is unavailable.

## Service Modules
- `backend/app/services/sentiment.py`
- `backend/app/services/market_data.py`
- `backend/app/services/forecaster.py`
- Orchestration in `backend/app/main.py`
