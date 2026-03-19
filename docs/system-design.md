# System Design: Hybrid Quant Forecasting MVP

## Objective
Provide a single API workflow that combines text sentiment with market indicators and returns short-horizon forecasts for both close price and volatility.

## End-to-End Flow
1. Frontend sends `POST /analyze` with `text`, `date`, optional `symbol`, `forecast_mode`, `horizon`, and optional `include_realized`.
2. Backend `sentiment.py` runs transformer inference and returns normalized sentiment output.
3. Backend `market_data.py` fetches yfinance close history, applies trading-day look-back fallback, and computes a rolling volatility proxy.
4. Backend `forecaster.py` builds sequence tensors and selects mode:
   - `fast`: deterministic singleton LSTM forward path
   - `quick_train`: lightweight local training on recent history before forecasting
5. API responds with `{ sentiment, prediction, market, series }`, including optional realized-forward overlay arrays when requested.

## Feature Construction
- **Sentiment features**
  - `score` from sentiment model output (0-1 scale).
- **Market features**
  - `close` price (normalized in forecaster by `/10000`).
  - `volatility_5d` from rolling standard deviation of percentage returns.
- **Sequence policy**
  - Forecaster windows use last 5 points.
  - Historical chart context uses up to last 30 points.
  - Pads with earliest point when sequence length is insufficient.

## Forecaster Assumptions (MVP)
- Model remains MVP-grade (not production-calibrated).
- Architecture:
  - LSTM (`input_size=3`, `hidden_size=16`, `num_layers=1`)
  - MLP head with 2 outputs (`close`, `volatility`)
- Inference behavior:
  - singleton model instance
  - `eval()` + `torch.no_grad()`
  - optional quick-train branch with bounded epochs
- Output:
  - scalar summary in `prediction.{close, volatility, horizon}`
  - `prediction.horizon` currently `3d`
  - chart-ready arrays under `series.*` with optional realized overlays and volatility-axis suggestions

## Series Metadata
- `series.timestamps`, `series.history_close`, `series.history_volatility`
- `series.forecast_timestamps`, `series.forecast_close`, `series.forecast_close_lower`, `series.forecast_close_upper`
- `series.forecast_volatility`, `series.forecast_volatility_lower`, `series.forecast_volatility_upper`
- `series.forecast_confidence_level`
- optional when `include_realized=true` and data is available:
  - `series.realized_timestamps`, `series.realized_close`, `series.realized_volatility`
- chart scaling aid:
  - `series.volatility_scale.{suggested_ymin,suggested_ymax}`

## Error Handling and Fallbacks
- Invalid date format returns `422`.
- Non-trading day requests use nearest prior trading day within look-back window.
- If no valid trading point exists in the look-back range, API returns a `500` pipeline error.
- Sentiment model uses fallback transformer when target model is unavailable.
- `forecast_mode` validation enforces `fast` or `quick_train`.
- `include_realized=true` is safe for past-date analyses; empty realized arrays are returned when future observations are unavailable.

## Service Modules
- `backend/app/services/sentiment.py`
- `backend/app/services/market_data.py`
- `backend/app/services/forecaster.py`
- `backend/app/train_forecaster.py`
- Orchestration in `backend/app/main.py`
