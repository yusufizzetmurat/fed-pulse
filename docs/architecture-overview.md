# Architecture Overview

## Request Flow
1. Client sends `POST /analyze` with `text`, `date`, `symbol`, `forecast_mode`, and `horizon`
2. Sentiment service scores the text
3. Market service fetches history and forward trading dates
4. Forecaster returns close/volatility predictions + confidence bands
5. API returns chart-ready payload with model diagnostics

For `forecast_mode=real_train`, `/analyze` starts an async job and returns `job_id`.  
Client polls `GET /train-jobs/{job_id}` until completion.

## Runtime Modes
- `fast`: checkpoint inference
- `quick_train`: short adaptation before inference
- `real_train`: async training with 252-day history and checkpoint update

## Core Feature Set
- sentiment score
- market close
- rolling 5-day volatility
- close change percent
- volatility change

## API Contract
Request fields:
- `text`, `date`
- `symbol` (optional)
- `forecast_mode` (`fast` | `quick_train` | `real_train`)
- `horizon` (`1d` | `3d` | `5d` | `10d`)
- `include_realized` (optional)

Response groups:
- `sentiment`
- `prediction`
- `market`
- `model`
- `series`

## Frontend Notes
- One-page dashboard for request + chart rendering
- Supports realized overlay
- Error metrics and overlays are timestamp-aligned
- Real Train includes async status UI (queued/running/failed/succeeded)

## Main Code Areas
- `backend/app/main.py`
- `backend/app/services/sentiment.py`
- `backend/app/services/market_data.py`
- `backend/app/services/forecaster.py`
- `backend/app/data/`
- `frontend/pages/index.js`
