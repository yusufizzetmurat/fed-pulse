# Architecture Overview

## Product Flow
1. User submits `POST /analyze` with `text`, `date`, `symbol`, `forecast_mode`, `horizon`, and optional `include_realized`.
2. NLP service generates normalized sentiment scores.
3. Market service fetches history and applies trading-day fallback if needed.
4. Forecast service builds feature vectors and runs:
   - `fast`: static inference
   - `quick_train`: bounded adaptation then inference
5. API returns sentiment, prediction, market context, model metadata, and chart-ready series.

## Backend Pipeline
1. `scraper.py` collects statements/minutes.
2. `prepare_training_data.py` creates market-aligned training records.
3. `train_forecaster.py` trains from discovered datasets or hyperparameter sweep.
4. Capability-first data modules orchestrate ingestion, normalization, quality validation, package build, and experiment preparation.

## Feature Set (Current)
- sentiment score
- market close and rolling 5-day volatility
- `close_change_pct`
- `volatility_change`
- sequence windows for model input and chart context

## Runtime Notes
- LSTM backbone with dual outputs (`close`, `volatility`) at present.
- Inference uses `eval()` + `torch.no_grad()`.
- CUDA is used when available.
- Runtime metadata includes checkpoint status and adaptation diagnostics.
- Docker stack is CPU-safe by default, with optional GPU profile support.

## API and Frontend Contract

Request fields:
- `text: string`
- `date: YYYY-MM-DD`
- `symbol?: string`
- `forecast_mode?: "fast" | "quick_train"`
- `horizon?: "1d" | "3d" | "5d" | "10d"`
- `include_realized?: boolean`

Response groups:
- `sentiment`
- `prediction`
- `market`
- `model`
- `series`

## UI Behavior
- States: idle, loading, success, error.
- One page handles form state, request lifecycle, and chart mapping.
- Rendering is defensive against partial response payloads.
- Realized overlays and evaluation metrics are date-aligned (timestamp intersection), not index-aligned.

## Error and Fallback Behavior
- invalid date or mode -> `422`
- missing market data in lookback -> `500`
- sentiment model has fallback path if preferred model fails to load
- data scripts skip invalid rows rather than aborting full run

## Main Code Areas
- `backend/app/main.py`
- `backend/app/services/sentiment.py`
- `backend/app/services/market_data.py`
- `backend/app/services/forecaster.py`
- `backend/app/data/source_ingestion.py`
- `backend/app/data/label_normalization.py`
- `backend/app/data/quality_validation.py`
- `backend/app/data/training_package_builder.py`
- `backend/app/data/pipeline_data_prep.py`
- `backend/app/data/phase3_training_execution.py`
- `frontend/pages/index.js`
- `frontend/styles/globals.css`
# Architecture Overview

## Product Flow
1. User submits `POST /analyze` with `text`, `date`, `symbol`, `forecast_mode`, `horizon`, and optional `include_realized`.
2. NLP service generates normalized sentiment scores.
3. Market service fetches history and applies trading-day fallback if needed.
4. Forecast service builds feature vectors and runs:
   - `fast`: static inference
   - `quick_train`: bounded adaptation then inference
5. API returns sentiment, prediction, market context, model metadata, and chart-ready series.

## Backend Pipeline
1. `scraper.py` collects statements/minutes.
2. `prepare_training_data.py` creates market-aligned training records.
3. `train_forecaster.py` trains from discovered datasets or hyperparameter sweep.
4. Capability-first data engineering modules under `backend/app/data` orchestrate ingestion, normalization, validation, packaging, and baseline execution.

## Feature Set (Current)
- sentiment score
- market close and rolling 5-day volatility
- `close_change_pct`
- `volatility_change`
- sequence windows for model input and chart context

## Runtime Notes
- LSTM backbone with dual outputs (`close`, `volatility`) at present.
- Inference uses `eval()` + `torch.no_grad()`.
- CUDA used when available.
- Runtime metadata includes checkpoint status and adaptation diagnostics.
- Docker stack is CPU-safe by default; optional GPU service is available via compose profile.

## API/Frontend Contract

Request fields:
- `text: string`
- `date: YYYY-MM-DD`
- `symbol?: string`
- `forecast_mode?: "fast" | "quick_train"`
- `horizon?: "1d" | "3d" | "5d" | "10d"`
- `include_realized?: boolean`

Response groups:
- `sentiment`
- `prediction`
- `market`
- `model`
- `series`

## UI Behavior
- States: idle, loading, success, error
- Single page handles form state, request lifecycle, and chart mapping
- Rendering is defensive against partial response payloads

## Error/Fallback Behavior
- invalid date or mode -> `422`
- missing market data in lookback -> `500`
- sentiment model has fallback path if preferred model fails to load
- data scripts skip invalid rows instead of aborting entire run

## Main Code Areas
- `backend/app/main.py`
- `backend/app/services/sentiment.py`
- `backend/app/services/market_data.py`
- `backend/app/services/forecaster.py`
- `backend/app/data/source_ingestion.py`
- `backend/app/data/label_normalization.py`
- `backend/app/data/quality_validation.py`
- `backend/app/data/training_package_builder.py`
- `backend/app/data/pipeline_data_prep.py`
- `backend/app/data/phase3_training_execution.py`
- `backend/app/services/scraper.py`
- `backend/app/prepare_training_data.py`
- `backend/app/train_forecaster.py`
- `frontend/pages/index.js`
- `frontend/styles/globals.css`
