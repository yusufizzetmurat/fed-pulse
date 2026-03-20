# GPU Setup Notes (RTX 4080)

## 1) NVIDIA Container Toolkit
Install and verify NVIDIA Container Toolkit so Docker can access GPU devices.

Quick check:
- `nvidia-smi` on host
- `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`

## 2) CUDA-compatible PyTorch
For CUDA 12.1 environments, use:
- `backend/requirements-gpu-cu121.txt`

This file installs `torch` from the CUDA 12.1 wheel index.

## 3) Data regeneration expectation
If `data/` is gitignored (recommended), new machines must regenerate local artifacts:
- Run scraper once to recreate `fomc_statements.*` and `fomc_minutes.*` under `data/`.

## 4) Cross-machine note
Keep `backend/requirements.txt` as portable default.
Use `backend/requirements-gpu-cu121.txt` only on CUDA-capable NVIDIA hosts.

## 5) Professional training entrypoint
Once your `data/` directory contains raw scraper output, prepare a trainable dataset first and then run the trainer.

Preparation examples:
- `cd backend && python -m app.prepare_training_data --symbols ^GSPC ^VIX DX-Y.NYB ^TNX BTC-USD`
- `docker compose exec backend python -m app.prepare_training_data --data-dir /data --symbols ^GSPC ^VIX DX-Y.NYB ^TNX BTC-USD`

The preparation step will:
- read `fomc_statements.json` and `fomc_minutes.json`
- run sentiment inference with CUDA when available
- fetch market snapshots per document and per symbol
- write grouped training data to `data/train_dataset.json`

Then run the trainer directly.

Local Python example:
- `cd backend && python -m app.train_forecaster --device cuda`

Docker example:
- `docker compose exec backend python -m app.train_forecaster --device cuda --data-dir /data`
- `docker compose exec backend python -m app.train_forecaster --device cuda --data-dir /data --list-data`
- `docker compose exec backend python -m app.train_forecaster --device cuda --data-dir /data --epochs 200 --hidden-size 128 --dropout 0.10`
- `docker compose exec backend python -m app.train_forecaster --device cuda --data-dir /data --sweep --hidden-sizes 64 96 128 --num-layers-grid 1 2 --dropouts 0.10 0.15 --learning-rates 0.001 0.0005 --epochs-grid 120 200`

The trainer will:
- detect and use CUDA when available
- read JSON, JSONL, and CSV datasets from the target data directory
- build sequence windows for training/validation
- save the best checkpoint to `backend/models/forecaster_best.pt`
- expose `hidden_size`, `num_layers`, `dropout`, and `head_hidden_size` as CLI knobs
- record validation metrics and selected architecture metadata inside the checkpoint payload

Sweep mode will also:
- evaluate the full search grid and rank trials by validation `combined_rmse`
- retrain the best configuration once to persist the final checkpoint
- write reports to `backend/models/forecaster_sweep_results.json` and `backend/models/forecaster_sweep_results.csv`

Expected trainable record fields:
- required: `date`, `close`, `volatility_5d`
- optional: `sentiment_score`
- passthrough metadata allowed: `symbol`, `document_type`, `title`, `source_file`

Accepted layouts:
- one flat record list in `records`, `rows`, `data`, or `items`
- grouped datasets in `sequences`, `series`, or `groups`
- CSV / JSONL files with one record per row

`data/train_dataset.json` is the intended output of the preparation step and is compatible with the current grouped training-data loader.

## 6) Docker test path
To run the backend suite inside the same isolated environment used by the app:
- `docker compose run --rm backend pytest tests/`
- `docker compose run --rm backend pytest tests/unit/test_forecaster.py tests/unit/test_train_forecaster.py tests/unit/test_main_api.py`

The Compose backend service now mounts the repo-level `tests/` directory and sets `PYTHONPATH=/app`, so this command works without needing local `pytest`.

## 7) Frontend evaluation checklist
After training finishes and `backend/models/forecaster_best.pt` exists:
- restart or reload the backend so the latest checkpoint metadata is picked up
- run a historical analysis with realized overlay enabled
- inspect `Model Evaluation Snapshot` in the frontend

That panel now shows:
- whether the saved checkpoint was actually loaded
- which LSTM architecture is active
- whether the current market close sits inside the projected confidence band
- whether the latest realized close stayed inside the forecast band on historical runs
