# Clean-room late-fusion rebuild — design note

Date: 2026-06-01

A text+market late-fusion forecaster is being rebuilt from scratch, independent of the existing `MultiModalForecasterModel`, to determine whether the earlier "text is null" results were a genuine null or the product of an implementation or data error. The rebuild itself is the test: each fault class is verified against a dedicated gate.

## Motivation

Prior event-based work concluded that FOMC text predicts neither direction nor magnitude of the intraday reaction (n=110). That null is suspected to be contaminated by one or more of:

1. Text–event misalignment — wrong timestamps or window, text matched to the wrong event, or 14:00 ET cutoff mishandled.
2. Wrong encoder silently used — the known bug where the default sentiment model 404s on HF and silently falls back to `distilbert-sst-2`, replacing FinBERT-fed with a generic encoder.
3. Training or architecture bug — the `n_classes>=2` + MSE-on-unit-0 magnitude hack, fusion wiring where text gradients do not flow, or optimisation killing the text branch.
4. Over-aggressive leak controls — per-fold PCA, residualisation, or embargo strict enough to regress real text signal out before the model sees it.

## Scope

Two evaluation frames, both with a direction head (BCE) and a magnitude head (regression):

- Event frame (n=110): FOMC statement-day announcement-window reaction (immediate 14:00→14:30 ET and delayed 14:30→15:00 ET), features strictly from the pre-announcement window. The exact setup under suspicion.
- Daily companion (~2000): each typed Fed communication's daily reaction (close→next-close), full corpus. A higher-power frame to settle text definitively where n=110 cannot.

## Data

- Corpus `data/external/fed_comms/fed_communications.parquet` — 2,000 typed docs 2006→2026 (`doc_type`: speech / testimony / statement / minutes / press_conference), columns `text, date, timestamp_et, time_known, speaker, title, url`.
- Intraday SPY/SPX — FOMC-day windows already in `…/tp_v3_full_rebuild_2026_05_30/intraday_events.parquet` (n=110, both heads' targets present); daily realised measures in `data/external/alphavantage_bars/spx_5min_daily_rv.parquet`.
- SEP (net-new): scrape `https://www.federalreserve.gov/monetarypolicy/fomcprojtabl{YYYYMMDD}.htm` (date = meeting decision day; pattern verified 2012 / 2015 / 2024). Parse structured projection tables: median + central-tendency + range for real GDP, unemployment, PCE, core-PCE, and the appropriate fed-funds-rate path (dot-plot median / central-tendency). Key by meeting date. Leak rule: the projection tables and dot plot are released at 2pm ET with the statement, so they are usable as an event-frame feature; the narrative SEP document ships about three weeks later with the minutes, so it is only usable in the daily frame aligned to its actual (minutes) release date, never the event frame.

## Architecture — late fusion

- Text encoder branch: frozen `finbert-fed-adjacent` embeddings (default), pooled over the event's text bundle (statement + recent speeches / minutes / testimony in the prior weeks + SEP structured features when available). A small MLP projects the pooled embedding to a text latent.
- Market encoder branch: sequence encoder (GRU) over pre-announcement market history and realised measures → market latent.
- Late fusion: concatenate text latent ⊕ market latent at the penultimate layer (no earlier mixing).
- Heads: two clean heads off the fused latent — `direction` (1 logit, BCE) and `magnitude` (1 unit, MSE / Huber on |return| or window RV). No `n_classes>=2` constraint.

## Fault-verification gates (one per fault class)

1. Alignment audit (fault #1): event table rebuilt from scratch; emit a human-readable audit (date → doc title → window returns); hard-assert pre-window strictly < 14:00 ET < reaction window and text↔event correctness.
2. Encoder assertion (fault #2): load `finbert-fed-adjacent` explicitly; hard-fail if resolved model id or embedding dim mismatch; log an embedding fingerprint. No silent fallback.
3. Gradient-flow assertion (fault #3): after one backward pass, assert text-branch parameters receive non-zero gradients; clean separate heads.
4. Leak-control ablation (fault #4): train with and without per-fold PCA / residualisation; report both, so the effect of leak-correction on signal is measured, not assumed.

## Evaluation

Walk-forward with embargo ≥ horizon, seed-swept. Baselines: majority (direction), market-only, HAR (magnitude / vol). Moving-block bootstrap CIs (block = horizon) on the text-vs-baseline gain. Report direction accuracy + CI and magnitude OOS-R² + CI on both frames, with and without leak controls.

## Environment

All runs in `fed-pulse-backend-gpu:latest` (`PYTHONPATH=/app`, `FED_PULSE_DATA_DIR=/data`, `FED_PULSE_SENTIMENT_MODEL=yusufizzetmurat/finbert-fed-adjacent`). Ruff 0.5.0 + mypy strict + pytest.

## Build sequence

- SEP ingestion — `sep_ingestion.py` → `data/external/fed_comms/sep.parquet`, with tests.
- Clean-room data assembly — both frames, plus the alignment audit, with tests.
- Encoder load — hard assertions + corpus / SEP embeddings.
- Late-fusion model — both heads, gradient-flow check, with tests.
- Train + eval — walk-forward, baselines, bootstrap, leak-control ablation, and a verdict on each fault class.
