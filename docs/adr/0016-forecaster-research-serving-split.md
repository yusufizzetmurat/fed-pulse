# ADR 0016 — Forecaster research / serving split + promotion contract

Status: accepted, in production (as of merge).
Date: 2026-05-27.
Supersedes: the monolithic `ForecasterModel` class that pre-#336 served as both the research training entrypoint and the `/analyze` serving path.
References:
- Issue #336.
- `backend/app/models/forecaster_base.py` — shared backbone (`ForecasterBase`) + `prepare_recurrent_input` helper.
- `backend/app/models/research_model.py` — `ForecasterResearchModel` (all knobs).
- `backend/app/models/serving_model.py` — `ForecasterServingModel` (frozen surface).
- `backend/app/models/lstm.py` — back-compat shim re-exporting `ForecasterModel` as the research class.
- `backend/app/models/factory.py` — `build_forecaster(role=...)`, `build_research_forecaster`, `build_serving_forecaster`.
- `scripts/promote_checkpoint.py` — promotion utility + CLI entry point.
- `backend/app/services/forecaster.py` — imports `ForecasterServingModel`; singleton + `_set_singleton_after_train` build serving instances.
- `tests/regression/test_forecaster_canonical_determinism.py` — replacement determinism guard on the post-#322 canonical config.
- ADR 0015 (regression-canonical objective) — the canonical default the serving class carries.

## Context

Pre-#336 the codebase had one `ForecasterModel` class (712 lines) that hosted every research knob plus every serving back-compat path. The single class carried:

- eight architecture types (`lstm` / `lstm_attn` / `gru` / `tcn` / `transformer` / `dlinear` / `informer` / `tft`),
- three text-feature variants (time-decay / chunk attention / LLM embeddings),
- two embedding channels (`scalar` / `embeddings`),
- three head modes (`classification` / `regression` / `dual`),
- two output modes (`regression` / `classification`),
- the multi-task head, optional log-RV regression head, and three rates heads,
- LoRA toggles, InfoNCE alignment fields, credibility features, class-weight knobs.

The class duplicated ~60 lines of input-prep verbatim across `forward()` and `forward_multi_task()`. Each research-side change had to preserve the serving back-compat surface because both consumers shared the same class. The back-compat layer ballooned every time a research knob was added.

## Decision

Split the class. Three modules ship:

- **`ForecasterBase` (shared backbone).** Owns the recurrent core (eight architectures), the chunk / LLM pooler, the credibility broadcast, the pooled-text adapter, and the sequence pooling (last-step / mean / learned attention). The pre-#336 inline duplication collapses to one module-level helper, `prepare_recurrent_input(model, x, ...)`.
- **`ForecasterResearchModel` (research class).** Subclasses `ForecasterBase`; carries every research knob and both `forward()` + `forward_multi_task()` methods. The two forwards now thread their input-prep through `prepare_recurrent_input`. Training, sweep harness, and the InfoNCE multimodal companion stay research-only.
- **`ForecasterServingModel` (serving class).** Subclasses `ForecasterBase`; frozen ctor surface for `/analyze`. Defaults: `output_mode='regression'`, `head_mode='regression'` (per ADR 0015 / #322). Still supports `output_mode='classification'` because the regime classification card and the market-reaction panel both pull `log_rv` and the multi-task dict off `forward_multi_task` on the serving instance.

The state_dict key layout is identical across the two classes for the shared backbone + the per-head modules; this is the contract that makes the promotion path metadata-only.

**Promotion contract.** Research checkpoints are explicitly promoted before they hit the serving path:

```
python scripts/promote_checkpoint.py <research.pt> <serving.pt>
```

The script copies the state_dict into a serving-shape payload, bumps the `model_version` field with a `+serving` suffix, and stamps `serving_class = "ForecasterServingModel"`. The in-process `_set_singleton_after_train` does the equivalent in memory after a fresh training run so the live `/analyze` path picks up the new weights without a process restart.

**Back-compat.** `backend/app/models/lstm.py` survives as a thin shim that re-exports `ForecasterModel` as an alias for `ForecasterResearchModel`. Existing importers (tests, training loop, factory call sites) keep working through the deprecation window. Removal is scheduled in a follow-up issue once in-repo importers have been migrated to the explicit module names.

**Dead-code sweep (scoped).** Issue #265 retired the `quick_train` / `real_train` adaptation paths from the backend; this split sweeps the residual frontend references:

- `frontend/lib/analyze/types.ts`: the `ForecastMode` union narrows to `"fast"`; the optional `forecast_mode` field is removed from `AnalyzeRequest`. The `HistoryEntry.forecast_mode` column stays typed as `string` so persisted rows from before the sweep continue to render.
- `backend/app/main.py` already had no `_run_real_train_job` thread orchestration or `/train-jobs` endpoint (retired in #265); no further removal needed.
- `backend/app/db.py` keeps the `analysis_runs.forecast_mode` column — this is the DB migration the issue intentionally scopes as out of bounds.

The pre-#322 v1 byte-identity regression test is replaced by `tests/regression/test_forecaster_canonical_determinism.py`, which exercises the post-#322 canonical config (regression-canonical head, lstm core) and pins (a) loss reproducibility at seed=11, (b) research-vs-serving forward parity at identical seed, and (c) determinism of the classification regime-card forward_multi_task path.

## Consequences

- Research-side changes no longer have to preserve the serving back-compat surface; the serving class is frozen and any new research knob lands on `ForecasterResearchModel` only.
- Serving callers (`services.forecaster._get_model`, `_set_singleton_after_train`, the regime + market-reaction cards) import a narrower class with a frozen surface. New consumers must promote a research checkpoint before serving from it.
- The shared `prepare_recurrent_input` helper is the single source of truth for the input-prep sequence. Any change to that sequence — a new feature broadcast, a different ordering, an extra guard — lands in one place rather than three.
- The promotion contract is non-destructive: the source research checkpoint is left in place and the promoted artefact lands next to it. Subsequent `/analyze` traffic only loads the serving artefact.
- Multi-modal (gated-InfoNCE) checkpoints remain research-only; the factory rejects `role="serving"` for them. A serving-shape multimodal artefact is out of scope for this split and would require either a separate serving subclass or a fusion-stripped promotion path; documented as a follow-up.
- The back-compat shim in `lstm.py` will be removed in a follow-up issue after the next merge wave; the in-repo migration to explicit module names happens once the test surface settles.
