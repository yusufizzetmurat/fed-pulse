# ADR 0025 — Inference observability

#341 (ADR 0023) shipped the inference-contract sidecar and the loader's hard-fail on serving-signature mismatch. The contract is greppable in test, in CI, and on `/health`. Three production-side gaps stayed open.

Cold-start bypassed contract validation: `_bootstrap_cold_start` calls `bootstrap_checkpoint` → `train_model`, then `_set_singleton_after_train` populates `_model` directly without crossing `_validate_serving_contract`. A bootstrap on a box whose registry / serving signature diverged from the training-time contract would silently bind an incompatible checkpoint. The loud-fail behaviour the #341 loader gives `/analyze` did not extend to cold-start.

`/settings/checkpoints` surfaced filename, role, encoder alias, and conformal-sidecar presence, but not the inference-contract sidecar. An operator could not tell which checkpoints carried the post-#341 contract and which were legacy, and could not see whether a present-but-mismatched sidecar was the reason a checkpoint refused to bind on `/health`.

Per-request kwarg drift was invisible. The sidecar declares the kwargs the serving forward consumes; the call site populates from the model's runtime gates (`_text_path_active`, `credibility_features`, `use_chunk_attention`, `use_llm_embeddings`). The two should agree, but the only signal of drift was a `TypeError` / `RuntimeError` from the forward, which #341 already wraps in structured surfaces. Neither gives the operator the request-rate distribution over the kwargs that are getting populated in production traffic.

Three observability surfaces close the gaps. Each is independent; the ordering below mirrors `tests/unit/test_inference_observability.py`.

## Cold-start parity with the canonical loader

`_bootstrap_cold_start` now invokes `_get_model` after `bootstrap_checkpoint` returns. The flow: train + write checkpoint + sidecar (existing); `_set_singleton_after_train` populates `_model` (existing, kept so the in-process singleton stays fresh); then drop `_model` + `_model_artifact_metadata` under `_model_lock`, and call `_get_model()` to force a cold load through `_validate_serving_contract`.

A `RuntimeError` from the contract validation propagates out of `_bootstrap_cold_start` (and out of the `/analyze` 500 surface) rather than being silently swallowed. `/health` continues to expose the structured reason via `get_serving_contract_status`. The redundant cold load is a one-time cost on the bootstrap path; the singleton stays cached for every subsequent request. The bootstrap path is now a loud-fail surface: a box whose registry / serving signature drifted from the training contract surfaces the mismatch on the first `/analyze`, not after silent degradation in production.

## Settings-page UI surface for the inference contract

`SettingsCheckpoint` (the response-schema row) grows three fields:

- `required_kwargs: list[str]` — copied from the sidecar's `required_kwargs`; empty list when no sidecar.
- `supplied_at_inference: dict[str, bool]` — each declared kwarg mapped to a bool. The truth set is `collect_serving_forward_kwargs(ForecasterServingModel)`. `True` means the live forward signature accepts the kwarg; `False` means the sidecar declares a kwarg the live forward will reject.
- `inference_contract_status: str | None` — `"present"` when the sidecar exists, `"sidecar_absent"` for a pre-#341 artefact, `None` for non-forecaster rows.

The endpoint resolves the serving-kwarg set once per request (not per row) and degrades to the static `SERVING_FORWARD_KWARGS` constant if model-class introspection fails, so an installation with a torch-import failure at the settings page still gets a usable inventory.

`CheckpointRow` in `frontend/pages/settings.tsx` renders three cases:

- `sidecar_absent`: single neutral "legacy" `Badge`, no per-kwarg badges. Pre-#341 fleet status; deliberately quiet so an operator can scan for unmigrated checkpoints without alarm.
- `present` + `required_kwargs.length === 0`: single neutral "no required kwargs" badge. Contract present but the checkpoint runs the legacy 6-feature regression path; nothing to mark red.
- `present` + per-kwarg badges: each declared kwarg as a `Badge`. Variant is `outline` (neutral) when supplied, `hawkish` (red Fed-context variant) when not. The mismatch surface is identical whether the kwarg is missing because the serving class never grew the parameter or because the sidecar declares an unknown name.

`frontend/tests/pages/settings.test.tsx` exercises three shapes (sidecar + unknown kwarg → red; sidecar absent → legacy; sidecar + no kwargs → neutral). Backend test asserts the JSON shape on a synthetic mismatch checkpoint.

## Per-request structured kwarg log

`forecast_quantitative_series` emits one INFO line per request via `_log_serving_forward_kwargs`:

```
analyze_serving_forward kwargs=<comma-separated> checkpoint=<stem>
```

The kwargs list mirrors the gates in `_predict_next_point`: `credibility` when `model.credibility_features` is on; `text_embedding` + `text_embedding_missing` when `model._text_path_active`; `chunks` + `elapsed_days` when `model.use_chunk_attention` or `model.use_llm_embeddings`. The list is empty for the legacy regression-only path. One line per `/analyze`, not per forecast step and not per kwarg; operators run `grep analyze_serving_forward | awk` to see the request-rate distribution over kwarg-sets, and per-request is the right granularity for that. The line lives at INFO so a logger configured at WARNING drops it without tuning; sites that want the drift signal lift to INFO.

## Failure-mode shifts

| Surface | Pre-#342 | Post-#342 |
| --- | --- | --- |
| Cold-start with mismatched sidecar | silently binds via `_set_singleton_after_train`; `/health` shows `inference_contract.status: "ok"` despite the drift | `RuntimeError` propagates out of `_bootstrap_cold_start`; `/analyze` 500s; `/health` reports `serving_signature_missing_kwargs` |
| Settings page on a mismatched forecaster | filename + active flag + conformal sidecar, no contract signal | per-kwarg red `Badge` per unsupplied kwarg; greppable by `data-testid="contract-kwarg-missing-<name>"` |
| Production traffic with text path silently disabled | no signal; only a quiet regression in card output | every `/analyze` emits `analyze_serving_forward kwargs= checkpoint=...`; `grep \| wc -l` against the expected-kwarg cohort surfaces the drift |

## Notes on cost and edges

The cold-start path is slower by one extra cold load (sub-millisecond on top of a multi-second training step). Settings carries three additional fields per row; both directions degrade cleanly under pydantic's extra-field defaults. The per-request log is unconditional but small, under 100 bytes typical, cardinality bounded by the six gate combinations under the current matrix.

`_log_serving_forward_kwargs` inspects model gates, not the kwargs the call site actually populates at the moment of the forward. The two should agree (the gates are what `_predict_next_point` switches on); a future drift between gates and call site stays invisible in this log. That class is already covered by `tests/properties/test_kwarg_superset.py` from #341; the log is request-rate evidence, not a parity check.

The red-badge surface fires only on the forecaster role. Multi-axis classifier, LoRA, and calibration checkpoints do not bind through `ForecasterServingModel`, so they carry no #341 sidecar; the frontend gates the section on `checkpoint.role === "forecaster"`. The cold-start re-validation reuses `_model_lock`, so a racing first `/analyze` blocks on the lock until the bootstrap finishes; no double-load.

The intended trade-off: if the bootstrap-trained sidecar declares kwargs the serving signature does not accept (e.g. the training script ran against a serving class that has since had a kwarg renamed), cold-start now fails loud instead of binding silently. The operator rolls the registry forward or the serving class back, and the failure is visible on `/health` immediately rather than degrading silently downstream.

## References

- `backend/app/main.py::_bootstrap_cold_start`, `list_settings_checkpoints`
- `backend/app/services/forecaster.py::_log_serving_forward_kwargs`
- `backend/app/schemas.py::SettingsCheckpoint`
- `frontend/pages/settings.tsx::CheckpointRow`
- ADR 0023 (#341), ADR 0016 (#336)
