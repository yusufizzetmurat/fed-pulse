# ADR 0025 — Inference observability

Status: accepted, in production (as of merge).
Date: 2026-05-27.
References:
- Issue #342.
- ADR 0023 (train / inference contract enforcement) — #341. Establishes the `<stem>.inference_contract.json` sidecar surface this ADR makes operationally visible.
- ADR 0016 (forecaster research / serving split) — #336. Provides the `ForecasterServingModel` whose forward signature this ADR introspects on every settings response.
- `backend/app/main.py::_bootstrap_cold_start` — resets the singleton + invokes `_get_model` after writing the bootstrap checkpoint.
- `backend/app/main.py::list_settings_checkpoints` — extended response with `required_kwargs` + `supplied_at_inference` + `inference_contract_status` per row.
- `backend/app/services/forecaster.py::_log_serving_forward_kwargs` — per-request structured INFO line.
- `backend/app/schemas.py::SettingsCheckpoint` — schema extension.
- `frontend/pages/settings.tsx::CheckpointRow` — red / neutral / legacy badge surface.
- `tests/unit/test_inference_observability.py` — backend coverage for all three surfaces.
- `frontend/tests/pages/settings.test.tsx` — Vitest coverage for the badge rendering on synthetic checkpoints.

## Context

#341 landed the inference contract sidecar + the loader's hard-fail on a serving-signature mismatch. The contract surface is greppable in test, in CI, and on `/health` — but three production-side gaps remained:

1. **Cold-start bypasses contract validation.** `_bootstrap_cold_start` calls `bootstrap_checkpoint` → `train_model`, which writes the checkpoint and then calls `_set_singleton_after_train` to populate `_model` directly. The post-train singleton populates without crossing `_validate_serving_contract`. A bootstrap run on a box whose registry / serving signature diverged from the training-time contract would silently bind an incompatible checkpoint — the loud-fail behaviour the #341 loader gives `/analyze` did not extend to the cold-start path.
2. **`/settings/checkpoints` does not expose the contract.** The endpoint surfaced filename / role / encoder alias / conformal-sidecar presence but not the inference-contract sidecar. An operator scanning the settings page could not tell which checkpoints carried the post-#341 contract and which were legacy, and could not see at a glance whether a present-but-mismatched sidecar was the reason a checkpoint refused to bind on `/health`.
3. **Per-request kwarg drift is invisible.** The sidecar declares the kwargs the serving forward consumes; the serving call site populates a kwarg set based on the model's runtime gates (`_text_path_active`, `credibility_features`, `use_chunk_attention`, `use_llm_embeddings`). The two should agree by construction, but the only signal of drift was a `TypeError` or `RuntimeError` from the forward — both of which #341 already wraps in structured surfaces, but neither of which gives the operator the request-rate distribution over the kwargs that *are* getting populated in production traffic.

The three gaps share a shape: the contract from #341 is correct in code and tested in CI, but it is not *observable* in a running deployment beyond `/health` + the counters.

## Decision

Three observability surfaces. Each is independent; the ordering below mirrors the test surface in `tests/unit/test_inference_observability.py`.

### 1. Cold-start parity with the canonical loader

`_bootstrap_cold_start` now invokes `_get_model` after `bootstrap_checkpoint` returns. The flow:

1. Train + write checkpoint + sidecar (existing).
2. `_set_singleton_after_train` populates `_model` (existing — kept so the in-process singleton is fresh).
3. **New.** Drop `_model` + `_model_artifact_metadata` under `_model_lock`, then call `_get_model()` to force a cold load through `_validate_serving_contract`.

A `RuntimeError` from the contract validation propagates out of `_bootstrap_cold_start` (and therefore out of the `/analyze` 500 response surface) rather than being silently swallowed. `/health` continues to expose the structured reason via `get_serving_contract_status` exactly as for the standard cold-load path. The redundant cold load is a one-time cost on the bootstrap path — the singleton stays cached for every subsequent request.

The intent: the bootstrap path is now a *loud-fail* surface. A box whose registry / serving signature has drifted from the training contract surfaces the mismatch on the first `/analyze`, not after silent degradation in production.

### 2. `/settings/checkpoints` UI surface for the inference contract

`SettingsCheckpoint` (the response-schema row) gains three fields:

- `required_kwargs: list[str]` — copied from the sidecar's `required_kwargs`. Empty list when no sidecar.
- `supplied_at_inference: dict[str, bool]` — each declared kwarg mapped to a boolean. The truth set is `collect_serving_forward_kwargs(ForecasterServingModel)`; a `True` entry means the live serving forward signature accepts the kwarg, a `False` entry means the sidecar declares a kwarg the live forward will reject.
- `inference_contract_status: str | None` — `"present"` when the sidecar exists, `"sidecar_absent"` for a pre-#341 legacy artefact, `None` for non-forecaster rows.

The endpoint resolves the serving-kwarg set once per request (not per row) and degrades to the static `SERVING_FORWARD_KWARGS` constant if the model-class introspection fails — so an installation with a torch-import failure at the settings-page level still gets a usable inventory.

Frontend rendering (`CheckpointRow` in `frontend/pages/settings.tsx`):

- `inference_contract_status === "sidecar_absent"`: single neutral "legacy" `Badge` next to the row, no per-kwarg badges. This is the pre-#341 fleet status; the row is deliberately quiet so an operator can scan the inventory for unmigrated checkpoints without alarm.
- `inference_contract_status === "present"` + `required_kwargs.length === 0`: single neutral "no required kwargs" badge. The contract is present but the checkpoint runs on the legacy 6-feature regression path; nothing to mark red.
- `inference_contract_status === "present"` + per-kwarg badges. Each declared kwarg renders as a `Badge`; the variant is `outline` (neutral) when `supplied_at_inference[name] === true`, and `hawkish` (the red Fed-context variant) when `supplied_at_inference[name] === false`. The mismatch surface is identical whether the kwarg is missing because the serving class never grew the parameter or because the sidecar declares an unknown name.

Test surface: `frontend/tests/pages/settings.test.tsx` exercises three checkpoint shapes (sidecar + unknown kwarg → red; sidecar absent → legacy; sidecar + no kwargs → neutral). Backend test in `tests/unit/test_inference_observability.py` asserts the JSON shape on a synthetic mismatch checkpoint.

### 3. Per-request structured kwarg log line

`forecast_quantitative_series` emits exactly one INFO line per request via the new `_log_serving_forward_kwargs` helper:

```
analyze_serving_forward kwargs=<comma-separated-list> checkpoint=<stem>
```

The kwargs list mirrors the gates in `_predict_next_point` (the canonical serving forward call site):

- `credibility` when `model.credibility_features` is on.
- `text_embedding` + `text_embedding_missing` when `model._text_path_active` is on.
- `chunks` + `elapsed_days` when `model.use_chunk_attention` or `model.use_llm_embeddings` is on.

The list is empty (`kwargs= checkpoint=...`) for the legacy regression-only path. One line per `/analyze` request, not one per forecast step and not one per kwarg — operators run `grep analyze_serving_forward | awk` to see the request-rate distribution over kwarg-sets in production traffic, and the per-request resolution is the right granularity for that.

The log line lives at INFO level so a production logger configured at WARNING (or higher) drops the line without further tuning; sites that want the drift signal lift the logger to INFO.

## Failure-mode dispatch

| Surface | Pre-#342 behaviour | Post-#342 behaviour |
| --- | --- | --- |
| Cold-start with contract-mismatched sidecar | silently binds via `_set_singleton_after_train`; `/health` would show `inference_contract.status: "ok"` despite the contract drift | `RuntimeError` propagates out of `_bootstrap_cold_start`; `/analyze` 500s; `/health` reports the structured `serving_signature_missing_kwargs` reason |
| Settings page on a forecaster with mismatched sidecar | shows filename + active flag + conformal sidecar status — no signal of the contract mismatch | per-kwarg red `Badge` for every unsupplied kwarg; row is greppable by `data-testid="contract-kwarg-missing-<name>"` |
| Production traffic with the text path silently disabled | no signal; the operator only sees a quiet regression in card output | every `/analyze` request emits `analyze_serving_forward kwargs= checkpoint=...` (no text kwargs); a `grep | wc -l` against the expected-kwarg cohort surfaces the drift |

## Consequences

- The cold-start path is now strictly slower by one extra cold load (the `_get_model` re-validation after the bootstrap train). On a fresh-clone bootstrap this is sub-millisecond overhead on top of the multi-second training step — not material.
- The settings page response carries three additional fields per row. The frontend `SettingsCheckpoint` interface keeps them optional so a pre-#342 backend behind a post-#342 frontend renders without the contract block; the badge UI degrades to "no kwargs visible" rather than crashing. The reverse (post-#342 backend + pre-#342 frontend) ignores the extra fields by pydantic's default extra-field behaviour on the client.
- The per-request log line is unconditional. A production logger at INFO sees one extra structured line per `/analyze` request. The line is small (under 100 bytes typical) and the cardinality is bounded by the number of distinct kwarg-sets the active checkpoint can populate (six combinations max under the current gate matrix). No tuning required.
- The `_log_serving_forward_kwargs` helper does NOT inspect the kwargs the call site actually populates at the moment of the forward — it inspects the model gates. The two should agree by construction (the gates are what `_predict_next_point` switches on); a future drift between the gates and the call site would not be visible in this log. That drift class is already covered by the `tests/properties/test_kwarg_superset.py` AST walk from #341 — the log line is request-rate evidence, not a parity check.
- The settings-page red-badge surface only fires on the forecaster role. Multi-axis classifier + LoRA + calibration checkpoints are not bound through `ForecasterServingModel` and therefore do not carry the #341 sidecar; the row renders without the contract block (the frontend gates the section on `checkpoint.role === "forecaster"`).
- The cold-start re-validation reuses the same `_model_lock` the `_get_model` cold load holds. A racing first `/analyze` that arrives while the bootstrap is still running blocks on the lock until the bootstrap finishes; the post-bootstrap re-validation then runs once and the racing request picks up the cached singleton on lock release. No double-load.
- One inverted-cost surface: if the bootstrap-trained sidecar declares kwargs the serving signature does not accept (e.g. the training script ran against a serving class that has since had a kwarg renamed), the cold-start now fails-loud instead of binding-silently. The operator must roll the registry forward or roll the serving class back — but the failure mode is now visible on `/health` immediately rather than degrading silently in the inference path. This is the intended trade-off.
