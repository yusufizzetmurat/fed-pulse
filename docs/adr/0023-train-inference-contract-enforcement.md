# ADR 0023 — Train / inference contract enforcement

Status: accepted, in production (as of merge).
Date: 2026-05-27.
References:
- Issue #341.
- ADR 0016 (forecaster research / serving split) — #336.
- Issue #339 (encoder usage + train/inference parity audit) — produced `docs/feature-provenance-audit.md`.
- `backend/app/training/inference_contract.py` — sidecar dataclass + derive / read / write / validate helpers.
- `backend/app/training/checkpoint.py::_save_model_checkpoint` — sidecar write site.
- `scripts/promote_checkpoint.py::promote_research_checkpoint_to_serving` — sidecar write on promotion.
- `backend/app/services/forecaster.py::_get_model` — sidecar validation on serving load.
- `backend/app/services/forecaster.py::_validate_serving_contract` — hard-vs-soft failure dispatch.
- `backend/app/services/forecaster.py::build_market_reaction_panel` — structured-state surface.
- `backend/app/main.py::_safe_regime_classification` — structured-state surface + counter increment.
- `backend/app/main.py::health_check` — exposes the validation surface + counters.
- `backend/app/schemas.py::InferenceStatusSurface` — response-side structured-state schema.
- `backend/app/models/registry.yaml::encoders[*].inference_features`, `artefacts[*].inference_features` — declarative feature pins.
- `tests/properties/test_forward_parity.py` — numeric parity (`torch.allclose`, atol=1e-6) between research and serving forwards on a real canonical checkpoint or its toy fallback.
- `tests/properties/test_kwarg_superset.py` — signature-only sibling of the parity test; AST-walks the loop + serving call sites.
- `tests/unit/test_inference_contract.py` — sidecar round-trip + loader-refusal coverage.

## Context

ADR 0016 split the forecaster into research (`ForecasterResearchModel`) and serving (`ForecasterServingModel`) classes, and #336 added a promotion contract (`scripts/promote_checkpoint.py`) that hands the persisted state-dict from the research artefact into a serving-shape payload. The #339 audit confirmed that even with the class split in place, the two surfaces could drift in subtle ways:

1. **Silent kwarg drift.** Training-side loaders fed `text_embedding`, `credibility`, `chunks`, and friends through to `forward_multi_task`; serving call sites in `app.services.forecaster` populated the same kwargs from a different source (live cached payloads, zero defaults). A train-side kwarg the serving forecaster forgot to thread through silently degraded to zero — the model evaluated against a different feature distribution at inference than at training, and the only signal was a quiet regression in the regime card. The #339 audit explicitly closed three such instances (`credibility`, `text_embedding`, `text_embedding_missing`).
2. **Silent failure swallowing.** `_safe_regime_classification` and `build_market_reaction_panel` both wrapped their forward dispatch in a `try / except: return None`. A real failure — a `TypeError` from a kwarg the call site missed, a `RuntimeError` from `prepare_recurrent_input` rejecting an absent text embedding — degraded to `None` on the JSON response with no greppable signal. The operator could not tell "model deliberately mute" from "model crashed silently".
3. **No declarative pin between the deployed checkpoint and the published registry.** `registry.yaml` declared the encoders the bake-off used and the artefacts the inference container pulled, but the linkage between "this checkpoint requires text_embedding + credibility" and "this serving call site supplies text_embedding + credibility" lived only in code review. A future PR that dropped a feature from the serving call site would not be rejected at boot — it would just silently zero the input.

The issue spec for #341 enumerated five interlocking items that together make "the deployed model is the published model" automated rather than reviewer-enforced.

## Decision

Ship the five items as a single PR, each load-bearing on the next. The order below mirrors the test surface in the property tests so the contract is greppable end-to-end.

### 1. Forward-parity property test

`tests/properties/test_forward_parity.py` builds the research and serving forecasters from the SAME `ModelConfig` and loads the SAME `state_dict` into both classes. The acceptance is `torch.allclose(research(x), serving(x), atol=1e-6, rtol=0.0)` on three input shapes:

- a deterministic zero-mean Gaussian over the (1, 30, FEATURE_SIZE) lookback;
- the same shape under a non-trivial mean / scale shift (the prior-N cache case the runtime path actually feeds);
- the `forward_multi_task` dispatch, when the canonical checkpoint mounts classification mode (skipped on the regression-only toy fallback).

The fixture prefers `backend/models/forecaster_best.pt` when present (canonical CI / production path) and falls back to a deterministic toy state_dict otherwise — but the fallback is NOT a stub: the toy state_dict still feeds both the research class and the serving class, so the parity assertion has identical semantics on a fresh clone and on a production-shaped box. The `atol=1e-6` floor is tight enough to catch a real divergence (e.g. an extra detach + reshape on one path that quietly changes the numeric output) without flagging float32 rounding noise.

### 2. Structured error surfacing on `_safe_regime_classification` + `build_market_reaction_panel`

Both helpers previously wrapped their forward dispatch in `try / except: return None`. The replacement preserves the "never raise" invariant but surfaces one of three structured states:

- `status: "not_classification_mode"` — the active checkpoint emits no card by contract. Legitimate `None`-shaped payload (or, for the panel, a structured payload carrying only the `status` key) so the operator can distinguish "model deliberately mute" from "model crashed silently".
- `status: "inference_kwarg_missing"` — the forward path raised `TypeError`, typically because the call site populated a kwarg the checkpoint did not declare in its inference contract sidecar, or omitted one the model requires. The kwarg name is parsed out of the Python `TypeError` message (`forward_multi_task() missing 1 required keyword-only argument: 'text_embedding'`) so the operator gets the exact missing field without parsing logs.
- `status: "unexpected_exception"` — anything else. Carries `exception_class` (the class name) + `detail` (the message). `RuntimeError` from the text/chunks-path mounted-but-not-threaded case is a sub-class of this branch.

Each branch increments a module-level counter (`app.services.forecaster._contract_counters`) and emits a `WARNING`-level log line with the structured fields as `key=value` pairs. The counters are surfaced through `/health` so an operator can spot a stuck contract without parsing logs.

The /analyze response carries the structured surface in two places:

- `regime_classification` stays `RegimeClassificationCard | None` (the legacy contract). A degraded card now lands as `null` on this field with the structured payload split into…
- `regime_classification_status` — a new sibling field of type `InferenceStatusSurface | None`. Mutually exclusive with the card being populated: either the card lands, or this field carries the structured reason.

`build_market_reaction_panel`'s structured payloads are collapsed by the `/analyze/market` route handler back to the legacy empty-panel response so the schema-side `MarketReactionPanel` does not need to grow a status field, while the structured detail still hits the logs + counters at the service layer.

### 3. Per-checkpoint `<stem>.inference_contract.json` sidecar

`backend/app/training/inference_contract.py` carries the `InferenceContract` dataclass (schema version + model class + required / optional kwargs + inference features + encoder alias) and the `derive_contract`, `write_sidecar`, `read_sidecar`, and `validate_against_serving` helpers. Every checkpoint write path is wired to emit the sidecar next to the `.pt` file:

- `_save_model_checkpoint` (the training-loop call site) writes one on every save, deriving the required-kwarg set from the model's runtime gates (`_text_path_active`, `credibility_features`, `use_chunk_attention`, `use_llm_embeddings`).
- `promote_research_checkpoint_to_serving` writes one on every promotion — preferring the source-side sidecar when present, falling back to deriving from the freshly built serving instance.

The sidecar write is a soft step: a failure logs at `WARNING` and degrades to "no sidecar emitted" so the training run still succeeds, but the default is to emit one on every save. Pre-#341 checkpoints with no sidecar continue to load — they hit the `sidecar_absent` branch in the loader and degrade to a soft warning.

### 4. Loader / serving kwarg-superset unit test

`tests/properties/test_kwarg_superset.py` AST-walks `backend/app/training/loop.py` and `backend/app/services/forecaster.py` to extract every kwarg the two sides populate (both via the `kwargs["name"] = ...` indirection and via direct `forward_multi_task(x, name=...)` calls), then asserts that every train-side kwarg is in the serving forward's signature. A drift in either direction trips the test before the artefact reaches CI's contract job. Includes a sibling assertion that the `SERVING_FORWARD_KWARGS` constant in `inference_contract.py` matches the live `ForecasterServingModel.forward` signature, so a downstream PR that adds a kwarg to the serving class but forgets to update the constant fails fast.

The point of the AST walk (rather than a pure runtime check) is that the test exercises both forward methods (`forward` and `forward_multi_task`) without instantiating the model, so it runs cheaply even on a fresh clone with no torch available — and the union-set semantics catch the "training-side adds, serving-side never threads" bug class even when the training kwarg is gated on a model flag that is off in the test fixture.

### 5. `registry.yaml::inference_features:` block

Every encoder + artefact entry in `backend/app/models/registry.yaml` now carries an `inference_features:` list declaring the kwargs the encoder / artefact contributes to a serving forecaster. Three population conventions:

- Encoders that feed pooled text vectors into the serving forward (`finbert_fed_adjacent`, `finbert_fed_adjacent_xbank_dapt`, the multi-axis classifier siblings, etc.) carry `[text_embedding, text_embedding_missing]`.
- Artefacts that bind the full serving forecaster (`forecaster_canonical`, `rates_heads_canonical`) carry `[text_embedding, text_embedding_missing, credibility]` — the canonical kwarg set the serving call site populates.
- Bake-off siblings and placeholder rows that never reach the serving path carry `[]` so the field is required-by-schema everywhere but the registry stays honest about what is and is not wired into inference.

The serving loader (`_validate_serving_contract` in `app.services.forecaster`) consults `encoder_ref(contract.encoder_alias).inference_features` and asserts the contract's `inference_features` are a subset. A registry that drops a feature mid-flight refuses to bind a checkpoint trained against the old declaration — the failure mode is `registry_inference_features_mismatch` and lands on `/health` rather than 5xx-ing the request path.

## Failure-mode dispatch

Two failure surfaces, two semantics:

- **Soft (legacy compatibility).** A checkpoint with no sidecar (`sidecar_absent`) loads normally. The legacy serving fleet from before this ADR continues to bind. `/health` exposes `inference_contract.status: "sidecar_absent"` so the operator can audit the fleet for unmigrated artefacts.
- **Hard (post-#341 contract).** A checkpoint with a sidecar that declares kwargs the serving signature does not accept (`serving_signature_missing_kwargs`) or features the registry does not pin (`registry_inference_features_mismatch`) refuses to bind. `_get_model` raises `RuntimeError`, `_model` stays `None`, and the next /analyze request retries the cold load (and surfaces the same error on `/health`). The intent: a known-incompatible artefact is a fast-fail signal, not a quiet degradation.

## Consequences

- The contract validation runs at LOAD time, not lazily on first request. A checkpoint that survives `_get_model()` is guaranteed to be kwarg-compatible with the serving signature; subsequent /analyze calls cannot trip the contract failure mode.
- The /health endpoint becomes the durable record of contract status. Counters reset on process restart by design — the structured logs alongside the increment are the persistent record.
- `MarketReactionPanel` keeps its existing schema; the structured status surface for market-reaction lives in the service-side logs + `/health` counters rather than on the panel itself. The /analyze response gains `regime_classification_status` as a new optional sibling field so the openapi snapshot rebases by one field.
- Per-checkpoint sidecar files (`<stem>.inference_contract.json`) become an audit artefact alongside the existing `.conformal.json` manifests. The rollout for existing fleet checkpoints is "they emit on next write" — a one-shot migration script is not strictly required because the soft `sidecar_absent` branch keeps the legacy fleet binding. Promotion + retrain are the natural backfill paths.
- Multi-axis classifier (`backend/app/data/train_text_multi_axis_classifier.py`) and trajectory bundle (`backend/app/trajectory/model.py`) write their own `torch.save` payloads under their own classes — they are NOT the serving forecaster, so they do not bind through `ForecasterServingModel`. Sidecar emission for those subsystems is out of scope for #341 and tracked as a follow-up if either subsystem grows a similar serving / research split.
- `_save_model_checkpoint` grew two keyword-only arguments (`encoder_alias`, `inference_features`) so callers can thread the registry context into the sidecar. Existing callers pass through unchanged (the new args default to `None` / `()`).
- The structured error surface on `_safe_regime_classification` is a behaviour change for existing tests: a previously-`None` return on the regression-only / no-manifest paths is now a structured `{"status": "not_classification_mode"}` payload. The /analyze response handler splits this off into the new `regime_classification_status` field so the `RegimeClassificationCard` schema is unchanged. Two unit tests in `tests/unit/test_regime_classification_response_block.py` and one in `tests/unit/test_rates_heads_endpoint.py` updated to match.
