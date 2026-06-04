# ADR 0023 — Train / inference contract enforcement

ADR 0016 split the forecaster into research (`ForecasterResearchModel`) and serving (`ForecasterServingModel`) classes; #336 added a promotion contract handing the persisted state-dict from the research artefact into a serving-shape payload. The #339 audit (`docs/feature-provenance-audit.md`) confirmed that even with the split in place, the two surfaces drifted in three ways the codebase couldn't catch without reviewer eyes.

Silent kwarg drift: training-side loaders fed `text_embedding`, `credibility`, `chunks`, and friends through to `forward_multi_task`; serving call sites populated the same kwargs from a different source (live cached payloads, zero defaults). A train-side kwarg the serving forecaster forgot to thread through silently degraded to zero. The model then evaluated against a different feature distribution at inference than at training, and the only signal was a quiet regression in the regime card. #339 closed three such instances (`credibility`, `text_embedding`, `text_embedding_missing`).

Silent failure swallowing: `_safe_regime_classification` and `build_market_reaction_panel` wrapped forward dispatch in `try / except: return None`. A real failure (a `TypeError` from a missed kwarg, a `RuntimeError` from `prepare_recurrent_input` rejecting an absent text embedding) degraded to `None` on the JSON response with no greppable signal. An operator couldn't tell "model deliberately mute" from "model crashed silently".

No declarative pin between the deployed checkpoint and the published registry. `registry.yaml` declared the encoders the bake-off used and the artefacts the inference container pulled, but the linkage between "this checkpoint requires text_embedding + credibility" and "this serving call site supplies them" lived only in code review. A future PR that dropped a feature from the serving call site wouldn't be rejected at boot; it would silently zero the input.

#341 ships five interlocking items so "the deployed model is the published model" is automated rather than reviewer-enforced. The ordering below mirrors the property-test surface so the contract is greppable end-to-end.

## Forward-parity property test

`tests/properties/test_forward_parity.py` builds research and serving forecasters from the SAME `ModelConfig` and loads the SAME `state_dict` into both. The acceptance is `torch.allclose(research(x), serving(x), atol=1e-6, rtol=0.0)` on three input shapes: a deterministic zero-mean Gaussian over `(1, 30, FEATURE_SIZE)`; the same shape under a non-trivial mean / scale shift (the prior-N cache case the runtime path feeds); the `forward_multi_task` dispatch when the canonical checkpoint mounts classification mode (skipped on regression-only toy fallback).

The fixture prefers `backend/models/forecaster_best.pt` when present (canonical CI / inference path) and falls back to a deterministic toy state_dict otherwise. The fallback is not a stub: the toy state_dict still feeds both classes, so the parity assertion has identical semantics on a fresh clone and on a canonical-shaped box. `atol=1e-6` is tight enough to catch a real divergence (an extra detach + reshape on one path that quietly changes the numeric output) without flagging float32 rounding noise.

## Structured error surfacing on the helpers

`_safe_regime_classification` and `build_market_reaction_panel` keep their "never raise" invariant but surface one of three structured states:

- `not_classification_mode` — the active checkpoint emits no card by contract. Legitimate `None`-shaped payload (or, for the panel, a structured payload carrying only the `status` key) so the operator can distinguish deliberate mute from silent crash.
- `inference_kwarg_missing` — the forward raised `TypeError`, typically because the call site populated a kwarg the sidecar didn't declare, or omitted one the model requires. The kwarg name is parsed out of the Python `TypeError` message (`forward_multi_task() missing 1 required keyword-only argument: 'text_embedding'`) so the operator gets the exact missing field without parsing logs.
- `unexpected_exception` — anything else. Carries `exception_class` + `detail`. `RuntimeError` from the text/chunks-path mounted-but-not-threaded case is a sub-class of this branch.

Each branch increments a module-level counter (`app.services.forecaster._contract_counters`) and emits a WARNING-level log line with structured key=value fields. The counters surface through `/health` so an operator can spot a stuck contract without log-grepping.

The /analyze response carries the structured surface in two places. `regime_classification` stays `RegimeClassificationCard | None` (legacy contract). A degraded card lands as `null` and the structured payload splits into a new sibling `regime_classification_status` (`InferenceStatusSurface | None`), mutually exclusive with the card being populated. `build_market_reaction_panel`'s structured payloads collapse at the `/analyze/market` handler back to the legacy empty-panel response so the `MarketReactionPanel` schema doesn't grow a status field; the structured detail still hits the service-side logs + counters.

## Per-checkpoint sidecar

`backend/app/training/inference_contract.py` carries the `InferenceContract` dataclass (schema version + model class + required / optional kwargs + inference features + encoder alias) and the `derive_contract` / `write_sidecar` / `read_sidecar` / `validate_against_serving` helpers. Every checkpoint write path emits a `<stem>.inference_contract.json` next to the `.pt`:

- `_save_model_checkpoint` writes one on every save, deriving the required-kwarg set from the model's runtime gates (`_text_path_active`, `credibility_features`, `use_chunk_attention`, `use_llm_embeddings`).
- `promote_research_checkpoint_to_serving` writes one on every promotion, preferring the source-side sidecar when present and falling back to deriving from the freshly built serving instance.

The write is a soft step: failure logs at WARNING and degrades to "no sidecar emitted" so the training run still succeeds. Pre-#341 checkpoints without a sidecar continue to load; they hit the `sidecar_absent` branch in the loader and degrade to a soft warning.

## Loader / serving kwarg-superset test

`tests/properties/test_kwarg_superset.py` AST-walks `loop.py` and `forecaster.py` to extract every kwarg the two sides populate (both `kwargs["name"] = ...` indirection and direct `forward_multi_task(x, name=...)` calls), then asserts every train-side kwarg is in the serving forward's signature. Drift in either direction trips the test before the artefact reaches CI's contract job. A sibling assertion pins `SERVING_FORWARD_KWARGS` to the live `ForecasterServingModel.forward` signature so a downstream PR that adds a kwarg to the serving class but forgets to update the constant fails fast.

AST walk over runtime check, because the test exercises both forwards (`forward` and `forward_multi_task`) without instantiating the model. It runs cheaply on a fresh clone with no torch, and the union-set semantics catch the "training-side adds, serving-side never threads" bug class even when the kwarg is gated on a flag that's off in the test fixture.

## `registry.yaml::inference_features`

Every encoder and artefact entry in `backend/app/models/registry.yaml` now carries an `inference_features:` list declaring the kwargs the encoder / artefact contributes to a serving forecaster:

- Encoders that feed pooled text vectors into the serving forward (`finbert_fed_adjacent`, `finbert_fed_adjacent_xbank_dapt`, the multi-axis classifier siblings) carry `[text_embedding, text_embedding_missing]`.
- Artefacts that bind the full serving forecaster (`forecaster_canonical`, `rates_heads_canonical`) carry `[text_embedding, text_embedding_missing, credibility]`.
- Bake-off siblings and placeholder rows that never reach the serving path carry `[]` so the field is required-by-schema everywhere but the registry stays honest about what's actually wired.

The serving loader (`_validate_serving_contract`) consults `encoder_ref(contract.encoder_alias).inference_features` and asserts the contract's `inference_features` are a subset. A registry that drops a feature mid-flight refuses to bind a checkpoint trained against the old declaration; the failure mode is `registry_inference_features_mismatch` and lands on `/health` rather than 5xx-ing the request path.

## Failure-mode dispatch

Soft (legacy compatibility): a checkpoint with no sidecar (`sidecar_absent`) loads normally. The pre-#341 fleet continues to bind. `/health` exposes `inference_contract.status: "sidecar_absent"` so the operator can audit for unmigrated artefacts.

Hard (post-#341 contract): a checkpoint with a sidecar declaring kwargs the serving signature doesn't accept (`serving_signature_missing_kwargs`) or features the registry doesn't pin (`registry_inference_features_mismatch`) refuses to bind. `_get_model` raises `RuntimeError`, `_model` stays `None`, and the next /analyze request retries the cold load (surfacing the same error on `/health`). A known-incompatible artefact is a fast-fail signal, not a quiet degradation.

## Downstream effects

Contract validation runs at LOAD time, not lazily on first request; a checkpoint that survives `_get_model()` is guaranteed kwarg-compatible. `/health` becomes the durable record; counters reset on process restart by design (the structured logs are the persistent record).

`MarketReactionPanel` keeps its schema; the market-reaction status surface lives in service-side logs + `/health` counters. The /analyze response gains `regime_classification_status` as a new optional sibling field; the openapi snapshot rebases by one field. The per-checkpoint sidecar becomes an audit artefact alongside the existing `.conformal.json` manifests. Rollout for existing fleet checkpoints is "emit on next write"; no one-shot migration is required because the soft branch keeps the legacy fleet binding, and promotion + retrain are the natural backfill paths.

`_save_model_checkpoint` grew two keyword-only args (`encoder_alias`, `inference_features`) so callers can thread registry context into the sidecar; existing callers pass through unchanged (new args default to `None` / `()`). The structured error surface on `_safe_regime_classification` is a behaviour change for existing tests: a previously-`None` return on the regression-only / no-manifest paths is now a structured `{"status": "not_classification_mode"}` payload. The /analyze handler splits this off into the new `regime_classification_status` field, so the `RegimeClassificationCard` schema is unchanged; two unit tests in `test_regime_classification_response_block.py` and one in `test_rates_heads_endpoint.py` updated to match.

Multi-axis classifier (`train_text_multi_axis_classifier.py`) and trajectory bundle (`trajectory/model.py`) write their own `torch.save` payloads under their own classes; they are not the serving forecaster, so they do not bind through `ForecasterServingModel`. Sidecar emission for those subsystems shipped as #393.

## #393 — sidecars for the other two serving artefacts

The two non-forecaster artefacts the inference container binds (`TextMultiAxisClassifier` at `text_multi_axis_best.pt`; the trajectory bundle's `model.pt`) now ship the same sidecar shape. Each subsystem reuses the shared dataclass + helpers; the derivation entry points (`derive_multi_axis_contract`, `derive_trajectory_contract`) declare the kwarg set the respective serving call site populates:

- Multi-axis classifier: `input_ids`, `attention_mask` (required). The serving call site in `app.services.multi_axis_classifier.score_text` populates both from the HF tokeniser; a forward refactor that drops either is rejected at boot.
- Trajectory bundle: `inputs` (required), `mask` (optional). LSTM and Transformer arms share the same forward signature, so one contract covers both.

Save sites: `_save_checkpoint` in `train_text_multi_axis_classifier` and `app.trajectory.model.save_model`. Emission is wrapped in the same soft-fail try/except the forecaster uses. Load sites: `app.services.multi_axis_classifier._load_state` and `app.services.trajectory._load_state` each call a local `_validate_contract` that hard-refuses on signature mismatch (`RuntimeError` with a structured status string, no `str(exc)` leak per the #341 standing rule) and soft-degrades on `sidecar_absent`. Each service exposes `get_serving_contract_status()` mirroring the forecaster's surface so `/health` can grep the structured reason.

The wire format is identical; the extension is two more save / load sites bound to the same contract surface, no new ADR needed.

## References

- `backend/app/training/inference_contract.py`, `backend/app/training/checkpoint.py`
- `backend/app/services/forecaster.py::_validate_serving_contract`, `_get_model`
- `backend/app/main.py::health_check`, `_safe_regime_classification`
- `backend/app/models/registry.yaml` (`encoders[*].inference_features`, `artefacts[*].inference_features`)
- `tests/properties/test_forward_parity.py`, `test_kwarg_superset.py`; `tests/unit/test_inference_contract.py`
- ADR 0016 (#336), Issues #339, #341, #393
