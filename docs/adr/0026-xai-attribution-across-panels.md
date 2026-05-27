# ADR 0026 — XAI attribution across panels (integrated gradients)

Status: accepted, in production (as of merge).
Date: 2026-05-27.
References:
- Issue #297. The §16 SHOULD line item that never shipped.
- Issue #216 (existing XAI surface — keyword salience for the stance panel). The new attribution layer composes with #216, it does not replace it.
- Issue #341 / ADR 0023 (train / inference contract enforcement). The standing rule that no raw exception detail leaves the /analyze response applies verbatim: every IG entry point that degrades surfaces a structured `unavailable` payload with a reason enum, never the underlying `str(exc)`.
- Issue #336 / ADR 0016 (forecaster research / serving split). The IG runner backprops against `ForecasterServingModel.forward_multi_task`; the research class is not on the request path.
- `backend/app/services/xai_attribution.py` — the integrated-gradients kernel + per-panel runners.
- `backend/app/services/forecaster.py::build_panel_attributions` — the wire-up that fans out across active panels on every `include_xai=true` request.
- `backend/app/main.py::_build_analyze_response` — the dispatch point.
- `backend/app/schemas.py::XaiResponse.panels` — the response surface.
- `frontend/components/analyze/XaiPanel.tsx::PanelAttributionRow` — the per-panel bar chart + "explanation unavailable" badge.
- `tests/unit/test_xai_panel_attribution.py` — IG kernel coverage (zero baseline ⇒ zero attribution, linear model ⇒ attribution lands on the responsible feature).
- `tests/unit/test_xai_panels_in_analyze_response.py` — wire-up coverage (panel dicts parse cleanly into the schema; structured degrade on regression-mode checkpoints; helper never raises).
- `frontend/tests/components/analyze/XaiPanel.test.tsx` — Vitest coverage for the panel bar render + unavailable badge.

## Context

The dashboard already ships a sentence-level attribution surface on the stance panel under #216, implemented in `app/evaluation/xai.py` as a keyword-weighted salience pass over the Loughran-McDonald hawkish/dovish vocabulary. That surface is fine for the headline "explain my stance" use case — it is cheap (one regex pass over the document), it lines up with the lexicon the user mentally consults when reading FOMC text, and it does not require a forward pass.

The §16 roadmap line item #297 carved out a wider scope: an XAI layer that covers the **non-stance** panels — vol-regime classification, rates heads, trajectory — and explains why the model landed on a specific number (a regime label, a bps prediction, a next-stance class) rather than the document-level hawkish/dovish read. The keyword salience does not cover this. Those panels read off a multi-feature input vector (market + credibility + linguistic + MP-surprise + multi-axis + realised-vol + cross-asset + LLM-feature families) and the keyword pass tells the user nothing about which family drove the prediction.

The gap is the only §16 SHOULD that never shipped through the rest of the roadmap; #297 closes it.

## Decision

Integrated gradients (Sundararajan et al. 2017) over the per-bar feature tensor, with the per-feature attribution bucketed into the named feature families that already exist as `RICH_*_SLICE` constants on `app.models.config`. One IG run per active panel.

### Why integrated gradients over the alternatives

* **SHAP (KernelSHAP / DeepSHAP).** The "right" choice on paper for tabular features. Rejected at the per-request latency budget: KernelSHAP needs `2 * n_features` model evaluations per sample with random masking on top, which on a `T=30 × F=80` input is ~5000 forward passes — two orders of magnitude over the budget. DeepSHAP needs a separately trained baseline distribution; we have no shipping pipeline for that.
* **LIME.** Same latency problem (1000+ random perturbations per sample). LIME's local linear surrogate is also a poor fit for the multi-task head — the surrogate's coefficients are not directly comparable to the IG attributions, so the per-family bar chart would need a separate calibration story.
* **Attention rollout.** Free if the model uses attention, but attention weights are a poor proxy for feature attribution on tabular inputs (Jain & Wallace 2019). The vol-regime classifier has no attention head against the per-feature axis to begin with.
* **Integrated gradients.** Chosen. Two forward + two backward passes per integration step; bounded total cost at `n_steps × n_panels` model evaluations. Per-feature gradients aggregate cleanly into per-family magnitudes against the existing slice constants. No surrogate model, no baseline distribution to train.

### Attribution targets

One target per panel, picked so the IG run explains the scalar the panel actually renders on the UI:

| Panel       | Target                                  | Forward call            |
| ----------- | --------------------------------------- | ----------------------- |
| `regime`    | argmax-class stance logit (scalar)      | `forward_multi_task`    |
| `rates_2y`  | `rates_2y_bps` regression scalar        | `forward_multi_task`    |
| `rates_5y`  | `rates_5y_bps` regression scalar        | `forward_multi_task`    |
| `rates_terminal` | `rates_terminal_bps` regression scalar | `forward_multi_task` |

Trajectory is deferred — see non-goals.

The argmax index is resolved once on the clean input under `torch.no_grad` so the integration tracks a stable target across alpha steps. Targeting a stance class index that flipped mid-integration would emit attribution against a moving target and the per-family bars would be uninterpretable.

### Baseline choice — zero baseline

Zero baseline. Justification: the rich feature vectors are normalised through a `RobustScaler` fitted on the training set (`apply_rich_feature_scaler_tensor` on `app.training.loaders`); the trained scaler centres most of the input around zero and the L1 attribution magnitudes from a zero baseline are interpretable as "how far did this feature push the prediction away from its centred-input neutral point".

The alternatives:
* **Per-feature mean baseline.** Equivalent to the zero baseline after the scaler — the scaler is fit to put the mean near zero — and requires loading the scaler params just to compute the baseline. Rejected on complexity grounds; it pays nothing the zero baseline does not already deliver.
* **Random perturbation baseline.** Standard for KernelSHAP but defeats one of IG's core properties (the *completeness* axiom: per-feature attributions sum to the prediction delta from baseline to input). Drops the per-family interpretability the bar chart relies on.

The zero baseline is the natural anchor for "what would the model say with a centred / neutral input"; the IG completeness axiom then guarantees the per-family bars sum (in signed form) to the delta between the model's prediction at zero input and at the actual input.

### Sentence-level attribution (text path)

The pooled-text-embedding path stays under the existing keyword-salience surface (`app/evaluation/xai.py::attribute_text`). Gradients through the frozen FinBERT encoder are impractical at request time — even a single backward pass through the encoder is ~50× the IG step cost on the feature path, and the gradient signal is dominated by tokenizer-level effects that are hard to map back to a sentence-level highlight without a token-to-sentence alignment pass we do not currently have. The keyword pass remains the right tool for the sentence-highlight UI; this ADR adds the feature-family layer on top, it does not replace the existing surface.

### Step count + compute bound

`n_steps=20` default. Resolved via `resolve_n_steps` with precedence: explicit keyword > `FED_PULSE_IG_N_STEPS` env var > default. Hard-clamped into `[2, MAX_N_STEPS=64]` inside the helper so a misconfigured deployment cannot blow the latency budget.

Per-panel cost: `n_steps` forward + `n_steps` backward passes on the multi-task head. Observed P95 latency added to `/analyze` on the dev box (M1 MacBook Pro, no GPU) for the canonical checkpoint with two active panels (regime + one rates head) at `n_steps=20`: ~1.4 s. The five-second budget for `/analyze` is preserved with margin.

The frontend toggle is opt-in: the default `/analyze` payload (no `include_xai`) does not carry attribution. The toggle reuses the existing `include_xai` field on `AnalyzeRequest` — no new request field, no new endpoint. The panel attribution is layered onto the existing sentence-level surface on the same request flag so a single UI toggle delivers both.

### Failure handling

Every IG entry point degrades through structured branches, never raises:

* `not_classification_mode` — regime panel on a regression-output checkpoint.
* `head_not_mounted` — rates panel for a head the active checkpoint does not carry.
* `no_multi_task_forward` — model is the legacy serving class without a multi-task forward.
* `inference_kwarg_missing` — `TypeError` from the forward (kwarg drift; #341 contract failure surface).
* `ig_runtime_error` / `unexpected_exception` — any other failure inside the IG kernel.
* `bundle_not_loaded` / `missing_logits` — trajectory-specific degradations.

Each branch logs at WARNING with `exception_class=…` (no `str(exc)` on the client-facing payload — the #341 standing rule) and returns a `PanelAttribution` with `unavailable=true` + the reason enum. The frontend renders the "Explanation unavailable" badge instead of an empty bar chart. The /analyze response stays valid against the schema; an IG failure in any panel cannot 500 the request.

`build_panel_attributions` itself is also wrapped in a `try/except` at the dispatch point in `_build_analyze_response` — even a non-degradation-handled failure (an `Exception` the per-panel runner did not catch) is logged at WARNING and the `panels` block is omitted from the response. The /analyze flow has no failure mode where the IG path can break a successful prediction.

## Consequences

**What the user gets.** A per-panel feature-family bar chart on the XAI surface when `include_xai=true`. Bars are sorted in the canonical family order (market → credibility → linguistic → mp_surprise → multi_axis → realised_vol → cross_asset → llm); the longest bar in each panel anchors at 100%. Sign of the contribution is encoded by colour (hawkish hue for positive contribution to the target, dovish for negative). Panels that are not active on the active checkpoint render the "Explanation unavailable" badge alongside a short reason string.

**What the operator gets.** A structured-reason audit trail: every IG degradation logs at WARNING with the reason enum and the exception class. The same reason enum surfaces on the response so an operator can grep the JSON for the structured branch without parsing logs.

**Latency cost.** Bounded: `n_steps × n_active_panels` forward + backward passes per `include_xai=true` request. The default-OFF design ensures the cost is only paid when the user explicitly opts into the explanation surface. Plain `/analyze` requests pay nothing.

**Schema impact.** `XaiResponse.panels` is a new optional field defaulting to an empty list. The OpenAPI snapshot has been regenerated; no breaking field renames or removals. Clients on the pre-#297 schema continue to function — they simply ignore the new field.

## Punts / explicit non-goals

* **Trajectory panel attribution.** `attribute_trajectory_panel` is implemented and tested in isolation but the live route does not dispatch through it — the trajectory singleton lives in a sibling service (`app.services.trajectory`) with its own bundle + input contract, and wiring the dispatch end-to-end is out of scope for this PR. The kernel stays on disk so the follow-up only needs to add the dispatch + an integration test. Filed for #297 follow-up.
* **SHAP value calibration story.** The IG magnitudes are not directly comparable across panels (each panel's target has a different scale: stance logits ~ O(1), bps predictions ~ O(10²)). The frontend normalises each panel chart independently; a cross-panel normalisation would need a calibration pass we do not have time to ship under #297. The per-panel bars are interpretable in isolation, which is the use case the surface supports.
* **Gradient-through-encoder text attribution.** As above; the keyword salience continues to drive the sentence-highlight surface. A future ADR may revisit if the trained encoder shrinks enough to make a per-request backward pass feasible.
