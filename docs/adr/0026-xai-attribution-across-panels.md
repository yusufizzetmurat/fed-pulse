# ADR 0026 — XAI attribution across panels (integrated gradients)

The dashboard already ships a sentence-level attribution surface on the stance panel under #216 — a keyword-weighted salience pass over the Loughran-McDonald hawkish/dovish vocabulary in `app/evaluation/xai.py`. That surface is fine for the headline "explain my stance" use case: it's cheap (one regex pass), it lines up with the lexicon the user mentally consults when reading FOMC text, and it doesn't need a forward pass.

The §16 SHOULD line item #297 carved out a wider scope: an XAI layer covering the non-stance panels — vol-regime classification, rates heads, trajectory — that explains why the model landed on a specific number (a regime label, a bps prediction, a next-stance class) rather than the document-level hawkish/dovish read. The keyword salience doesn't cover any of this; those panels read off a multi-feature input vector (market + credibility + linguistic + MP-surprise + multi-axis + realised-vol + cross-asset + LLM-feature families) and the keyword pass tells the user nothing about which family drove the prediction. This was the only §16 SHOULD that never shipped through the rest of the roadmap.

## What this is

Integrated gradients (Sundararajan et al. 2017) over the per-bar feature tensor, with per-feature attribution bucketed into the named feature families that already exist as `RICH_*_SLICE` constants on `app.models.config`. One IG run per active panel.

SHAP would be the textbook choice for tabular features but won't fit the per-request latency budget — KernelSHAP needs `2 * n_features` model evaluations per sample with random masking on top, which on a `T=30 × F=80` input is ~5000 forward passes, two orders past the budget. DeepSHAP needs a separately trained baseline distribution we don't have a pipeline for. LIME has the same problem (1000+ random perturbations per sample) and its local linear surrogate is a poor fit for the multi-task head — surrogate coefficients aren't directly comparable to IG attributions, so the per-family bar chart would need its own calibration story. Attention rollout is free if the model uses attention, but attention weights are a poor proxy for feature attribution on tabular inputs (Jain & Wallace 2019), and the vol-regime classifier has no attention head against the per-feature axis. IG gives two forward + two backward passes per integration step, bounded total cost at `n_steps × n_panels` evaluations, and per-feature gradients aggregate cleanly into per-family magnitudes against the existing slice constants.

## Attribution targets

One target per panel, picked so the IG run explains the scalar the panel actually renders:

| Panel        | Target                                       | Forward call                            |
| ------------ | -------------------------------------------- | --------------------------------------- |
| `regime`     | argmax-class stance logit (scalar)           | `forward_multi_task`                    |
| `rates_2y`   | `rates_2y_bps` regression scalar             | `forward_multi_task`                    |
| `rates_5y`   | `rates_5y_bps` regression scalar             | `forward_multi_task`                    |
| `rates_terminal` | `rates_terminal_bps` regression scalar   | `forward_multi_task`                    |
| `trajectory` | argmax next-stance logit (scalar)            | trajectory bundle `model(inputs, mask)` |

The trajectory dispatch reads the runtime singleton (`app.services.trajectory.get_state`), reuses `build_trajectory_inputs` to assemble the `(B, T, embedding_dim + market_dim)` input + boolean mask, then runs `attribute_trajectory_panel`. A missing bundle, an empty strict-backward history window, or an invalid `as_of_date` degrade into structured `unavailable` payloads (`bundle_not_loaded`, `trajectory_history_empty`, `invalid_as_of_date`) so the trajectory entry is always present on the panel list, even when the model couldn't be exercised.

The argmax index is resolved once on the clean input under `torch.no_grad` so the integration tracks a stable target across alpha steps. Targeting a stance class index that flipped mid-integration would emit attribution against a moving target and the per-family bars would be uninterpretable.

## Zero baseline

The rich feature vectors are normalised through a `RobustScaler` fitted on the training set (`apply_rich_feature_scaler_tensor`). The trained scaler centres most of the input around zero, and the L1 attribution magnitudes from a zero baseline are interpretable as "how far did this feature push the prediction away from its centred-input neutral point." Per-feature mean baseline is equivalent after the scaler — the scaler is fit to put the mean near zero — and requires loading the scaler params just to compute the baseline; rejected on complexity. Random perturbation baseline (standard for KernelSHAP) defeats IG's completeness axiom (per-feature attributions sum to the prediction delta from baseline to input), which is what the per-family bar chart relies on.

## Sentence-level attribution (text path)

The pooled-text-embedding path stays under the existing keyword-salience surface. Gradients through the frozen FinBERT encoder are impractical at request time — even a single backward pass through the encoder is ~50× the IG step cost on the feature path, and the gradient signal is dominated by tokenizer-level effects that are hard to map back to a sentence-level highlight without a token-to-sentence alignment pass we don't currently have. The keyword pass remains the right tool for the sentence-highlight UI; this ADR adds the feature-family layer on top.

## Step count and compute bound

`n_steps=20` default, resolved via `resolve_n_steps` with precedence `explicit > FED_PULSE_IG_N_STEPS env > default`, hard-clamped into `[2, MAX_N_STEPS=64]` inside the helper so a misconfigured deployment can't blow the latency budget. Per-panel cost: `n_steps` forward + `n_steps` backward passes on the multi-task head. Observed P95 latency added to `/analyze` on the dev box (M1 MacBook Pro, no GPU) for the canonical checkpoint with two active panels at `n_steps=20`: ~1.4 s. The five-second budget for `/analyze` is preserved with margin.

The frontend toggle is opt-in: the default `/analyze` payload (no `include_xai`) does not carry attribution. The toggle reuses the existing `include_xai` field on `AnalyzeRequest` — no new request field, no new endpoint. Panel attribution layers onto the existing sentence-level surface on the same request flag.

## Failure handling

Every IG entry point degrades through structured branches, never raises:

- `not_classification_mode` — regime panel on a regression-output checkpoint.
- `head_not_mounted` — rates panel for a head the active checkpoint does not carry.
- `no_multi_task_forward` — model is the legacy serving class without a multi-task forward.
- `inference_kwarg_missing` — `TypeError` from the forward (kwarg drift; #341 contract failure).
- `ig_runtime_error` / `unexpected_exception` — anything else inside the IG kernel.
- `bundle_not_loaded` / `missing_logits` — trajectory-specific degradations.

Each branch logs at WARNING with `exception_class=…` (no `str(exc)` on the client-facing payload — the #341 standing rule) and returns a `PanelAttribution` with `unavailable=true` + the reason enum. The frontend renders an "Explanation unavailable" badge instead of an empty bar chart. The /analyze response stays valid against the schema; an IG failure in any panel cannot 500 the request.

`build_panel_attributions` is also wrapped in a try/except at the dispatch point in `_build_analyze_response` — even a non-degradation-handled failure is logged at WARNING and the `panels` block is omitted. The /analyze flow has no failure mode where the IG path can break a successful prediction.

## What the surface delivers

The user gets a per-panel feature-family bar chart on the XAI surface when `include_xai=true`. Bars sort in the canonical family order (market → credibility → linguistic → mp_surprise → multi_axis → realised_vol → cross_asset → llm); the longest bar in each panel anchors at 100%. Sign of contribution is encoded by colour (hawkish hue for positive contribution to the target, dovish for negative). Panels not active on the active checkpoint render the unavailable badge alongside the reason string. The operator gets a structured-reason audit trail: every degradation logs at WARNING with the reason enum and the exception class; the same reason surfaces on the response so a JSON grep finds the structured branch without parsing logs.

Latency cost is bounded: `n_steps × n_active_panels` forward + backward per `include_xai=true` request. Default-off ensures cost is only paid when the user opts in. `XaiResponse.panels` is a new optional field defaulting to `[]`; the OpenAPI snapshot regenerated with no breaking renames or removals. Pre-#297 clients ignore the new field.

## Explicit non-goals

Cross-panel normalisation. IG magnitudes are not directly comparable across panels — each target has a different scale (stance logits ~ O(1), bps predictions ~ O(10²)). The frontend normalises each panel chart independently; a cross-panel calibration would need a pass we don't have time to ship under #297. Per-panel bars are interpretable in isolation, which is the use case the surface supports.

Gradient-through-encoder text attribution. The keyword salience continues to drive the sentence-highlight surface. A future ADR may revisit if a smaller trained encoder makes a per-request backward pass feasible.

## References

- `backend/app/services/xai_attribution.py`, `forecaster.py::build_panel_attributions`
- `backend/app/main.py::_build_analyze_response`
- `backend/app/schemas.py::XaiResponse.panels`
- `frontend/components/analyze/XaiPanel.tsx::PanelAttributionRow`
- `tests/unit/test_xai_panel_attribution.py`, `test_xai_panels_in_analyze_response.py`
- Sundararajan, Taly, Yan (2017); Jain & Wallace (2019)
- Issues #216, #297; ADR 0023 (#341), ADR 0016 (#336)
