# ADR 0035 — Multi-target heads on the shared encoder

Issue #292 lands three live heads on a single encoder: the vol-regime classifier from #195 and the rates-complex regression heads on the 2y, 5y, and terminal-rate columns ingested in #291. The pre-#292 path mounted only the regime head; this work generalises the shared-backbone surface so a single training run produces a checkpoint that emits every active head off one forward, and the conformal sidecar carries one calibrated object per head.

The rates path is regression-primary. The events.parquet target columns are 5-day yield changes in basis points; the natural unit is the bps move, not a tertile bucket. Predicting bps directly avoids the information-destructive collapse the classification framing imposes on a continuous quantity (#322 made the parallel call on the vol-regime head; the rates argument is sharper because the tail magnitude matters for the read). The auxiliary 3-class direction head (easing / neutral / tightening) stays opt-in via `--rates-classification-heads`; default off mounts only the regression heads and the product surface emits the bps point plus the conformal interval with a null directional bucket.

## What lands

`backend/app/models/research_model.py` and `backend/app/models/serving_model.py` each mount a `ModuleDict` of per-head regression stacks (LayerNorm → Linear → GELU → Dropout → Linear-to-1) keyed on the head short name (`2y` / `5y` / `terminal`). The aux classifier `ModuleDict` is empty unless `rates_aux_classification=True`; the forward emits `rates_{name}_bps` for every active head and `rates_{name}_cls_logits` only when the paired classifier was mounted. The flag rides on `ModelConfig.rates_aux_classification` and is round-tripped through `_coerce_payload_config` so a saved checkpoint rehydrates with the same head topology the run trained.

The loss is the existing `MultiTaskLoss` plus a `_compute_rates_loss` helper that sums one MSE-or-joint term per active rates head. Each per-head term reads a per-fold standardiser fitted on the train slice (`fit_rates_scaler`) so the rates branch contributes a unit-variance MSE comparable in magnitude to the regime CE term. When more than one rates head is active the helper averages across the heads, otherwise three mounted heads would triple the rates branch's gradient share relative to a single-head run. The CLI surface exposes `--targets regime,rates_2y,rates_terminal` as the canonical comma-separated knob (the legacy `--rates-heads` alias still resolves to the same internal tuple), plus `--rates-classification-heads` (opt-in mount of the aux classifier) and `--rates-head-mode` (regression / classification / dual selector for the joint loss). The dual / classification modes fail-fast at the factory when the aux flag is off because the CE term has no head to land on otherwise.

## Per-head conformal calibration

Per-head residual distributions differ in shape and scale. The 2y reaction to a hawkish surprise is a different distribution from the terminal-rate reaction, both in width and in skew (long-end is more diffuse, short-end is more bunched around zero). A shared calibration would set the band on whichever distribution wins on the residual ladder; per-head calibration sets the band that actually covers the per-head 1 - α target.

`ConformalManifest` grows two optional dicts: `rates_residual_quantiles` maps the head short name to the (1 - α) absolute-residual quantile in raw bps, and `rates_softmax_quantiles` maps the same key to the APS threshold when the aux classifier rode. Both are populated by `_maybe_write_rates_conformal_manifest` off `EvaluationMetrics.rates_metrics` after the best-fold model is restored, on the val partition only. Each fit is independent — no residual pooling across heads, no shared calibration sample. The classification-side fields (`softmax_quantile` for the regime head) stay where they were on the manifest, so the rates extension is purely additive.

`save_manifest` strips `None` keys before writing, so a manifest without active rates heads is byte-identical to the pre-#292 sidecar (no `rates_*` keys on disk). `load_manifest` reads pre-#292 files and resolves the new fields to `None`, so a stale single-head sidecar binds against the post-#292 inference path with no migration step. The backwards-compatible read is pinned by `tests/unit/test_conformal_multi_head.py::test_pre_292_single_head_manifest_loads_cleanly`.

## Why opt-in on the aux classifier

The 3-class direction head is product surface, not the primary training target. Three reasons it stays opt-in instead of mounted by default:

The supervised label is derived from per-fold tertile cuts on the raw bps target. Fitting the cuts on the train slice introduces a stronger train-side dependence than the regression head needs — the regression target is a column off events.parquet, the classification target is a quantile-bucketed transform of that column. Mounting the classifier by default makes the per-fold cut implicit; opt-in surfaces the choice on the CLI.

The classifier sees a 3-way categorical, the regression head sees a continuous bps target. When both are trained jointly the gradient sharing on the shared backbone favours whichever loss has the cleaner signal — on the rates targets we are training, that is the regression branch. Letting an operator run the aux head ON / OFF as an ablation is the cheap way to measure the auxiliary contribution.

The inference path can render a directional bucket off the regression sign without an aux head, just less informative (no probability mass, no APS prediction set). The cost of a default-on aux head is therefore non-zero (extra parameters, extra calibration step, extra response field plumbing) for a feature that an operator can choose to mount later by re-fine-tuning.

## What stays out

The trajectory head is tracked separately (#272 + ADR 0022). The fourth head would lift the joint-loss combinatorics into a 4 × 4 sweep over per-head lambdas, which is a separate scope question.

The dual-head retrofit on the vol-regime classifier landed under #304 / ADR pending. The regression head on the regime classifier sits on a different target column (`log(forward_realized_vol_10d)`) and trained under a different alpha-blend. This PR does not touch that retrofit; the rates heads inherit the same dual-head pattern but with their own targets, their own scalers, their own conformal interval.

The §6 multi-head vs single-head comparison row is GPU-blocked. The trainer-side knobs (`--targets`, `--rates-classification-heads`, `--rates-head-mode`, `--rates-alpha`) all ship; producing the comparison numbers requires the canonical sweep to return, which is a separate ops question.

## Acceptance

Once a multi-head sweep returns, the comparison row reads (single-head regime baseline) vs (regime + 2y + terminal multi-head) on macro-F1 for the regime head and per-head MAE-bps / R² / directional-accuracy for the rates heads. The two regression heads are expected to depress regime macro-F1 slightly (gradient sharing on a fixed backbone capacity) and produce per-head MAE-bps within the literature band for FOMC-event yield-curve reactions (~5 bps on the short end, ~10 bps on the terminal). Lift on the rates heads inside that band — methodology supports the multi-target framing on this corpus; outside the band — the shared encoder is not the bottleneck and per-head encoder specialisation deserves the follow-up.

`pytest tests/unit/test_multi_target_heads.py tests/unit/test_conformal_multi_head.py` covers the head-mount contract, the `--targets` resolver, the aux-flag round-trip through `_coerce_payload_config`, the per-head calibration math, the schema growth, and the backwards-compatible read on a pre-#292 manifest.

## References

- `backend/app/models/{research_model,serving_model,factory,config,rates_heads}.py`
- `backend/app/training/{loop,checkpoint}.py`
- `backend/app/evaluation/conformal.py`
- ADR 0015 (regression-canonical objective on continuous targets)
- ADR 0027 (FOMC-attributable rates target — #305 strict-prior projection)
- ADR 0028 — retrieval-augmented features (per-event analog block, different surface)
- Lei, J., & Wasserman, L. (2014). *Distribution-Free Prediction Bands for Non-parametric Regression*.
- Romano, Y., Sesia, M., & Candès, E. (2020). *Classification with Valid and Adaptive Coverage* (APS).
