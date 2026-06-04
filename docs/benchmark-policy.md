# Benchmark Policy

## Status
Policy is fixed for the current benchmark cycle. Method changes require a new version.

## Evaluation Scope
- Targets: close, volatility
- Horizons: `1d`, `3d`, `5d`, `10d`
- Modes: `fast` (only runtime mode shipped after PR #265); replay-mode pinned per-fold checkpoints are used for reproducibility

## Split and Seed Rules
- Split: expanding walk-forward only
- Constraint: `train_end < val_start < test_start`
- Identical folds across model comparisons
- Official seed set: `{11, 29, 47, 71, 97}`
- Report mean and standard deviation across seeds

## Metrics
- Forecast quality: RMSE, MAE, MAPE
- Reliability: coverage (and calibration error if available)
- Runtime: latency `p50`/`p95`, adaptation time, peak memory (if available)

## Canonical Training Objective
ADR 0015 (`docs/adr/0015-regression-canonical-objective.md`, issue #322; amended 2026-05-27 after the three-way comparison sweep at `backend/artifacts/experiments/dual_head_comparison_canonical.json`) sets the canonical training objective for the vol-regime head to the dual head (`head_mode='dual'` with `regression_alpha=0.5`). The joint loss is `(1 - 0.5) * CE + 0.5 * MSE` on the same backbone; CE supervises the 3-class calm / normal / high label, MSE supervises per-fold standardised `log(forward_realized_vol_10d)`. The 3-class label is recovered at the UI as both the dual head's classifier-branch argmax and a bucketing of the regression branch against the per-fold `vol_regime_quantiles` cutoffs persisted in `fold_manifest_expanding_walk_forward.json`. The sweep showed regression-only loses ~20pp macro-F1 vs classification on the UI-bucket label space (0.220 ± 0.081 vs 0.417 ± 0.051), while dual matches classification F1 within block-bootstrap CI overlap (0.419 ± 0.070) and ships the regression band (RMSE-log_rv 1.004 ± 0.200). The remaining head modes (`head_mode={classification,regression}` plus `regression_alpha`) stay available for the methodology comparison in §6.15 of the deep-learning roadmap and for ablation. Headline vol-regime rows in published reports cite both the UI-bucket macro-F1 and the RMSE-log_rv from the dual head's regression branch.

## Aggregation Rule
The canonical pooling rule for macro-F1 across walk-forward folds is mean-of-fold-means: per-fold macro-F1 averaged unweighted across folds. The secondary rule is row-pooled: every per-row prediction is concatenated across all folds into one confusion matrix, then macro-F1 is computed once. Both numbers are published on every honest macro-F1 cell so the per-fold class-balance variance stays visible. The row-pooled number weights folds by support; the mean-of-fold-means number weights folds equally and is the conservative read against the `wf_fold_4` zero-`calm` slice (R-17).

Honest macro-F1 reports MUST include:
1. Per-class P/R/F1 footnote on every published macro-F1 cell — precision, recall, F1, and support per class. Pooled headlines that suppress this footnote can mask a degenerate per-class distribution and are not eligible for publication.
2. Fold-4 with-and-without rows on every headline cell — both `with all folds` and `without wf_fold_4`, so the magnitude of the zero-`calm`-class fold's contribution is visible. Flag if the two readings diverge by more than the bootstrap CI half-width.
3. Macro-release with-and-without rows on every headline cell — both `with macro-release-augmented rows` and `FOMC-only`, so the §6.7 Chunk-3 lift attribution stays auditable. The FOMC-only cell is the primary number; the mixed-pool cell is the secondary comparator.

The four reporting variants (mixed-pool, FOMC-only, fold-4 with/without, macro-release with/without) are computed by `backend/app/evaluation/reporting.py` (issue #323). Block-bootstrap CIs use 1000 resamples at `block_size=20` by default, matching the convention in `backend/app/evaluation/regime_pooled_aggregator.py`.

## Leakage Rules
1. No future-derived features
2. No future target leakage in feature creation
3. No near-duplicate leakage across train/test in the same fold
4. Pseudo-labeling excludes final reporting holdout
5. Scaling/statistics fit on train only

## Per-Feature Provenance
Every `FeatureVector` column carries a declared `as_of` offset relative to `row.event_date`. The declarations live in `docs/feature-provenance-audit.md` and cover four bands: `T-Δ` (data observable strictly before the event), `T (snapshot)` (a quantity defined on the event itself and observable from the released document or its calendar entry), `T+Δ` (post-event data), and `future-derived` (training targets only).

Contract:
1. No scalar feature column on a lookback bar may read from a source post-dating `row.event_date`. Lookback bars consume only `T-Δ` data; `T (snapshot)` columns are document-level signals on `T` itself.
2. `T+Δ` columns are training targets only and are mounted on the appended event-day target frame, not on lookback bars.
3. New `FeatureVector` columns require a row in the audit table before merge; the regression test enforces the column inventory.
4. `T (snapshot)` columns whose sources are post-event by construction (e.g. monetary-policy surprise quantities defined on a `[T-1, T+1]` window) are flagged in the audit's "Leaks found" section and tracked through an ADR + canonical-cell re-baseline.

Regression coverage: `tests/regression/test_feature_provenance_as_of.py` walks a canonical training package and asserts the contract per row. The test is the gate; the audit is the human-readable inventory.

## Sealed Holdout

The canonical training package's `event_date` cutoff is 2024-12-31. Every FOMC event filed after that date sits under `data/external/sealed_holdout/` and has never been part of any sweep / val / test partition, never been visible to any embedding / DAPT / encoder used during training, and never been touched by any HP grid. It is the reserve slice that closes R-14 (`fed-pulse.wiki/09_Risk_Register.md`).

One-shot rule. The sealed slice is queried exactly once at final-report time. The canonical configuration — already chosen, frozen, and published in-protocol — is evaluated against the sealed slice and the resulting headline is filed alongside the in-protocol headline. No iteration on the sealed result is permitted: no re-tuning, no model swap, no follow-up sweep. A repeat read against the same on-disk seal is a build failure unless the operator passes `force=True` and accepts the warning trail.

AUDIT_TOKEN integrity contract. `data/external/sealed_holdout/AUDIT_TOKEN` is a plain-JSON object with three fields:
- `seal_status` — `"sealed"` until the one-shot fires, then `"broken_by:<audit_caller>"`.
- `usage_count` — strictly monotone; increments on every call to `load_sealed_holdout`, including forced repeats.
- `last_accessed_utc` — ISO-8601 timestamp of the last consumption; `null` while sealed.

The committed token reads `{"seal_status": "sealed", "usage_count": 0, "last_accessed_utc": null}`. `tests/regression/test_sealed_holdout_audit.py` fails the build if any of those fields drift, if the sealed JSONL drops below four entries, or if any production module under `backend/app/` (outside the loader itself) imports `load_sealed_holdout`. `audit_status()` is the only public hook safe to import from production code; it returns the token as a dict without mutating disk state.

Stub rows whose `text` field starts with `# pragma: stub` emit a hard warning on load so a sealed-eval headline cannot be silently published against placeholder text. Real scraped text must replace stubs before the one-shot run.

Final-report contract. The report's headline carries both the in-protocol number and the sealed-eval number. The two are reported side-by-side; neither replaces the other; the delta is the honest read on search-space variance.

## Versioning
Required IDs:
- `dataset_version`
- `feature_version`
- `model_version`
- `run_id`
- `training_package_id`

Immutability rules:
- Published `run_id` is never reused
- Checkpoints behind a version are not replaced silently
- Protocol/split/feature changes require version bump

## NLP Baseline Selection
Candidates run on the same folds/seeds. Winner order:
1. Macro-F1
2. Worst-class F1
3. Latency `p95` (tie-break)

## Report Requirements
Official reports must include protocol ID, all version IDs, runtime mode, fold/seed info, metrics, and known deviations. Reports that include source-type-stratified breakdowns must report per-`source_type` mean and standard deviation alongside the headline aggregates so cross-source generalisation stays visible.

## Source-mix Stratification
Labelled sources have different stance priors (TDW skews hawkish 49/26/24, Op-Fed skews dovish 59/30/11). To prevent a model from learning the source instead of the stance, every published training package must record:
- `source_drift_max` per fold in `fold_manifest_*.json` — the largest absolute share-point gap between the train slice's source distribution and either the val or test slice (e.g. 0.50 = train is 100% TDW but val is 50/50 TDW/Op-Fed).
- `source_drift_per_fold` and `source_drift_max_across_folds` in `dataset_metadata.json` for a one-glance package-wide check.

`build_training_package.py --source-drift-tolerance T` (default `0.0`, report only) aborts the build with a non-zero exit code if any fold's drift exceeds `T`. The convention for published packages is `T = 0.15` (15 percentage points); packages built without the gate must record `source_drift_tolerance: 0.0` in the manifest so reviewers can see the check was disabled.

## Contamination Handling
Encoders that may have seen the project's training rows during their own pretraining are excluded from the primary NLP comparison table. The current case: `gtfintechlab/FOMC-RoBERTa` was trained on the same Trillion Dollar Words corpus that this project ingests as `hf_fomc_communication`. The 1-epoch smoke macro-F1 of 0.8409 on fold-2-test is almost certainly a test-leak signal.

Policy:
1. Exclude `gtfintechlab/FOMC-RoBERTa` from the headline comparison row.
2. Cite the contaminated number as a ceiling reference in the discussion chapter only — labelled as "upper-bound for what's achievable on this corpus given likely contamination".
3. The encoder may stay in `finetune_batch.ENCODERS` for completeness, but the published table must use only independent encoders (BERT-base, FinBERT, FinBERT-FOMC, DistilBERT, DeBERTa-v3).
4. Any future encoder whose pretraining mixture overlaps with this project's training rows falls under the same exclusion rule. Add a one-line entry to the deny-list in `finetune_batch.py` and document the rationale in `06_Deep_Learning_Roadmap.md`.

### `gtfintechlab/fomc-roberta-any-exp` audit (2026-05-27, #339)

The dashboard sentiment service has carried `gtfintechlab/fomc-roberta-any-exp` as its primary HF fallback (`PRIMARY_HF_MODEL_ID` in `backend/app/services/text_encoder.py`). Two findings from the encoder-parity audit:

1. The repository is not resolvable on the Hugging Face Hub: `HfApi.model_info('gtfintechlab/fomc-roberta-any-exp')` returns 404 under an authenticated token, and the gtfintechlab org listing does not surface the name. The dashboard has therefore been falling through to `distilbert/distilbert-base-uncased-finetuned-sst-2-english` (POSITIVE / NEGATIVE labels, not hawkish / dovish / neutral) the entire time the registry pin claimed otherwise.
2. The `-any-exp` name structure follows the gtfintechlab Trillion Dollar Words family. Absent an upstream model card declaring an independent training corpus, the conservative call is to inherit the sibling's R-13 contamination flag.

Verdict: deny-list. The key `gtfintechlab_fomc_roberta_any_exp` joins `gtfintechlab_fomc_roberta` in `finetune_batch.CONTAMINATED_ENCODER_KEYS` and is excluded from the headline NLP table. The registry `revision: main` value is intentionally left as a non-reproducible marker so future readers see "the audit happened and the encoder is gated, not pinned"; the encoder must not be served until either the upstream repo becomes reachable with a clean training-corpus declaration, or a replacement encoder is wired into the dashboard fallback chain.
