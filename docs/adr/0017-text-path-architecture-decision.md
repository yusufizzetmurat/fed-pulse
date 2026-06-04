# ADR 0017 — Text-path architecture A/B (issue #327)

Status: Proposed (awaiting sweep results).
Date: 2026-05-27.
References:
- Issue #327.
- ADR 0015 (regression-canonical objective) — the canonical training objective the A/B reads off.
- ADR 0016 (forecaster research / serving split) — the class layout the new arms slot into.
- `backend/app/models/forecaster_base.py` — `prepare_recurrent_input` per-bar branch.
- `backend/app/models/research_model.py` / `serving_model.py` — per-bar forward signature plumbing.
- `backend/app/models/flat_mlp.py` — Arm B (`ForecasterFlatMLP`).
- `backend/app/models/text_adapter_warm_start.py` — proxy-task adapter warm-start utility.
- `backend/app/training/loaders.py` — `build_per_bar_text_tensor` helper.
- `scripts/run_text_path_ab.py` — three-way runner.
- `fed-pulse.wiki/06_Deep_Learning_Roadmap.md §6.15` — methodology section + results table.

## Context

The forecaster pools per-event FOMC text into a single vector per event and tiles that vector across all 20 lookback bars before feeding the recurrent core. The sequence model's capacity therefore contributes nothing to the text path: the text feature is constant across the sequence axis, so any architecture variance the 8 cores produce on the text portion of the input is a market-feature ablation in disguise.

Three text-side adapters compound the problem:

- `ChunkAttentionPooler` (Variant B / C) — pools the chunk axis.
- `EmbeddingAdapter` — projects the pooled chunk vector.
- `TextEmbeddingAdapter` — projects the encoder-native pooled embedding.

All three are zero-initialised, so an activated text path starts from the rich-features-only baseline. The text adapter sees no gradient signal until the loss surface accidentally finds a non-zero direction worth pushing in.

The advisor scope mandates a clean text-vs-market separation for the LLM-vs-LSTM comparison. The current broadcast-static framing makes that separation impossible to measure.

## Decision

Run a three-way comparison on the canonical fold protocol (post-#322; `head_mode='dual'` per the PR #354 default flip) and let the result decide whether the sequence-model framing of the text path retires.

### Arm A — per-bar text features

Feed a per-bar pooled-text tensor `(B, seq_len, in_dim)` so the recurrent core consumes actual per-day text dynamics over the lookback window. Implementation:

- `FeatureVector.text_per_bar: list[list[float]] | None` carries the per-bar payload through the loader.
- `app.training.loaders.build_per_bar_text_tensor` materialises the `(num_windows, T, in_dim)` tensor + per-bar missing-flag mask. Default smoke path: tile-replicate the prior-N FOMC pool across the lookback so the contract holds without a new corpus ingestion. A future per-day loader populates `text_per_bar` directly and slots into the same tensor.
- `app.models.forecaster_base.prepare_recurrent_input` projects each bar's pooled embedding through the same adapter (no broadcast), then concatenates onto the per-bar market vector.
- `--text-channel per_bar` is the CLI knob.

The implementation preserves the broadcast-static byte-identity contract: with `text_channel='scalar'` (the default) the forward path is unchanged.

### Arm B — flat MLP

Drop the sequence wrap on the text path entirely. `app.models.flat_mlp.ForecasterFlatMLP` mean-pools the market window across the sequence axis and feeds `[pooled_market || pooled_text_adapter || rich]` through a two-layer MLP into the same multi-task / regression head shapes the recurrent forecaster mounts. This is the direct comparator for the LSTM-on-broadcast baseline: if Arm B matches or beats the current arm on the canonical objective, the LSTM's contribution to the text-conditional forecast is the broadcast plumbing, not the sequence-model capacity.

- `architecture='flat_mlp'` is added to `FORECASTER_ARCHITECTURES`.
- `build_forecaster` dispatches the new architecture to `ForecasterFlatMLP`.
- The class is research-only; serving rejects it (the `/analyze` close/vol time series presumes the recurrent core).

### Adapter warm-start

`app.models.text_adapter_warm_start.pretrain_text_adapter(corpus_path, output_path)` fits the `text_embedding -> stance` proxy task on the FOMC corpus (or any pooled-embedding + stance-label corpus) and persists the warmed adapter `state_dict`. `app.training.loop._build_model` accepts a `text_adapter_warm_start` path and loads the persisted weights into the live `text_adapter` submodule at construction time, replacing the zero-init starting point.

The proxy task is intentionally narrow: one linear classifier head, fixed LR, no warmup. The objective is to lift the adapter off the zero subspace, not to chase macro-F1 on the proxy.

## Consequences

- The three-way comparison cell lands in `06_Deep_Learning_Roadmap.md §6.15` with the runner output JSON at `backend/artifacts/experiments/text_path_ab.json`.
- Acceptance: if Arm B wins or matches the broadcast-static arm on `regression_rmse_log_rv` (the canonical objective post-#322), the sequence-model framing of the text path is retired. The ADR is re-statused to `Accepted` and a follow-up ADR documents the resulting serving-class change.
- If Arm A wins, the per-bar loader path is promoted from "tile-replicate the prior-N pool" to "real per-day daily-frequency sentiment over the prior-N FOMC docs". The loader change is scoped as a follow-up to issue #327 because it requires a new data-pipeline stage.
- If the broadcast-static arm holds, the ADR records the verdict and the adapter warm-start lands as the only on-by-default change; the per-bar / flat-MLP plumbing stays in-tree as opt-in CLI flags so a future re-test can re-run without re-wiring.
- The flat-MLP class shares the head shape contract with the recurrent forecaster (multi-task head, optional log-RV regression head, rates heads), so a downstream comparison on the regression band or the rates panel remains directly comparable.

## Sweep contract

- Training package: the canonical `tp_v3_macro_aug_2026_05_25_fwd_strict_sentiment_market_core_v1.1_epv1_v1.0` (matches the §6.10 canonical baseline).
- Seeds: `{11, 29, 47, 71, 97}` (`docs/benchmark-policy.md`).
- Folds: every fold in `fold_manifest_expanding_walk_forward.json`.
- Epochs: 40 (matches the `make canonical-comparison` budget).
- Head mode: `dual` with `regression_alpha=0.5`.
- Text encoder: `finbert_fed_adjacent` (the canonical FOMC encoder).
- Text adapter dim: 64.
- Runner: `make text-path-ab TRAINING_PACKAGE_ID=<id>`.

The sweep is a follow-up. This PR ships the code, tests, and the placeholder methodology section so the cells populate via `make text-path-ab` and the table updates in a follow-on commit.
