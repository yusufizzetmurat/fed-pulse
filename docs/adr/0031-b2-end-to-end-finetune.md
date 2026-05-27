# ADR 0031 — B2 end-to-end fine-tune on vol-regime classification

Status: accepted, harness code path live; canonical sweep deferred to operator.
Date: 2026-05-28.
References:
- Issue #213 (closes).
- Issue #330 — canonical encoder split, classifier vs retrieval substrate.
- ADR 0019 — canonical encoder split (`role: classifier` resolves to `finbert_fomc_only`, fallback `finbert_fed_adjacent`).
- ADR 0023 — train/inference contract enforcement (the LSTM stack this harness bypasses).
- `backend/app/data/finetune_pilot_b2.py` — harness module.
- `backend/app/data/finetune_pilot.py` — Phase 3 stance-label fine-tune (the layout precedent this harness mirrors).
- `backend/app/training/loaders.py` — `fit_vol_regime_quantiles` + `vol_regime_class_for` (per-fold tertile fit, train-slice only).
- `backend/app/evaluation/classification_breakdown.py` — `compute_classification_breakdown` (per-class metrics surface the §6 tier table consumes).
- `backend/artifacts/experiments/dual_head_comparison_canonical.json` — sweep-JSON schema this harness mirrors.

## Context

The §6 Tier 3 / Tier 5 / Tier 6 / Tier 7 headline cells all wire a pre-trained encoder through a frozen pooled-embedding interface to a downstream LSTM regime classifier. The encoder produces a 768-dim mean-pooled document embedding once per FOMC document and never sees the regime-classification gradient. The LSTM head reads the pooled embedding, concatenates it with the rich market block, and is the only piece of the stack that fits to the vol-regime target.

Two readings of the post-Phase-A/B/C plateau are consistent with the §6.6 numbers:

1. The encoder substrate is the bottleneck. A FinBERT pre-trained on financial news has not seen FOMC-style hawkish / dovish framing during pre-training; the pooled-document representation is not aligned with the regime axis. An end-to-end fine-tune that lets the regime gradient reach the encoder would align the representation with the target, and the headline macro-F1 would lift.
2. The LSTM detour is the bottleneck. The mean-pool collapses 200-300 tokens of FOMC text into a single 768-dim vector before the head sees them. Per-token signal is lost. An end-to-end fine-tune with attention over the full token sequence (the standard `BertForSequenceClassification` head) would preserve the signal the mean-pool destroys, and the headline macro-F1 would lift.

Both readings predict an end-to-end fine-tune produces a different macro-F1 than the LSTM-on-frozen-embeddings stack. The §6.6 placeholder row reserved for B2 frames this as the architectural diagnostic. The point of the experiment is to settle which reading is correct; the result is thesis-quality either way:

- A lift on B2 means the encoder freeze is the bottleneck and the LSTM-on-frozen-embeddings architecture is the right shape but the wrong gradient flow. The §6 reporting then frames the headline tier table as a lower bound and B2 as the production recipe.
- A null result on B2 means the encoder freeze is not the bottleneck. The SNR ceiling holds at the corpus level (4 100 supervised events, three-class tertile labels with structurally bounded discriminability). The §6 reporting then doubles down on the Phase A/B/C ensemble headline as the architectural ceiling.
- A negative result on B2 means the end-to-end recipe over-fits the small corpus and the frozen-embedding stack's regularising effect is doing real work. The §6 reporting then keeps the canonical stack and rejects the encoder-freeze-is-bottleneck reading.

The methodology contribution is the diagnostic itself. The single-cell delta against the §6.6 placeholder row is the load-bearing finding, not the absolute macro-F1.

## Decision

Ship a standalone fine-tune harness at `backend/app/data/finetune_pilot_b2.py`. The harness:

- Reads FOMC document text from the training package's `registry_normalized.jsonl` (dropping cross-bank `sample_weight=0` rows).
- Reads `forward_realized_vol_10d` per event from `events.parquet`, joins on `event_date`.
- Walks the canonical 4-fold walk-forward protocol (`fold_manifest_expanding_walk_forward.json`).
- Per fold, fits 3-class vol-regime tertile cutoffs on the TRAIN slice only via `app.training.loaders.fit_vol_regime_quantiles`, then assigns the 3-class label to every row in the fold via `vol_regime_class_for`. Rows whose forward-vol is missing fall out of the kept set.
- Constructs `AutoModelForSequenceClassification.from_pretrained(encoder_alias, num_labels=3)` against the canonical classifier-role encoder per ADR 0019 (defaults to `finbert_fomc_only`, fallback `finbert_fed_adjacent`). Operator overrides via `--encoder-alias`.
- Fine-tunes for 5 epochs at AdamW `lr=2e-5`, `weight_decay=0.01`, `train_batch_size=16`, `max_length=256` (the canonical HF defaults for a 110M-param BERT-family backbone on a small classification corpus).
- Evaluates on the test slice; emits the per-(seed, fold) `classification_breakdown` block via `compute_classification_breakdown` so the result drops into the existing §6 tier table without bespoke aggregation.
- Writes a sweep artefact at `backend/artifacts/experiments/finetune_pilot_b2.json` whose schema mirrors `dual_head_comparison_canonical.json` (per-trial cells with `metrics`, per-fold `tertile_cutoffs`, summary block with macro-F1 mean + std + block-bootstrap CI).

### Hyperparameter choices

| Knob | Value | Reason |
| --- | --- | --- |
| Optimiser | AdamW | Standard for BERT-family fine-tune; the only choice consistent with the `transformers<5` import surface the pod runs. |
| Learning rate | 2e-5 | Canonical AdamW LR for BERT-family fine-tune on small classification corpora (Devlin et al. 2018; the same LR `finetune_pilot.py` uses for the stance head). Smaller (1e-5) underfits at 5 epochs; larger (5e-5) destabilises a 110M-param encoder on a 4 100-event train slice. |
| Weight decay | 0.01 | Standard AdamW weight decay; the same value `finetune_pilot.py` and `train_text_multi_axis_classifier.py` use. |
| Epochs | 5 | Long enough for the encoder to see every train row five times; short enough that the run finishes in ~30 min per (seed, fold) cell on an A100, keeping the 5-seed × 4-fold sweep under 10 GPU-hours. The stance-head fine-tune `finetune_pilot.py` runs for 3 epochs on a 4 800-row corpus and converges; 5 is a slight margin for the harder 3-class target. |
| Train batch size | 16 | Fits a 24 GB A100 with `max_length=256` on a 110M-param backbone. Larger batches (32, 64) are an operator override; smaller is a fallback for the 16 GB pods if Runpod allocates one. |
| Eval batch size | 32 | Inference-only memory; doubles the train batch. |
| Max tokens | 256 | The same `max_length` `finetune_pilot.py` ships with. FOMC statements run 1 500-3 000 chars (~200-400 BERT-WPS tokens); 256 keeps the head of the document and covers the policy-decision paragraph that carries the regime signal. Longer windows (512) double the memory and runtime for marginal additional context, deferred to a follow-up sweep. |
| LR scheduler | none | Constant LR over 5 epochs. AdamW with constant 2e-5 is the canonical Devlin recipe; a warmup-linear-decay schedule is a downstream knob the operator can flip via `--lr-scheduler` in a follow-up if the constant-LR sweep is unstable. |

The harness uses the canonical `(11, 29, 47, 71, 97)` seed set + the canonical 4-fold walk-forward protocol per `docs/benchmark-policy.md`, so the sweep total is 20 cells.

### Per-fold tertile cutoffs — train-slice only

The 3-class vol-regime label is fitted per fold on the TRAIN slice only via `fit_vol_regime_quantiles`. The same helper backs the LSTM-on-frozen-embeddings stack and the dual-head canonical comparison; reusing it guarantees the labels B2 fits against are byte-identical to the labels the §6 tier table reports. The contract is:

- `cutoffs = fit_vol_regime_quantiles(train_forward_vols, n_classes=3)` returns the `(q33, q67)` tuple on the TRAIN slice.
- `class_idx = vol_regime_class_for(test_forward_vol, cutoffs)` labels every test row against the cutoffs the TRAIN slice fitted. The test slice never influences the cutoffs.
- Rows whose forward-vol is missing / non-finite (`vol_regime_class_for` returns `-1`) fall out of the kept set so the labelling pass never silently coerces a missing target to class 0 (calm).

The unit test `test_build_partition_targets_fits_cutoffs_on_train_slice_only` locks the contract: it builds a train slice spanning `[0.0, 0.06]` and a test slice spanning `[0.50, 0.60]`, asserts the train + test cutoffs are byte-identical, and asserts every test row lands in class 2 (high) because the train slice never saw a value that high. If the cutoffs leaked the test slice the upper boundary would shift and the assertion would fail.

### Encoder choice — canonical classifier role per ADR 0019

The default encoder alias resolves via `resolve_by_role("classifier")` per ADR 0019. That returns `finbert_fomc_only` against the current `registry.yaml`. Operator overrides via `--encoder-alias` for sibling runs against `BAAI/bge-large-en-v1.5`, `finbert_fed_adjacent`, or any other registered encoder. The fallback to `finbert_fed_adjacent` mirrors `train_text_multi_axis_classifier.py::DEFAULT_ENCODER_ALIAS` so the harness inherits the same unpinned-local guard the other classification training entrypoints have.

The harness does NOT touch the LSTM-on-frozen-embeddings code path. The existing `make canonical-comparison` target stays byte-identical to pre-#213 — both the embedding cache and the `app.train_forecaster` entrypoint are unchanged. The §6 tier table populated by the canonical comparison is unaffected by anything in this PR; the B2 row is its own placeholder row pending the GPU sweep.

### Output schema — mirrors `dual_head_comparison_canonical.json`

The sweep artefact lives at `backend/artifacts/experiments/finetune_pilot_b2.json`. Top-level keys: `pipeline`, `training_package_id`, `encoder_alias`, `seeds`, `fold_ids`, `epochs`, `train_batch_size`, `learning_rate`, `weight_decay`, `max_length`, `n_classes`, `labels`, `started_at_utc`, `trials`, `summary`. `trials` is a list keyed by seed; each entry carries a `folds` list of per-fold cells with `metrics.regime_f1_macro`, per-class `classification_breakdown` block, tertile cutoffs, and class-count distributions. The `summary` block carries `regime_f1_macro` (mean / std / min / max / n across the 20 cells) plus `regime_f1_macro_ci` (block-bootstrap 95% CI with block size = number of folds).

The schema deliberately mirrors `dual_head_comparison_canonical.json` so the §6 tier table reads B2 + the canonical-comparison numbers off the same aggregator code. No bespoke aggregation surface for B2.

### Acceptance row in §6.6

The B2 row lands in §6.6 with the same shape as the existing tier rows:

| Tier | Channels | Macro-F1 (n=20) | Δ vs Tier 2 |
| --- | --- | ---: | ---: |
| B2 | end-to-end fine-tune | _pending Runpod sweep_ | _pending_ |

The cell stays a placeholder until the canonical Runpod sweep lands the JSON artefact. The framing in the §6 prose calls out the architectural-question framing — the row is a diagnostic against the encoder-freeze-is-bottleneck hypothesis, not a horse-race entry.

## Alternatives considered

**Layer the end-to-end fine-tune into `app.train_forecaster`.** Extend the existing training loop to optionally unfreeze the encoder and back-propagate the regime CE through the encoder weights. Rejected: the existing loop builds a sequence-of-bars `FeatureVector` per supervised event, pools the encoder output once per row, and concatenates with the rich market block. Wiring an end-to-end fine-tune through that path would require either (a) recomputing the encoder forward on every batch (~10× memory + runtime) or (b) gradient-checkpointing across the mean-pool, which is a substantial refactor. A standalone harness on a separate JSON artefact is cheaper and keeps the canonical-comparison byte-identical.

**LoRA fine-tune over the same harness.** Mount LoRA adapters on the encoder (rank 8, alpha 16) and fine-tune the adapters instead of the full encoder. Rejected as the headline: LoRA is a parameter-efficient compromise; the methodology question is whether end-to-end fine-tuning lifts the headline at all, and LoRA leaves the encoder frozen modulo a low-rank correction. A future PR can layer LoRA on top of the same harness as a `--use-lora` flag; this PR ships the full-fine-tune diagnostic so the LoRA delta has a baseline to read against.

**Multi-task head (stance + vol-regime).** Add a stance auxiliary loss on top of the vol-regime CE so the encoder gets supervision from both axes simultaneously. Rejected: the multi-task ablation is its own methodology question (Bundle A.2 in #228 already established the substitute-not-complement reading of stance vs vol-regime). Conflating the two axes in B2 would muddy the architectural-question framing. A single-task vol-regime fine-tune is the cleanest control for the encoder-freeze diagnostic.

**Run the full sweep in CI.** Run the 5-seed × 4-fold × 5-epoch sweep end-to-end inside the CI matrix. Rejected on the compute budget: ~30 min per cell on an A100 = ~10 GPU-hours for the full sweep, far past the CI envelope. CI runs the 1-epoch synthetic-fixture smoke (under 60 s on CPU); the full sweep is a Runpod follow-up.

## Consequences

### Honest framing per ADR 0019 / #330

The result is thesis-quality either way. The §6.6 prose around the B2 row calls out the architectural-question framing: "B2 tests whether the LSTM detour is the bottleneck. The current §6 tier table's Tier 3 / Tier 5 / Tier 6 / Tier 7 cells freeze the encoder; B2 unfreezes it. The cell is a diagnostic, not a horse-race entry."

### Methodology

The per-fold tertile contract is the load-bearing claim. The labels B2 fits against are byte-identical to the labels the §6 tier table reports, and the same `fit_vol_regime_quantiles` helper backs both paths. The unit test locks the train-only contract; the cutoffs on a train slice spanning `[0.0, 0.06]` are unchanged when a test slice spanning `[0.50, 0.60]` is appended.

### Compute

Per-cell budget: ~30 min on an A100 at AdamW 2e-5, batch 16, max_length 256, 5 epochs on ~3 200 train rows. Full 5-seed × 4-fold = 20 cells → ~10 GPU-hours. Tractable on a single Runpod A100 pod overnight. CI smoke stays under 60 s on CPU via the tiny-random-bert stub at 1 epoch over a 6-row synthetic fixture.

### Sweep hand-off

The canonical sweep against `make finetune-pilot-b2 TRAINING_PACKAGE_ID=<id>` is a Runpod follow-up. The harness writes to `backend/artifacts/experiments/finetune_pilot_b2.json`; the §6.6 placeholder row in the wiki populates once the artefact lands. Default override surface: `ENCODER_ALIAS=BAAI/bge-large-en-v1.5` to swap to the BGE substrate; `ENCODER_ALIAS=finbert_fed_adjacent` to pin the pre-classifier-role canonical encoder.

The harness disables `torch.compile` via `app.training.runtime_compat.ensure_compile_safe` for the same reason the canonical-comparison runner does — the pod's triton install mismatches the runtime; TorchDynamo crashes on a clean import. The `TORCHDYNAMO_DISABLE=1` env var is set at runner entry.
