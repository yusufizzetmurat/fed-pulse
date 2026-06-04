# ADR 0031 — B2 end-to-end fine-tune on vol-regime classification

The §6 Tier 3 / Tier 5 / Tier 6 / Tier 7 headline cells wire a pre-trained encoder through a frozen pooled-embedding interface to a downstream LSTM regime classifier. The encoder produces a 768-dim mean-pooled document embedding once per FOMC document and never sees the regime-classification gradient. The LSTM head reads the pooled embedding, concatenates it with the rich market block, and is the only piece of the stack that fits to the vol-regime target.

Two readings of the post-Phase-A/B/C plateau are consistent with the §6.6 numbers. Either the encoder substrate is the bottleneck: a FinBERT pre-trained on financial news has not seen FOMC-style hawkish / dovish framing during pre-training, so the pooled-document representation is not aligned with the regime axis, and an end-to-end fine-tune that lets the regime gradient reach the encoder would align the representation. Or the LSTM detour is the bottleneck: the mean-pool collapses 200-300 tokens of FOMC text into a single 768-dim vector before the head sees them, per-token signal is lost, and an end-to-end fine-tune with attention over the full token sequence (the standard `BertForSequenceClassification` head) would preserve what the mean-pool destroys.

Both readings predict an end-to-end fine-tune produces a different macro-F1 than the LSTM-on-frozen-embeddings stack. The §6.6 placeholder row for B2 frames this as the architectural diagnostic. The result is publishable either way: a lift means the encoder freeze is the bottleneck and the LSTM-on-frozen-embeddings architecture is the right shape with the wrong gradient flow (§6 then frames the headline tier table as a lower bound and B2 as the production recipe); a null result means the encoder freeze isn't the bottleneck and the SNR ceiling holds at the corpus level (4 100 supervised events, three-class tertile labels with structurally bounded discriminability — §6 doubles down on the Phase A/B/C ensemble headline as the architectural ceiling); a negative result means the end-to-end recipe over-fits the small corpus and the frozen-embedding stack's regularising effect is doing real work (§6 keeps the canonical stack and rejects the encoder-freeze hypothesis). The single-cell delta against §6.6 is the finding, not the absolute macro-F1.

## What lands

A standalone harness at `backend/app/data/finetune_pilot_b2.py`. The harness:

- Reads FOMC document text from the training package's `registry_normalized.jsonl`, dropping cross-bank `sample_weight=0` rows.
- Reads `forward_realized_vol_10d` per event from `events.parquet`, joins on `event_date`.
- Walks the canonical 4-fold walk-forward protocol (`fold_manifest_expanding_walk_forward.json`).
- Per fold, fits 3-class vol-regime tertile cutoffs on the TRAIN slice only via `app.training.loaders.fit_vol_regime_quantiles`, then assigns the 3-class label to every row via `vol_regime_class_for`. Rows whose forward-vol is missing fall out of the kept set.
- Constructs `AutoModelForSequenceClassification.from_pretrained(encoder_alias, num_labels=3)` against the canonical classifier-role encoder per ADR 0019. After the 2026-05-30 addendum re-pointed `role: classifier` off the unproduced `finbert_fomc_only` placeholder, this resolves to `finbert_fed_adjacent`; the alias fallback remains `finbert_fed_adjacent`. Operator overrides via `--encoder-alias`.
- Fine-tunes for 5 epochs at AdamW `lr=2e-5`, `weight_decay=0.01`, `train_batch_size=16`, `max_length=256`.
- Evaluates on the test slice; emits the per-(seed, fold) `classification_breakdown` block via `compute_classification_breakdown` so the result drops into the existing §6 tier table without bespoke aggregation.
- Writes a sweep artefact at `backend/artifacts/experiments/finetune_pilot_b2.json` whose schema mirrors `dual_head_comparison_canonical.json` (per-trial cells with `metrics`, per-fold `tertile_cutoffs`, summary block with macro-F1 mean + std + block-bootstrap CI).

The canonical `(11, 29, 47, 71, 97)` seed set × 4 folds = 20 cells per sweep.

## Hyperparameters

| Knob | Value | Reason |
| --- | --- | --- |
| Optimiser | AdamW | Standard for BERT-family fine-tune; consistent with the `transformers<5` import surface. |
| Learning rate | 2e-5 | Canonical AdamW LR for BERT-family fine-tune on small classification corpora (Devlin et al. 2018; same LR `finetune_pilot.py` uses for the stance head). Smaller (1e-5) underfits at 5 epochs; larger (5e-5) destabilises a 110M-param encoder on a 4 100-event train slice. |
| Weight decay | 0.01 | Same value `finetune_pilot.py` and `train_text_multi_axis_classifier.py` use. |
| Epochs | 5 | Long enough for the encoder to see every train row five times; short enough that the run finishes in ~30 min per cell on an A100, keeping 5-seed × 4-fold under 10 GPU-hours. The stance-head fine-tune runs for 3 epochs on a 4 800-row corpus and converges; 5 is a margin for the harder 3-class target. |
| Train batch size | 16 | Fits a 24 GB A100 with `max_length=256` on a 110M-param backbone. Larger (32, 64) is an operator override. |
| Eval batch size | 32 | Inference-only memory; doubles train batch. |
| Max tokens | 256 | Same as `finetune_pilot.py`. FOMC statements run 1 500-3 000 chars (~200-400 BERT-WPS tokens); 256 covers the head of the document including the policy-decision paragraph that carries the regime signal. Longer (512) doubles memory and runtime for marginal context, deferred. |
| LR scheduler | none | Constant LR over 5 epochs is the canonical Devlin recipe; warmup-linear-decay is a downstream knob via `--lr-scheduler` if the constant-LR sweep is unstable. |

## Per-fold tertile cutoffs

The 3-class vol-regime label is fitted per fold on the TRAIN slice only via `fit_vol_regime_quantiles`. The same helper backs the LSTM-on-frozen-embeddings stack and the dual-head canonical comparison; reusing it guarantees the labels B2 fits against are byte-identical to the labels the §6 tier table reports. Contract:

- `cutoffs = fit_vol_regime_quantiles(train_forward_vols, n_classes=3)` returns `(q33, q67)` on the TRAIN slice.
- `class_idx = vol_regime_class_for(test_forward_vol, cutoffs)` labels every test row against the cutoffs the TRAIN slice fitted. The test slice never influences the cutoffs.
- Rows whose forward-vol is missing / non-finite (`vol_regime_class_for` returns `-1`) fall out of the kept set so the labelling pass never silently coerces a missing target to class 0 (calm).

`test_build_partition_targets_fits_cutoffs_on_train_slice_only` pins it: a train slice spanning `[0.0, 0.06]` and a test slice spanning `[0.50, 0.60]`, asserts train + test cutoffs are byte-identical, and asserts every test row lands in class 2 (high) because the train slice never saw a value that high. If the cutoffs leaked the test slice the upper boundary would shift and the assertion would fail.

## Encoder + scope

The default encoder alias resolves via `resolve_by_role("classifier")` per ADR 0019, returning `finbert_fed_adjacent` against the current `registry.yaml` (the 2026-05-30 addendum moved the role tag off the unproduced `finbert_fomc_only` placeholder onto the produced FinBERT-BIS DAPT substrate). Operator overrides via `--encoder-alias` for sibling runs against `BAAI/bge-large-en-v1.5`, `finbert_fed_adjacent_xbank_dapt`, or any other registered encoder. The alias fallback (also `finbert_fed_adjacent`) mirrors `train_text_multi_axis_classifier.py::DEFAULT_ENCODER_ALIAS` so the harness inherits the same unpinned-local guard the other classification entrypoints have.

The harness does not touch the LSTM-on-frozen-embeddings path. `make canonical-comparison` stays byte-identical to pre-#213; both the embedding cache and `app.train_forecaster` are unchanged. The §6 tier table populated by the canonical comparison is unaffected; the B2 row is its own placeholder pending the GPU sweep.

The sweep artefact lives at `backend/artifacts/experiments/finetune_pilot_b2.json`. Top-level keys: `pipeline`, `training_package_id`, `encoder_alias`, `seeds`, `fold_ids`, `epochs`, `train_batch_size`, `learning_rate`, `weight_decay`, `max_length`, `n_classes`, `labels`, `started_at_utc`, `trials`, `summary`. `trials` is a list keyed by seed; each entry carries a `folds` list with `metrics.regime_f1_macro`, per-class `classification_breakdown`, tertile cutoffs, and class-count distributions. `summary` carries `regime_f1_macro` (mean / std / min / max / n across 20 cells) plus `regime_f1_macro_ci` (block-bootstrap 95% CI, block size = fold count). The schema mirrors `dual_head_comparison_canonical.json` so the §6 tier table reads B2 + the canonical-comparison numbers off the same aggregator code.

The §6.6 row lands as a placeholder:

| Tier | Channels | Macro-F1 (n=20) | Δ vs Tier 2 |
| --- | --- | ---: | ---: |
| B2 | end-to-end fine-tune | _pending Runpod sweep_ | _pending_ |

The §6 prose frames the row as a diagnostic against the encoder-freeze-is-bottleneck hypothesis, not a horse-race entry.

## Why not the alternatives

Layer the end-to-end fine-tune into `app.train_forecaster`. The existing loop builds a sequence-of-bars `FeatureVector` per supervised event, pools the encoder output once per row, and concatenates with the rich market block. Wiring an end-to-end fine-tune through that path would require either recomputing the encoder forward on every batch (~10× memory + runtime) or gradient-checkpointing across the mean-pool, which is a substantial refactor. A standalone harness on a separate JSON artefact is cheaper and keeps the canonical-comparison byte-identical.

LoRA fine-tune over the same harness. LoRA is a parameter-efficient compromise; the question is whether end-to-end fine-tuning lifts the headline at all, and LoRA leaves the encoder frozen modulo a low-rank correction. A future PR can layer LoRA on top of the same harness via `--use-lora`; this PR ships the full-fine-tune diagnostic so the LoRA delta has a baseline to read against.

Multi-task head (stance + vol-regime). Bundle A.2 in #228 already established the substitute-not-complement reading of stance vs vol-regime. Conflating the two axes in B2 would muddy the architectural-question framing. A single-task vol-regime fine-tune is the cleanest control for the encoder-freeze diagnostic.

Run the full sweep in CI. ~30 min per cell on an A100 = ~10 GPU-hours for the full sweep, far past the CI envelope. CI runs the 1-epoch synthetic-fixture smoke (under 60 s on CPU); the full sweep is a Runpod follow-up.

## Downstream effects

The §6.6 prose around the B2 row frames it: "B2 tests whether the LSTM detour is the bottleneck. The current §6 tier table's Tier 3 / Tier 5 / Tier 6 / Tier 7 cells freeze the encoder; B2 unfreezes it. The cell is a diagnostic, not a horse-race entry."

The per-fold tertile contract is the methodology lock. The labels B2 fits against are byte-identical to the labels the §6 tier table reports, and the same `fit_vol_regime_quantiles` helper backs both paths. The unit test pins the train-only contract.

Per-cell budget ~30 min on an A100; full sweep ~10 GPU-hours, tractable on a single Runpod A100 pod overnight. CI smoke stays under 60 s on CPU via the tiny-random-bert stub at 1 epoch over a 6-row synthetic fixture. The canonical sweep against `make finetune-pilot-b2 TRAINING_PACKAGE_ID=<id>` is the Runpod hand-off; default override surface: `ENCODER_ALIAS=BAAI/bge-large-en-v1.5` or `ENCODER_ALIAS=finbert_fed_adjacent`.

The harness disables `torch.compile` via `app.training.runtime_compat.ensure_compile_safe` for the same reason the canonical-comparison runner does: the pod's triton install mismatches the runtime and TorchDynamo crashes on a clean import. `TORCHDYNAMO_DISABLE=1` is set at runner entry.

## References

- `backend/app/data/finetune_pilot_b2.py`, `finetune_pilot.py` (Phase 3 layout precedent)
- `backend/app/training/loaders.py::fit_vol_regime_quantiles`, `vol_regime_class_for`
- `backend/app/evaluation/classification_breakdown.py::compute_classification_breakdown`
- `backend/artifacts/experiments/dual_head_comparison_canonical.json` (schema precedent)
- ADR 0019 (encoder split), ADR 0023 (the LSTM stack this harness bypasses)
- Issues #213, #228, #330; Devlin et al. (2018)
