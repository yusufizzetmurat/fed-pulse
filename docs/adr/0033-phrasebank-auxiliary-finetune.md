# ADR 0033 — PhraseBank as a supervised auxiliary task

Issue #33 originally framed PhraseBank as a continued-pretraining substrate. We rejected that on size grounds: 4,840 PhraseBank sentences against 909k BIS NSP pairs is 0.5% extra data, and the expected stance-task lift is under 0.005 macro-F1 — below the per-cell std on our canonical sweep.

Path B uses PhraseBank's labels instead. The B2 harness (ADR 0031) already fine-tunes the encoder end-to-end on FOMC regime labels. We bolt a second 3-way classification head on the shared encoder, supervise it with PhraseBank sentiment, and add its cross-entropy to the main loss with a lambda multiplier. The encoder body sees both gradients; the heads stay independent.

## What landed

`backend/app/data/finetune_pilot_b2.py` gets `--enable-phrasebank-aux` and `--phrasebank-aux-lambda` (default 0.3, matching #273's per-axis lambdas). Loader is `backend/app/data/phrasebank.py`, reads from a parquet cache under `data/external/phrasebank/<subset>__<rev>.parquet`, falls back to `datasets.load_dataset("takala/financial_phrasebank", ...)`, and exposes a JSONL fixture path for tests so CI doesn't hit the network.

Aux head is `nn.Linear(hidden_size, 3)` over the encoder's pooled output. The aux DataLoader is zipped one-for-one with the FOMC loader via a manual `_cycle(...)`. Loss is `main_ce + lambda * aux_ce`. The aux pool drives no extra epochs; the FOMC fold's epoch count is still the budget.

Default off is byte-identical: `enable_phrasebank_aux=False` skips the aux head, the second DataLoader, and the combined-loss branch entirely. The metrics dict reports `phrasebank_aux_lambda=0.0`, `phrasebank_aux_rows=0`, `phrasebank_aux_train_loss=None`, and the per-fold artefact row omits the `phrasebank_aux` block. The top-level `phrasebank_aux` meta is `{"enabled": false}`.

`--enable-phrasebank-aux` with `lambda <= 0` is treated as aux-disabled and warns at startup (#429 — would have silently zeroed every aux gradient otherwise).

The FOMC fold split is untouched. PhraseBank rows are loaded once at sweep start and shared across every (seed, fold) cell; the aux loader indexes a disjoint pool, so auxiliary rows can never bleed into a fold's test slice.

## Label-space caveat

PhraseBank labels company-news sentiment (positive/negative/neutral). The B2 primary task labels FOMC vol-regime (calm/normal/high). Same arity, no semantic alignment. The theory of operation is intermediate-task fine-tuning (Phang et al. 2018; Pruksachatkun et al. 2020): the auxiliary task pushes the encoder onto a finance-domain manifold that transfers to the primary task — the aux *labels* themselves aren't supposed to predict regime.

The literature reports +0.01 to +0.02 macro-F1 on small downstream pools under this setup. That's the band we expect. If we land inside it the methodology contribution is the diagnostic itself; if we land outside it the result is still publishable as a negative finding on the label-space mismatch.

## Acceptance

Once the GPU sweep runs (Runpod, operator-driven), `backend/artifacts/experiments/finetune_pilot_b2_phrasebank.json` carries the PhraseBank-augmented row. We compare it against the existing B2 baseline on the same fold surface (same training package, same 5 seeds, same fold manifest, same encoder alias — `finbert_fed_adjacent` per ADR 0019). Lift inside the +0.01 to +0.02 band → in-domain auxiliary supervision helps; below band → label mismatch caps the benefit; negative → the aux objective competes with the primary on this corpus size.

The CI smoke (`tests/unit/test_finetune_pilot_b2_phrasebank.py`) only covers the wire-up: loss flows, gradients reach both heads through the encoder body, default-off is unchanged, the lambda<=0 guard fires.

## Cost note

The aux path roughly doubles per-step runtime on the encoder (one extra forward + backward through the shared body). 5-seed × 4-fold × 5-epoch sweep is ~20 GPU-hours with aux on, ~10 without. Budget accordingly. Sweeping lambda across {0.1, 0.3, 0.5, 1.0} to isolate the knob is a follow-up; 0.3 is the starting point.

## References

- `backend/app/data/phrasebank.py`, `backend/app/data/finetune_pilot_b2.py`
- ADR 0019 (encoder split), ADR 0031 (B2 harness)
- Malo et al. (2014), *Good debt or bad debt: detecting semantic orientations in economic texts*
- `takala/financial_phrasebank` on HF
