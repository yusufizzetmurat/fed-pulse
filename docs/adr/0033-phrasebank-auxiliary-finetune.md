# ADR 0033 — PhraseBank as a supervised auxiliary task on the B2 fine-tune

Status: accepted, harness code path live; canonical sweep deferred to operator.
Date: 2026-05-28.
References:
- Issue #33 (closes).
- ADR 0019 — canonical encoder split (`role: classifier` resolves to `finbert_fomc_only`, fallback `finbert_fed_adjacent`).
- ADR 0031 — B2 end-to-end fine-tune harness (this ADR extends the B2 entry point with an auxiliary-task knob).
- `backend/app/data/phrasebank.py` — PhraseBank loader module.
- `backend/app/data/finetune_pilot_b2.py` — fine-tune harness with the `--enable-phrasebank-aux` flag.
- Malo, P. et al. (2014). *Good debt or bad debt: Detecting semantic orientations in economic texts.* — the PhraseBank dataset.
- `takala/financial_phrasebank` — public HF mirror.

## Context

PhraseBank is a 4 840-sentence financial-sentiment corpus (3-way labels: positive / negative / neutral). The corpus first appeared in this codebase as a candidate continued-pretraining substrate (Path A in the original #33 framing); the 2026-05-23 review rejected that scoping as noise-level. The math: PhraseBank carries ~4 800 sentences against BIS-MLM's 909 877 NSP pairs (two orders of magnitude smaller); even a generous read of the SNR yields an expected lift under 0.005 macro-F1 on downstream stance — below the per-cell std on the canonical sweep (0.070 on fold 4, dual-head comparison). Adding it to the DAPT pool would change nothing measurable.

Path B re-scopes PhraseBank as a *supervised auxiliary task* during the ENCODER fine-tune stage. The B2 harness (ADR 0031, PR #416) already fine-tunes the encoder end-to-end against the FOMC vol-regime target. Layering a second classification head over the same encoder — fed by PhraseBank sentiment labels — gives the encoder an in-domain supervised signal alongside the primary regime task. The auxiliary loss is added to the main loss with a per-axis weight knob (`phrasebank_aux_lambda`); when the knob is zero or the flag is off, the harness is byte-identical to pre-#33 B2.

The two readings that B2 was designed to disambiguate (encoder freeze is the bottleneck vs LSTM detour is the bottleneck) are still in play; this auxiliary-task variant adds a third diagnostic axis:

- *PhraseBank lifts B2 macro-F1 by +0.01 to +0.02.* The auxiliary supervision regularises the encoder away from the vol-regime label noise and toward a financial-sentiment manifold that overlaps the hawkish / dovish stance axis the §6 baselines target. The +0.01 to +0.02 band brackets what the literature on auxiliary classification heads typically shows on small downstream pools (Liu et al. 2019; Phang et al. 2018) — large enough to be visible on the 5-seed × 4-fold sweep CI, small enough not to redefine the headline.
- *No lift / no change.* The label-space mismatch (PhraseBank is *financial sentiment* on company-level news, not *monetary-policy stance* on FOMC text) breaks the auxiliary's transfer. The encoder learns to discriminate sentiment in a context the FOMC pool never reaches, and the gradient on the main task is unchanged.
- *Negative lift.* The auxiliary objective competes with the primary objective in the small-corpus regime; the encoder ends up worse on regime classification because half its capacity is allocated to a task with no downstream value.

All three readings settle whether the in-domain supervised auxiliary signal helps, doesn't help, or hurts. The +0.01 to +0.02 band is the expectation the experiment fronts; the methodology contribution is the diagnostic itself.

## Decision

Add a `--enable-phrasebank-aux` flag (plus `--phrasebank-aux-lambda`, `--phrasebank-subset`, `--phrasebank-cache-root`, `--phrasebank-jsonl`) to `backend/app/data/finetune_pilot_b2.py`. Default off so the existing sweep is reproduced byte-identically. When on, the harness:

- Loads PhraseBank rows via `backend/app/data/phrasebank.py::load_phrasebank_rows`. The loader reads from a local parquet cache under `data/external/phrasebank/<subset>__<rev>.parquet` and falls back to `datasets.load_dataset("takala/financial_phrasebank", subset)` on cache miss. The cache is read-only from this account; no write tokens are exercised. Tests pin a local JSONL fixture via `--phrasebank-jsonl` so the CI smoke is air-gapped.
- Constructs a small linear head (`nn.Linear(hidden_size, 3)`) over the encoder's pooled output alongside the main `AutoModelForSequenceClassification` head. The pooled output comes from `model.base_model(...)` so the aux gradient flows through the shared encoder body.
- Adds the PhraseBank batch to the optimiser step by zipping a second DataLoader (`_cycle(...)`) one-for-one with the FOMC batches. The aux pool drives no extra epochs; the FOMC fold's epoch count is the harness's epoch budget.
- Computes the combined loss `L = main_ce + lambda * aux_ce` where `lambda = --phrasebank-aux-lambda` (default 0.3, matching the per-axis lambdas the LSTM-stage `MultiTaskLoss` ships with from #273).
- Tracks the aux-loss mean per cell + reports it under `phrasebank_aux.train_loss` in the per-fold artefact row. The sweep payload's top-level `phrasebank_aux` meta block carries the subset name, row count, lambda, and per-class counts so downstream consumers can reproduce the auxiliary slice.
- Leaves the FOMC fold split untouched. PhraseBank rows are loaded once and shared across every (seed, fold) cell; the aux loader and the FOMC loader index disjoint pools, so the auxiliary supervision can never bleed into a fold's test slice.

### Default off — byte-identity contract

Default-off path: `enable_phrasebank_aux=False` → `phrasebank_rows is None` → `enable_aux is False` → the harness skips the aux-head construction, the second DataLoader, and the combined-loss code path entirely. The main loss equals `model(...).loss` exactly as in pre-#33 B2. The unit test `test_default_off_metrics_have_zero_aux_fields` locks this; the metrics dict carries `phrasebank_aux_lambda = 0.0`, `phrasebank_aux_rows = 0`, `phrasebank_aux_train_loss = None`.

The sweep artefact's per-fold rows omit the `phrasebank_aux` block when the flag is off — the JSON schema only grows when the operator opts in. The top-level `phrasebank_aux` meta block stays as `{"enabled": false}` on default-off runs.

### Auxiliary loss — additive, lambda-weighted, separate CE

The aux loss is plain cross-entropy on the linear head over the encoder's pooled output, weighted by `lambda` in the combined loss. The lambda default of 0.3 mirrors the `lambda_factor / lambda_certainty / lambda_topic` defaults the LSTM-stage `MultiTaskLoss` (#273) ships with — the canonical lambda for an auxiliary supervision axis that should regularise but not dominate the primary gradient. The headline stance/regime CE is the dominant signal at `lambda=1.0` implicitly (no scaling on the main term).

The auxiliary head is a fresh `nn.Linear` with random init; the encoder forward sees the auxiliary backwards-pass only through the additive loss combination. No shared classifier weights, no logit summing across heads. This keeps the two tasks decoupled at the head level — the aux head learns the PhraseBank label space, the main head learns the regime label space, and the encoder body is the only shared substrate.

### Label-space mismatch — explicit caveat

PhraseBank labels are *financial sentiment* (positive / negative / neutral) on company-level news headlines. The B2 primary task labels are *forward-realised vol regime* (calm / normal / high) on FOMC document text. The two label spaces share three classes by accident; they do not align semantically. The expectation is that the encoder learns a finance-domain representation under the auxiliary signal that *transfers* to the regime task, not that the auxiliary labels themselves predict regime.

This is the same theory of operation that motivates intermediate-task fine-tuning (Phang et al. 2018) and STILTs (Pruksachatkun et al. 2020) — the auxiliary task is the *vehicle*, not the *destination*. The +0.01 to +0.02 lift the experiment fronts is the level at which the literature reports transfer benefit on small downstream pools; deviations from that band are the diagnostic signal.

### Acceptance — tier-table row

A PhraseBank-augmented row sits alongside the existing B2 row in wiki §6.x once the GPU sweep populates `backend/artifacts/experiments/finetune_pilot_b2_phrasebank.json` (operator names the output file via `--output`). The acceptance assertion is whether the +0.01 to +0.02 macro-F1 lift materialises against the existing `finbert_fed_adjacent` B2 baseline on the same fold surface (same seeds, same fold manifest, same encoder alias). The §6 reporting frames the result as a methodology diagnostic: lift → in-domain auxiliary supervision helps the small-corpus encoder fine-tune; no lift → the label-space mismatch caps the benefit at noise; negative lift → the aux objective competes with the primary on this corpus size.

The CI smoke (`tests/unit/test_finetune_pilot_b2_phrasebank.py`) covers the wire-up: loss flows, gradients reach both heads, default-off path is unchanged. The 5-seed × 4-fold sweep against the classifier-role encoder per ADR 0019 is a Runpod follow-up.

## Consequences

- PhraseBank is now a first-class loader in the B2 harness. The same loader can be reused as an auxiliary axis for any downstream fine-tune that wants in-domain financial-sentiment supervision.
- `--phrasebank-aux-lambda` is a per-axis weight knob; sweeping it across `{0.1, 0.3, 0.5, 1.0}` would isolate the lambda where the auxiliary signal is maximally regularising. The default 0.3 is a thesis-defensible starting point; the sweep is operator follow-up.
- The auxiliary path adds one extra forward + backward per step (~2x runtime per cell when aux is on). The 5-seed × 4-fold sweep at AdamW `lr=2e-5`, batch 16, 5 epochs runs ~20 GPU-hours instead of ~10. Operator budgets for this when planning the comparison run.
- The label-space caveat documented above is the load-bearing limitation; the report frames the result honestly regardless of which reading wins.
- HF write tokens are revoked; the loader reads PhraseBank from the public mirror without authentication. If the mirror disappears the loader fails closed (raises `RuntimeError`) rather than silently producing an empty pool.
