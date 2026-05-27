# ADR 0018 — Multi-task auxiliary loss null finding + factor-axis disposition

Status: accepted, in production (as of merge).
Date: 2026-05-27.
References:
- Issue #328.
- PR #282 — wired `MultiTaskLoss` through the regime classifier (the joint training pass this ADR concludes).
- ADR 0009 — multi-axis label set the factor axis lives on.
- `backend/app/training/loss.py` — `MultiTaskLoss` (kept; no behavioural change).
- `backend/app/data/train_text_multi_axis_classifier.py` — text-axis classifier trainer; now persists per-checkpoint factor-axis coverage onto `training_args.factor_coverage`.
- `backend/app/services/multi_axis_classifier.py` — inference service; gates the `factor` card on the persisted coverage.
- `backend/app/schemas.py` — `MultiAxisBlock.factor` (already typed `Optional` since #78).
- `06_Deep_Learning_Roadmap.md` §6.7 (joint-vs-stance-only comparison row) and §6.13 (null-result framing).

## Context

Two related questions about the multi-task surface had to be settled together.

**1. Did the joint training pass lift the regime headline?**

PR #282 ran the canonical Transformer cell with `multi_task_loss=True` against the strict-forward training package. Same 5 seeds × 4 folds protocol, default lambdas (λ_stance=1.0, λ_factor=λ_certainty=λ_topic=0.3).

| Variant | Macro-F1 (n=20) | 95% CI | Std |
| --- | ---: | --- | ---: |
| Stance-only baseline | 0.4308 | [0.419, 0.443] | 0.037 |
| Multi-task (λ_aux = 0.3 each) | 0.4371 | [0.421, 0.459] | 0.064 |
| Δ | +0.0063 | overlapping | +0.027 (worse) |

Point estimate moved up 1.5 %. The CIs overlap heavily, std grew ~70 % (0.037 → 0.064), and three of four folds gained but wf_fold_1 dropped 0.057. The auxiliary axes are pulling the backbone in inconsistent directions across folds at the default lambda balance. The follow-up "rescue" options were filed (lambda sweep, two-stage warm-up, per-fold λ fitting) but the wiki framed the wiring as a "foundation for cheap follow-ups" rather than as a result, which overclaims.

**2. Why does the factor card on `/analyze` look like noise?**

The text-axis classifier ships a four-branch head (stance / factor / certainty / topic). On the canonical training package the supervised pool used for the text classifier has 0 % `axis_factor` coverage — the `gss_factor` source rows are not joined into the events.parquet aggregation, and the gtfintechlab cross-bank rows do not carry a factor label by schema. The factor branch trains almost exclusively on the masked-out path (mask=False everywhere, loss contribution always zero) and emits effectively-random tanh-bounded values at inference. One of the four `/analyze` multi-axis cards was rendering noise dressed as a prediction.

## Decision

### Multi-task null framing

Report the joint-vs-stance-only result as a documented null. The auxiliary axes (factor / certainty / topic) do not carry information the encoder is not already extracting from stance on the vol-regime target — the 1.5 % point-estimate move is inside the CI, the variance penalty is real, and the per-fold variance increase rules out a "just-needs-tuning" framing without further evidence. Stop tuning lambdas hoping for a lift on the headline; the multi-task wiring stays in the codebase because the per-axis output surface (factor / certainty / topic cards on `/analyze`) consumes it, but it is no longer pitched as a regime-headline lever.

Concretely:
- The §6.7 architecture-sweep section keeps the joint-vs-stance-only row (already landed) but the prose around it now reads as an explicit null rather than a "first pass" framing.
- §6.13 ("Multi-task auxiliary-loss foundation") is rewritten to lead with the null finding (CIs overlap, std grew 70 %) rather than the deferred-joint-training rationale.
- Future lambda-tuning work is descoped — the filed follow-ups (lambda sweep, two-stage warm-up) stay listed for completeness but are not on the critical path.

### Factor-axis disposition — option (a), don't mount

Two candidate paths were on the table:

- **(a) Don't mount the head when factor coverage = 0 %.** The inference service reads the persisted `factor_coverage` off the active checkpoint and omits the factor card from the `/analyze` response when coverage falls below a threshold. `MultiAxisBlock.factor` is already `Optional` (it has been since #78), so wire compat is free. Estimated cost: ~30 lines on the service + ~10 lines on the trainer to stamp the field.
- **(b) Backfill from gss_factor source rows.** Pull the `axis_factor` regression target off the `gss_factor` source rows into the supervised text classifier's training pool. Estimated cost: changes in `app.data.ingest_sources`, the events-parquet aggregator, and the classifier's row loader; requires re-running the canonical text-classifier checkpoint to consume the backfilled rows. Days of work + a checkpoint refresh.

**Picked (a).** The cost gap is large (hours vs days), the user-facing outcome of (a) — honest absence on the card — is what the brief flagged as "cleanest from a no-noise standpoint", and (b) does not retire the gate anyway: even with a backfill, a future training pool change that drops factor coverage back to zero would put the surface in the same noise-rendering state. The gate is the right contract regardless; (b) is the orthogonal data-pipeline change that lifts the gate when the coverage is real, and stays available as a follow-up issue.

### Implementation surface

- `backend/app/data/train_text_multi_axis_classifier.py`: the trainer computes `factor_coverage = populated_rows / total_rows` on the train slice (the supervision the head actually trained on, not the whole corpus) and stamps it on the checkpoint envelope under `training_args.factor_coverage`. The log line carries the coverage so the operator sees it inline with `checkpoint_written`.
- `backend/app/services/multi_axis_classifier.py`: `_ClassifierState` carries the coverage stamp + the gate threshold; `score_text` calls `_build_factor_card` which returns `None` when `coverage < threshold` (default 0.01 = 1 %). The threshold is overridable via `FED_PULSE_TEXT_MULTI_AXIS_FACTOR_GATE` so a future operator can tune without a code change. Pre-#328 checkpoints (no stamp on the payload) are treated as unknown → card stays absent, consistent with the new default.
- `backend/app/schemas.py`: `MultiAxisBlock.factor: MultiAxisFactorCard | None = None` was already in the schema since #78, so existing OpenAPI consumers do not see a contract change. The `factor` key now reliably surfaces as `None` on a 0 %-coverage checkpoint rather than as a noise card.
- `tests/unit/test_factor_axis_gating.py`: pins the gate end-to-end on the service surface — 0 % coverage drops the card, above-threshold coverage emits a real value, pre-#328 payloads (no stamp) drop the card, the env override knob works, the trainer's coverage helper matches the service's gate semantics.

## Consequences

- The `/analyze` response no longer renders a factor card on the canonical checkpoint. The frontend already handles `factor: null` (the schema has been `Optional` since #78), so no frontend change is required.
- The multi-task wiring stays in the codebase. It still powers the certainty / topic cards on `/analyze` (those axes have ≥ 35 % label coverage on the supervised corpus today), and the joint-loss code path is what those cards' classifier learned on.
- A future operator who backfills the `gss_factor` source rows into the supervised pool will see coverage climb above the 0.01 gate automatically; no code change required.
- A future operator who decides 0.01 is too lax / too strict can override via the env knob without redeploying — the gate is the contract, not the literal threshold.
- Documentation in `06_Deep_Learning_Roadmap.md` §6.7 and §6.13 carries the null finding explicitly. Wiki §12 (ADR index) is intentionally not touched in this PR; ADR 0018 sits in `docs/adr/` alongside the other accepted ADRs and the §12 index is updated in a separate change.
