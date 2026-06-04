# ADR 0044 — Drop the topic axis from the multi-axis surface (no upstream source ships topic labels)

Status: accepted, in production (as of merge).
Date: 2026-05-30.
Supersedes parts of: ADR 0009 (multi-axis label schema), ADR 0010 (multi-task head replaces single classification head), ADR 0018 (multi-task null + factor-axis disposition).
References:
- Audit on the rebuilt training package `tp_v3_full_rebuild_2026_05_30` (2026-05-30): `axis_topic` populated on 0 / 7172 rows.
- `MULTI_TASK_TOPIC_LABELS` consumer scan (2026-05-30): every populated `axis_topic` value historically traced back to the internal macro-release augmentation (`scripts/build_macro_augmented_registry.py`), which emitted `axis_topic = "economic_indicator"` → `macro` via `_TOPIC_ALIASES`.
- `gtfintechlab/federal_reserve_system` schema audit (2026-05-30): per-row columns are `sentences, stance_label, time_label, certain_label, year` — no topic field. Every other gtfintechlab cross-bank corpus mirrors the same schema.

## Context

The multi-task head shipped under #78 (ADR 0010) carries four output branches: stance, factor, certainty, topic. The topic branch was sized for four classes (`macro`, `forward_guidance`, `market_reaction`, `other`) on the assumption that an upstream Fed-text corpus would ship topic-style labels suitable for masked training.

That upstream never arrived. Every ingested corpus — gtfintechlab/federal_reserve_system, the cross-bank ECB/BoJ/BoE/BoC/RBA siblings, Op-Fed, Swanson, GSS, vtasca, the Kaggle Fed statements/minutes mirror, the scraped federalreserve.gov archive — emits stance / certainty / factor labels but no topic taxonomy. The only path that ever populated `axis_topic` was an in-tree augmentation that synthesised macro-release event rows from CPI / NFP releases and emitted `axis_topic = "economic_indicator"` → `macro`. That augmentation only ever populated one of the four topic classes (the other three never saw a single labelled example), and the rebuild path in `docs/training-package-rebuild.md` no longer fires it (the macro_release event_kind is absent from the rebuilt TP).

The audit on `tp_v3_full_rebuild_2026_05_30` reported 0 / 7172 rows populated on `axis_topic`. The trainer's masked loss therefore contributed exactly zero gradient on the topic branch on every batch of every fold; the inference card always rendered the same low-confidence fallback because the head was trained on zero positives.

The earlier framing under ADR 0018 ("multi-task null + factor-axis disposition") already concluded the topic axis carried no information gain over stance. This ADR follows that conclusion to its logical end: a head that trains on zero labels and predicts no signal is dead code in every meaningful sense.

## Decision

Remove the topic axis from the project's data + training + inference + presentation surface. Concretely:

**Schema.** `axis_topic` column dropped from `EventRowSchema` and `NormalizedDocSchema` in `backend/app/data/schemas.py`. The `axes` dict no longer carries a `topic` key (`_axes_dict_ok` validator updated). The `_axes_topic_ok` validator is removed entirely.

**Data pipeline.** `event_dataset_builder.COLUMN_ORDER` no longer lists `axis_topic`. The row-emission dict no longer writes a `topic` column. `_EventDoc.multi_axis` no longer carries a `topic` key; the axis-name iteration loop drops `topic`. `normalize_labels._normalize_one_row` no longer extracts `topic_value` from `axis_topic` / `multi_axis_extras`.

**Training surface.** `MULTI_TASK_TOPIC_CLASSES` and `MULTI_TASK_TOPIC_LABELS` removed from `app.models.config`. `multi_task_lambda_topic` field removed from `ModelConfig` (resume path drops the matching `getattr` lookup). `MultiTaskHead` no longer constructs the topic linear layer; `forward` returns a 3-key dict. `TextMultiAxisClassifier` no longer takes a `topic_classes` kwarg and no longer surfaces a `topic_classes` field on its `metadata()` payload. `MultiTaskLoss` no longer accepts `topic_weight` / `lambda_topic` and no longer computes `topic_loss`. `_topic_target` / `_TOPIC_ALIASES` removed from `train_text_multi_axis_classifier`. Per-axis loops over `("stance", "factor", "certainty", "topic")` collapse to `("stance", "factor", "certainty")`; per-axis class-weight + loss-breakdown dicts drop the `topic` key.

**Aux block on the loader.** `_MULTI_TASK_AUX_KEYS` collapses from six tensors (factor / factor_mask / certainty / certainty_mask / topic / topic_mask) to four. The DataLoader arity dispatch in `_unpack_batch` retables from `{2, 3, 4, 5, 8, 9, 10, 11}` to `{2, 3, 4, 5, 6, 7, 8, 9}`; the unsupported-arity test re-pins against arity 10 so the negative-path coverage stays.

**Loaders.** `target_topic_idx` and `target_topic_present` fields removed from `FeatureVector`. `_attach_rich_features` no longer parses `axis_topic` and no longer writes the topic targets. `_build_multi_task_target_tensors` no longer accumulates `topic_targets` / `topic_masks` and the returned dict drops the `topic` / `topic_mask` keys.

**Inference.** `app.services.multi_axis_classifier.score_text` no longer computes `topic_probs` and no longer emits a topic card. `app.main._build_multi_axis_block` drops the `"topic": None` fallback dict entry. `MultiAxisAnalysisCard.topic` and `MultiAxisTopicCard` removed from `app.schemas`.

**Retrieval.** `SHARED_AXIS_COLUMNS` on `app.retrieval.train` collapses from `("axis_stance", "axis_factor", "axis_topic")` to `("axis_stance", "axis_factor")`.

**Frontend.** `MultiAxisTopic` interface, `TopicAxis` type, and the `MultiAxisResponse.topic` field removed from `frontend/lib/analyze/types.ts`. `TopicCard` / `TopicTile` components and their conditional render guards removed from `MultiAxisCards.tsx` and `MultiAxisInterpretation.tsx`. The 4-column grid on `MultiAxisCards` collapses to 3 columns. `multiAxisSummary` no longer surfaces a topic clause. PDF and CSV export rows lose their `multi_axis.topic.primary` entries. `MultiAxisDelta.topicChanged` and the `topicA / topicB` comparator branch removed from `frontend/lib/analyze/compare.ts`.

**Tests.** 26 test fixtures across `tests/unit/` and `tests/regression/` updated: `"axis_topic": …` dict entries dropped, `target_topic_idx` / `target_topic_present` removed from `FeatureVector` constructors, `MULTI_TASK_TOPIC_*` imports removed, axis loops trimmed. The arity tests rebaseline on the new (6, 8) shapes.

## Why this is recorded

Removing a multi-task branch is the kind of change a future reviewer will second-guess unless the audit trail is one click away. The audit numbers are the load-bearing fact: 0 / 7172 rows populated on the rebuilt TP, no upstream corpus shipping topic labels, the only synthetic path collapsing to a single class out of four. The earlier ADR 0018 framing flagged the same gap; this ADR closes it by removing the dead branch rather than carrying it for the rest of the project's lifetime.

The decision is reversible if a future corpus does ship topic labels. The changes are localised and git history retains the prior shapes. The cost of adding the branch back is bounded; the cost of carrying a dead branch indefinitely is unbounded and silently grows.

## Consequences

- The multi-task head is a 3-branch surface instead of 4. The per-axis loss is `lambda_stance * stance_loss + lambda_factor * factor_loss + lambda_certainty * certainty_loss`; no `lambda_topic` term, no `topic_weight` class weights, no fitted weights persisted on the checkpoint payload's `multi_task_class_weights.topic` key.
- Existing checkpoints that carry a topic head will fail strict state-dict load against the new module. This is acceptable: every active checkpoint in scope (canonical, B2, retrieval) is rebuildable from the source corpora, and the rebuild ride after this PR uses the new module shape.
- The API response shape changes (`MultiAxisAnalysisCard.topic` removed). The frontend ships the corresponding change in the same PR, so the live deploy is consistent. Any external client that depended on the topic card now sees no field at all.
- The deferred work in ADR 0021 (retrieval supervision rebuild on shared-axis pairs) loses `axis_topic` as a pair-policy axis. Shared-axis pairs now match on stance OR factor only. The recall@k probe set is small enough that the change is one config-file edit; no encoder retrain required to pick up the policy change.
- ADR 0018 ("multi-task null + factor-axis disposition") is partially superseded: the topic-axis specific items are closed by this ADR. The factor-axis disposition (`MultiAxisFactorCard | None` gated on coverage) remains the live contract for factor.
- ADR 0009 ("multi-axis label schema") becomes historically narrower: the four-axis schema it described is now three-axis. A short addendum on ADR 0009 captures this without rewriting the original framing.
- The `_drop_axis_topic_no_upstream_source` branch in `quality_validation` does not need a special-case: any old events.parquet on disk that carries an `axis_topic` column will load with an unknown column (pandera permissive) or fail validation (pandera strict). Both are acceptable; the resolution is to rebuild from source.
