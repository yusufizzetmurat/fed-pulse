# ADR 0032 — Cross-source transfer matrix + first external-corpus adapter

## Context

Issue #72 calls for `BaseSourceScraper`-compliant adapters for the four obtainable replication-package corpora flagged in the wiki inventory: TDW expansion, GSS factor decomposition, Hansen-McMahon topic shares, Lucca-Trebbi communication index. Issue #83 asks for a cross-source transfer matrix evaluating a FOMC-trained checkpoint on every non-FOMC source stratum (speeches, testimony, press conferences, Beige Book, NY Fed regional research) so the §6 report can carry a generalisation diagnostic.

The two scopes interlock. The cross-source eval needs labelled rows under each `source_type` to score; the external-corpus adapters are the route by which non-statement / non-minutes labelled rows reach the registry. Without at least one adapter the matrix collapses to a single column.

The wiki inventory has aged out three of the original four packages since the issue was filed. Lucca-Trebbi was abandoned 2026-05-14; the paper PDF and methodology appendix carry zero data tables and the series exists only as plotted figures. Hansen-McMahon was blocked 2026-05-14; every public download URL 404s or hits a login wall and reintroduction depends on direct contact with Stephen Hansen. The Swanson factor update was abandoned 2026-05-14; the 2021 JME supplement is a 3-page formula PDF with no per-meeting series. The fourth, GSS, is already in the registry via `_iter_gss_factors_records`, but the rows carry a continuous `factor` axis not the categorical stance axis the cross-source eval scores; wrapping it as a `BaseSourceScraper` is a refactor without a near-term consumer.

Op-Fed (Keith et al. 2025), ingested 2026-05-15 via `_iter_op_fed_records`, does have a near-term consumer. The corpus carries 159 stance-labelled sentences on FOMC meeting transcripts (1977-2008) under MIT licence — labels no other source in the registry covers. The `fomc_meeting_transcript` source_type is in the canonical `_VALID_SOURCE_TYPES` vocabulary but no `BaseSourceScraper` is registered for it. Wrapping Op-Fed closes the first row of the cross-source matrix.

## Decision

Ship two pieces in one PR.

The Op-Fed `BaseSourceScraper` adapter lands at `backend/app/data/sources/op_fed.py`, registered against `source_type="fomc_meeting_transcript"` with `Provenance.PEER_REVIEWED`. The adapter wraps the existing CSV read path so the source registry advertises the same metadata / contract surface as the in-house HTML scrapers. `fetch_listing(html)` reads the entire CSV content (the Protocol's `html` parameter carries the file text); `parse_entry(json_row, source_url)` deserialises one row and emits the registry-record dict; `write(parsed, output_path)` serialises to JSONL. This is the first file-backed `BaseSourceScraper`, and the pattern is reusable for the deferred GSS / Hansen-McMahon / Swanson adapters once data materialises.

The cross-source transfer harness lands at `backend/app/evaluation/cross_source_transfer.py`. Given a training package's `registry_normalized.jsonl` and `{alias: checkpoint_path}` map, the harness:

- Loads labelled rows filtered to the canonical `CROSS_SOURCE_TYPES` strata (FOMC statements / minutes / meeting transcripts / press conferences, chair / governor speeches, congressional testimony, Beige Book, regional research, NY Fed Liberty Street).
- Drops rows with `sample_weight == 0` by default so cross-bank pretrain rows and unlabelled archive rows do not enter the eval set. Cross-bank rides on its own harness; the two stay disjoint.
- Buckets rows by `source_type` and runs HuggingFace inference per bucket via `AutoModelForSequenceClassification` against the canonical TDW label order (dovish, hawkish, neutral) per the `cross_bank_transfer` note.
- Emits per-cell macro-F1 / weighted-F1 / accuracy plus per-class precision / recall / F1 and per-label support (`dovish_n` / `hawkish_n` / `neutral_n`) so under-populated cells stay visible in the CSV.
- Writes `matrix.json` + `matrix.csv` to `data/artifacts/v2_cross_source/<run_id>/`.

The harness mirrors the `cross_bank_transfer` API surface. `evaluate_source` accepts a `predict_fn` for deterministic unit tests; production code passes `None` to fall through to HF inference. The `build_matrix` aggregator emits explicit `status="no_rows"` cells for source_types the registry carries zero labelled rows for, instead of silently omitting them. An empty cell is itself a finding — the source stratum has no labelled supervision yet — which is the §6 framing the matrix supports.

### Schema differences vs `cross_bank_transfer`

| Aspect | `cross_bank_transfer` | `cross_source_transfer` |
|---|---|---|
| Filter axis | `source` (gtfintechlab dataset id) | `source_type` (canonical FOMC-side vocabulary) |
| Per-cell support | One bank per cell | One source_type per cell |
| Multi-axis slice | time + certainty (gtfintechlab schema) | None — most FOMC-side sources carry no secondary axes |
| Bootstrap CI | Block-bootstrap across ≥2 checkpoints when supplied | Single checkpoint per alias; CI is a follow-up sweep |
| Output filename | `transfer_matrix.{json,csv,md}` | `matrix.{json,csv}` |

The single-checkpoint contract is deliberate. The cross-bank harness was extended to multi-checkpoint CI bands after the first round of point estimates; the cross-source harness ships at point-estimate stage to keep the PR scoped. Per-seed CI is a one-line change once the canonical Runpod sweep produces multiple checkpoints per alias.

### Scope marking

§6.25 frames the matrix as "1-of-4 adapters landed." Adapters for GSS, Hansen-McMahon, and Swanson are filed as follow-up issues with explicit "blocked on upstream data" or "refactor candidate" markers. The matrix's column count grows as those follow-ups land; the harness already handles unknown source_types by emitting `no_rows`, so adding the next adapter is purely additive.

The matrix is a generalisation diagnostic, not a leaderboard. A drop in macro-F1 from `fomc_statement` to `chair_speech` is informative regardless of which arm wins. The §6 prose states this explicitly: the FOMC-trained checkpoint is the fixed reference; the question is whether stance labels generalise across the document-genre axis.

### Rejected alternatives

Wrapping GSS first instead of Op-Fed was rejected. GSS already has registry rows and carries `peer_reviewed` provenance, but the rows carry a `factor` axis (continuous) not a `stance` axis (categorical), so they do not score under the existing TDW classification head. The matrix's stance column for GSS would be empty whether the adapter ships or not. Op-Fed contributes labelled stance rows under a source_type no other corpus covers, so it moves the matrix farther forward per unit of adapter work.

Refactoring `BaseSourceScraper` to admit non-HTML adapters was rejected as out-of-scope. Changing the Protocol so `fetch_listing` takes `bytes` / `Path` instead of `html: str` would force a signature update on every existing scraper. The file-backed pattern lands cleanly by passing the file text as the `html` argument and documenting the convention in the adapter's docstring. The Protocol is `runtime_checkable` so `isinstance` validation continues to pass without a base-class change.

Re-training per source_type instead of inference-only was rejected as a different experiment. The cross-source diagnostic asks "does FOMC-stance supervision transfer to genre X?", which is the question §6 needs to answer. Per-source re-training would muddy that signal and double the GPU budget.

Stratifying post-hoc via `source_type_stratified_analysis.py` was rejected as the primary surface. The existing helper already buckets per-row predictions by source_type, but it consumes per-row prediction JSONs emitted by `finetune_batch` / `finetune_pilot`, which couples the cross-source eval to a full fine-tune sweep. The new harness runs inference directly against a checkpoint + registry, so it can be re-run cheaply as new checkpoints land.

## Consequences

§6.25 prose calls out the 1-of-4 status. The matrix carries `fomc_statement`, `fomc_minutes`, `fomc_meeting_transcript`, `fomc_press_conference`, `chair_speech`, `governor_speech`, `congressional_testimony`, `beige_book`, `regional_research`, and `ny_fed_liberty_street` as columns; per-source counts make under-populated cells visible. The Op-Fed adapter populates `fomc_meeting_transcript` exclusively — the other columns are filled (or not) by the in-house HTML scrapers' downstream registry rows.

Inference-only. Each source bucket is a few hundred rows at most; the full matrix against a single A100 checkpoint runs in under 10 minutes. CI smoke uses a stub `predict_fn` and runs under 60 s on CPU.

The canonical sweep against `make cross-source-transfer TRAINING_PACKAGE_ID=<id> ENCODER_CHECKPOINTS=finbert_fed_adjacent=<path>` is a Runpod follow-up. The harness writes `matrix.{json,csv}` under `data/artifacts/v2_cross_source/<run_id>/`; §6.25 populates once the artefact lands.

Follow-ups: GSS as `BaseSourceScraper` (refactor candidate; registry rows on the factor axis, not stance); Hansen-McMahon (blocked on upstream data access — every URL probed 2026-05-14 either 404s or hits a login wall); Swanson factor update (blocked on upstream data; the 2021 JME supplement is methodology-only); multi-checkpoint CI bands in the harness (one-line extension once Runpod produces a per-seed sweep). Each follow-up is filed as its own issue rather than expanded into this PR's scope.

## References

- `backend/app/data/sources/op_fed.py`, `backend/app/data/ingest_sources.py::_iter_op_fed_records`
- `backend/app/evaluation/cross_source_transfer.py`, `cross_bank_transfer.py` (sibling schema)
- `fed-pulse.wiki/13_External_Corpora_Inventory.md`
- ADR 0019, ADR 0023; Issues #72, #83
