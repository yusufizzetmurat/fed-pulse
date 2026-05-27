# ADR 0032 — Cross-source transfer matrix + first external-corpus adapter

Status: accepted, harness code path live; canonical sweep deferred to operator.
Date: 2026-05-28.
References:
- Issue #72 (closes — first replication-package adapter; remaining three filed as follow-ups).
- Issue #83 (closes — cross-source transfer harness).
- ADR 0019 — canonical encoder split (the harness defaults to `resolve_by_role("classifier")` when an operator passes a registry alias).
- ADR 0023 — train/inference contract enforcement (the eval is inference-only, no contract delta).
- `backend/app/data/sources/op_fed.py` — Op-Fed `BaseSourceScraper` adapter.
- `backend/app/evaluation/cross_source_transfer.py` — eval harness.
- `backend/app/evaluation/cross_bank_transfer.py` — sibling cross-CB harness whose schema this one mirrors.
- `backend/app/data/ingest_sources.py::_iter_op_fed_records` — pre-existing read path the adapter wraps.
- `fed-pulse.wiki/13_External_Corpora_Inventory.md` — source-of-truth for the four replication packages flagged in #72.

## Context

Issue #72 calls for `BaseSourceScraper`-compliant adapters for the four obtainable replication-package corpora flagged in the wiki inventory: TDW expansion, GSS factor decomposition, Hansen-McMahon topic shares, Lucca-Trebbi communication index. Issue #83 asks for a cross-source transfer matrix evaluating a FOMC-trained checkpoint on every non-FOMC source stratum (speeches, testimony, press conferences, Beige Book, NY Fed regional research) so the §6 report can carry an honest generalisation diagnostic.

The two scopes interlock. The cross-source eval needs labelled rows under each `source_type` to score; the external-corpus adapters are the route by which non-statement / non-minutes labelled rows reach the registry. Without at least one adapter the matrix collapses to a single column.

The wiki inventory has aged out three of the original four replication packages since the issue was filed:

- **Lucca-Trebbi.** Abandoned 2026-05-14. The paper PDF and methodology appendix together carry zero data tables; the series exists only as plotted figures.
- **Hansen-McMahon.** Blocked 2026-05-14. Every public download URL 404s or hits a login wall; reintroduction depends on direct contact with Stephen Hansen.
- **Swanson factor update.** Abandoned 2026-05-14. The 2021 JME supplement is a 3-page formula PDF with no per-meeting series.

The fourth package — GSS — is already in the registry via `_iter_gss_factors_records`, but the rows carry a continuous `factor` axis not the categorical stance axis the cross-source eval scores. Wrapping it as a `BaseSourceScraper` adapter is a refactor without a near-term consumer.

What does have a near-term consumer is **Op-Fed (Keith et al. 2025)**, ingested 2026-05-15 via `_iter_op_fed_records`. The corpus carries 159 stance-labelled sentences on FOMC meeting transcripts (1977-2008) under MIT licence — labels that no other source in the registry covers. The `fomc_meeting_transcript` source_type is in the canonical `_VALID_SOURCE_TYPES` vocabulary but no `BaseSourceScraper` is registered for it. Wrapping Op-Fed closes the first row of the cross-source matrix.

## Decision

Ship two pieces under one PR:

1. **Op-Fed `BaseSourceScraper` adapter** at `backend/app/data/sources/op_fed.py`, registered against `source_type="fomc_meeting_transcript"` with `Provenance.PEER_REVIEWED`. The adapter wraps the existing CSV read path so the source registry advertises the same metadata / contract surface as the in-house HTML scrapers. `fetch_listing(html)` reads the entire CSV content (the Protocol's `html` parameter carries the file text); `parse_entry(json_row, source_url)` deserialises one row and emits the registry-record dict; `write(parsed, output_path)` serialises to JSONL. This is the first "file-backed" `BaseSourceScraper` and the pattern is reusable for the deferred GSS / Hansen-McMahon / Swanson adapters once data materialises.

2. **Cross-source transfer harness** at `backend/app/evaluation/cross_source_transfer.py`. Given a training package's `registry_normalized.jsonl` and `{alias: checkpoint_path}` map, the harness:
   - Loads labelled rows, filtered to the canonical `CROSS_SOURCE_TYPES` strata (FOMC statements / minutes / meeting transcripts / press conferences, chair / governor speeches, congressional testimony, Beige Book, regional research, NY Fed Liberty Street).
   - Drops rows with `sample_weight == 0` by default so cross-bank pretrain rows and unlabelled archive rows do not enter the eval set. Cross-bank rides on its own harness; the two stay disjoint.
   - Buckets rows by `source_type` and runs HuggingFace inference per bucket via `AutoModelForSequenceClassification` against the canonical TDW label order (dovish, hawkish, neutral) per the `cross_bank_transfer` note.
   - Emits per-cell macro-F1 / weighted-F1 / accuracy plus per-class precision / recall / F1 and per-label support (dovish_n / hawkish_n / neutral_n) so under-populated cells stay visible in the CSV.
   - Writes `matrix.json` + `matrix.csv` to `data/artifacts/v2_cross_source/<run_id>/`.

The harness mirrors the `cross_bank_transfer` API surface — `evaluate_source` accepts a `predict_fn` for deterministic unit tests; production code passes `None` to fall through to the HF inference path. The `build_matrix` aggregator emits explicit `status="no_rows"` cells for source_types the registry carries zero labelled rows for, instead of silently omitting them. This is load-bearing for the §6 framing: an empty cell in the matrix is itself a finding (the source stratum has no labelled supervision yet).

### Schema differences vs `cross_bank_transfer`

| Aspect | `cross_bank_transfer` | `cross_source_transfer` |
|---|---|---|
| Filter axis | `source` (gtfintechlab dataset id) | `source_type` (canonical FOMC-side vocabulary) |
| Per-cell support | One bank per cell | One source_type per cell |
| Multi-axis slice | time + certainty (gtfintechlab schema) | None — most FOMC-side sources carry no secondary axes |
| Bootstrap CI | Block-bootstrap across ≥2 checkpoints when supplied | Single checkpoint per alias; CI is a follow-up sweep |
| Output filename | `transfer_matrix.{json,csv,md}` | `matrix.{json,csv}` |

The single-checkpoint contract is deliberate. The cross-bank harness was extended to multi-checkpoint CI bands after the first round of point estimates; the cross-source harness ships at the point-estimate stage to keep the PR scoped. Adding per-seed CI is a one-line change once the canonical Runpod sweep produces multiple checkpoints per alias.

### Honest scope marking

The §6 subsection (§6.25) frames the matrix as "1-of-4 adapters landed". Adapters for GSS, Hansen-McMahon, and Swanson are filed as follow-up issues with explicit "blocked on upstream data" or "refactor candidate" markers. The cross-source matrix's column count grows as those follow-ups land; the harness already handles unknown source_types by emitting a `no_rows` cell, so adding the next adapter is purely additive.

### What this is not

The cross-source matrix is **not** a horse race against the cross-bank matrix and **not** a leaderboard. It is a generalisation diagnostic. A drop in macro-F1 from `fomc_statement` to `chair_speech` is informative regardless of which arm wins. The §6 prose calls this out explicitly: the FOMC-trained checkpoint is the fixed reference; the question is whether stance labels generalise across the document-genre axis.

## Alternatives considered

**Wrap GSS first instead of Op-Fed.** GSS already has rows in the registry and carries the `peer_reviewed` provenance the issue body asks for. Rejected: the rows carry a `factor` axis (continuous) not a `stance` axis (categorical), so they do not score under the existing TDW classification head. The matrix's stance column for GSS would be empty whether the adapter ships or not. Op-Fed contributes labelled stance rows under a source_type no other corpus covers, so it moves the matrix farther forward per unit of adapter work.

**Refactor the `BaseSourceScraper` Protocol to admit non-HTML adapters.** Change the Protocol so `fetch_listing` takes `bytes` / `Path` instead of `html: str`. Rejected as out-of-scope for this PR: every existing scraper would need a signature update, and the file-backed adapter pattern lands cleanly by passing the file text as the `html` argument and documenting the convention in the adapter's docstring. The Protocol is `runtime_checkable` so `isinstance` validation continues to pass without a base-class change.

**Re-train per source_type instead of inference-only.** Train a separate classifier head per source stratum and report in-domain macro-F1. Rejected: that is a different experiment. The cross-source diagnostic is specifically "does FOMC-stance supervision transfer to genre X?", which is the question the §6 generalisation framing needs to answer. Per-source re-training would muddy that signal and double the GPU budget.

**Stratify post-hoc via `source_type_stratified_analysis`.** The existing `source_type_stratified_analysis.py` already buckets per-row predictions by source_type. Rejected as the primary surface: that helper consumes per-row prediction JSONs emitted by `finetune_batch` / `finetune_pilot`, which couples the cross-source eval to a full fine-tune sweep. The new harness runs inference directly against a checkpoint + registry, so it can be re-run cheaply as new checkpoints land.

## Consequences

### Honest framing per §6

The §6.25 prose calls out that one of the four planned adapters landed. The matrix carries `fomc_statement`, `fomc_minutes`, `fomc_meeting_transcript`, `fomc_press_conference`, `chair_speech`, `governor_speech`, `congressional_testimony`, `beige_book`, `regional_research`, and `ny_fed_liberty_street` as columns; per-source counts make under-populated cells visible. The Op-Fed adapter populates `fomc_meeting_transcript` exclusively — the other columns are filled (or not) by the in-house HTML scrapers' downstream registry rows.

### Compute

Inference-only. Each source bucket is a few hundred rows at most; the full matrix against a single A100 checkpoint runs in under 10 minutes. The CI smoke uses a stub `predict_fn` and runs under 60 s on CPU.

### Sweep hand-off

The canonical sweep against `make cross-source-transfer TRAINING_PACKAGE_ID=<id> ENCODER_CHECKPOINTS=finbert_fed_adjacent=<path>` is a Runpod follow-up. The harness writes `matrix.{json,csv}` under `data/artifacts/v2_cross_source/<run_id>/`; the §6.25 placeholder table populates once the artefact lands.

### Follow-ups

- GSS as `BaseSourceScraper` — refactor candidate; rows already in registry but on the factor axis, not stance.
- Hansen-McMahon — blocked on upstream data access (every URL probed 2026-05-14 either 404s or hits a login wall).
- Swanson factor update — blocked on upstream data; the 2021 JME supplement is methodology-only.
- Multi-checkpoint CI bands in the harness — one-line extension once Runpod produces a per-seed sweep.

Each follow-up is filed as its own issue rather than expanded into this PR's scope.
