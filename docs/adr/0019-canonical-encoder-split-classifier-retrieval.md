# ADR 0019 — Canonical encoder split: classifier substrate vs retrieval substrate

Status: accepted, in production (as of merge).
Date: 2026-05-27.
Supersedes: the single-canonical-encoder arrangement under which `finbert_fed_adjacent_xbank_dapt` was pinned as both the headline classifier substrate and the retrieval base.
References:
- Issue #330.
- `backend/app/models/registry.yaml` — `role:` tags on the two canonical entries.
- `backend/app/models/registry.py` — `resolve_by_role(role)` + `EncoderRole` literal.
- `backend/app/data/train_text_multi_axis_classifier.py` — classification training entrypoint, picks the encoder via `resolve_by_role("classifier")`.
- `backend/app/retrieval/train.py` — retrieval entrypoint, picks the base encoder via `resolve_by_role("retrieval")`.
- `tests/unit/test_registry_role_resolution.py` — role-resolver contract tests.
- `fed-pulse.wiki/06_Deep_Learning_Roadmap.md §6.7` — canonical headline cell, now keyed off the classifier substrate.

## Context

The DAPT checkpoint `finbert_fed_adjacent_xbank_dapt` was pinned as the sole canonical encoder. It carried both jobs: the headline classifier substrate that the multi-axis classifier fine-tunes against, and the retrieval base that `app.retrieval.train` layers a sentence-transformer head on top of.

Bundle A.2 (cross-bank supervision arms — stance-masked + weighted) and Bundle A.4 (cross-bank DAPT substrate extension) both returned null on the vol-regime headline target. The substitute-not-complement diagnosis under §16 framed the cross-bank corpus as a substrate rather than a target-side complement. The justification for keeping the cross-bank DAPT encoder canonical was the downstream retrieval task — same-meeting positive pairs that span statement / minutes / press_conference rows benefit from a multilingual-fed substrate — not the headline classifier.

One substrate doing two jobs poorly. The classifier picked a substrate that has no measured lift on its own target; the retrieval task picked a substrate that was justified on the wrong axis.

## Decision

Split the canonical encoder slot into two role-tagged entries. Both ship in `backend/app/models/registry.yaml`:

- **Classifier substrate (`role: classifier`):** `finbert_fomc_only` — the pre-cross-bank FOMC-only DAPT encoder. Owns the headline classification training contract.
- **Retrieval substrate (`role: retrieval`):** `finbert_fed_adjacent_xbank_dapt` — the cross-bank DAPT encoder. Continues as the retrieval base; the contrastive head trains on top of it under `app.retrieval.train`.

The registry loader (`load_registry`) reads the new `role:` field into `EncoderRef.role` (defaults to `None` for every untagged entry — bake-off siblings, control ablations, placeholder rows). A new `resolve_by_role(role: Literal["classifier", "retrieval"]) -> str` function returns the first registered alias whose role matches; an unknown role label raises `KeyError` at the boundary.

The two training entrypoints route through the role resolver with a hard-coded fallback that mirrors the pre-#330 default:

- `backend/app/data/train_text_multi_axis_classifier.py` resolves `DEFAULT_ENCODER_ALIAS` via `resolve_by_role("classifier")`, falling back to `"finbert_fed_adjacent"` if the registry has no classifier tag.
- `backend/app/retrieval/train.py` resolves `DEFAULT_BASE_ENCODER_ALIAS` via `resolve_by_role("retrieval")`, falling back to `"finbert_fed_adjacent_xbank_dapt"`.

Back-compat: callers that resolve by alias (the previous default — `encoder_ref(alias)`, `revision_for(alias)`, etc.) keep working unchanged. The new `role:` field is additive; registries written before this ADR continue to load with every `EncoderRef.role` at `None`.

## Consequences

- The canonical headline cell in `fed-pulse.wiki/06_Deep_Learning_Roadmap.md §6.7` is now keyed off the classifier substrate. The cross-bank cell that previously held the headline stays in §6.7 for historical record, annotated as deprecated-as-canonical. The §6.7 re-run against the classifier substrate is filed as a follow-up — placeholder row pinned until the next 5-seed × 4-fold sweep lands.
- The retrieval pipeline (`/analyze/analogs`) keeps the cross-bank DAPT encoder as its base. The retrieval bundle on HF Hub (`yusufizzetmurat/fed-pulse-retrieval`) does not need a re-push; the artefact pin in `registry.yaml` `artefacts:` block stays unchanged because the retrieval base is the same encoder.
- The `EncoderRef` dataclass grows one optional field (`role: str | None`). Existing callers that destructure `EncoderRef` by position would break if the dataclass were positional, but every callsite uses keyword access (`ref.repo`, `ref.alias`, etc.), so the addition is non-breaking.
- Future role additions (e.g. `role: rerank`, `role: nsp_aux`) drop into the `KNOWN_ROLES` tuple without further plumbing. The `Literal` type and the runtime tuple share one source of truth, so a typo at the callsite fails type-check and runtime symmetrically.
- The `forecaster.py` singleton path does not consume the new role tag — `/analyze` loads a serving checkpoint by file path (`backend/models/forecaster_best.pt`) rather than by registry alias. Role tagging is a training-time contract; it does not change the serving entrypoint surface.
- Wiki `12_Architecture_Decision_Records.md` index does not list this ADR yet; the index update is filed as a follow-up to keep this PR's wiki diff narrow.
