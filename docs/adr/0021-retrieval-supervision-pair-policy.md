# ADR 0021 — Retrieval supervision pair-policy + recall@k promotion contract

Status: accepted, in production (as of merge).
Date: 2026-05-27.
References:
- Issue #329.
- ADR 0019 (canonical encoder split: classifier vs retrieval substrate).
- `backend/app/retrieval/train.py` — `--pair-policy {same_meeting, shared_axis}` flag and pair builders.
- `backend/app/retrieval/recall_at_k.py` — `compute_recall_at_k` helper.
- `tests/fixtures/retrieval_recall_at_k.jsonl` — hand-labelled probe set.
- `tests/integration/test_retrieval_recall_at_k.py` — smoke test for the recall@k contract.
- `fed-pulse.wiki/06_Deep_Learning_Roadmap.md §6.16` — rebuild verdict + recall@k numbers.

## Context

The retrieval encoder shipped under #294 was trained with `MultipleNegativesRankingLoss` (MNRL) on same-meeting positive pairs: anchor = FOMC statement, positive = the minutes or press conference released on the same `event_date`. The supervision teaches the encoder to maximise similarity between siblings of one meeting — that is, "same-meeting-ness," not "same-stance-ness."

The downstream product question the `/analyze/analogs` panel answers is cross-meeting semantic similarity: *"When has the Fed previously sounded like this?"* Two statements from different meetings expressing the same stance are negatives under the original recipe. Hand-labelled recall@k probes flagged the resulting encoder as degenerate — same-meeting siblings dominated the top-k and cross-meeting analogs collapsed to date-adjacent rows.

ADR 0019 already split the canonical encoder into a classifier substrate (`finbert_fomc_only`) and a retrieval substrate (`finbert_fed_adjacent_xbank_dapt`). The retrieval substrate's justification was the multilingual-fed pretrain, not the contrastive recipe layered on top. The recipe is what this ADR replaces.

## Decision

Add a `--pair-policy` flag to `backend/app/retrieval/train.py` with two values:

- **`same_meeting` (default).** The pre-#329 builder. Anchors are statements; positives are minutes / press_conference rows on the same `event_date`. Default to preserve byte-identical behaviour for any caller that does not explicitly opt into the rebuild.

- **`shared_axis` (#329 rebuild).** Anchors are statements; positives are statements from a DIFFERENT meeting that share at least one multi-axis label (`axis_stance` / `axis_factor` / `axis_topic`). The first matching axis wins so `positive_kind` is deterministic across reruns. Different-meeting requirement is enforced at the builder layer; same-day matches are filtered out so the policy cannot silently collapse onto same-meeting supervision on uniform-label corpora.

Hard-negative mining stays implicit: MNRL's in-batch contrast treats every other positive in the batch as a negative, so a shuffled batch that mixes axes delivers a different-axis hop for free. The reproducible shuffle (PR #294's seeded generator) carries over unchanged.

The promotion contract: only flip the `role: retrieval` tag in `backend/app/models/registry.yaml` to the new encoder if its recall@k on the hand-labelled probes beats the FinBERT-FedAdjacent cosine baseline at k=1, k=3, AND k=5 (no cherry-picking a single k). If the rebuild does not lift, the baseline encoder stays canonical and the verdict is documented in wiki §6.16; the `--pair-policy shared_axis` recipe stays in the codebase as an opt-in for future re-runs against an expanded probe set.

The recall@k helper at `backend/app/retrieval/recall_at_k.py` is a pure function over numpy arrays (no torch / sentence-transformers / HF dep beyond cosine similarity) so it survives encoder-stack changes and supports stub-based tests on CI.

## Consequences

- `backend/app/retrieval/train.py` now carries one extra CLI flag; existing invocations (smoke runs, GPU-sweep scripts) work unchanged because the default is `same_meeting`.
- The training-args manifest persists the `pair_policy` field so downstream consumers can audit which recipe produced an on-disk encoder bundle. Pre-#329 bundles will deserialise with `pair_policy = None`; consumers must treat the absent field as `same_meeting`.
- `tests/fixtures/retrieval_recall_at_k.jsonl` is a starter (~10 probes). The 30-pair target from #329 is a follow-up — the file header carries `# pragma: stub` so a code-search makes the gap visible. Expanding to 30 is a label-only follow-up; no code change is required to lift the eval signal.
- The recall@k helper is intentionally encoder-agnostic. The integration test stubs the encoder with a deterministic bag-of-keywords function so default CI runs do not pay for an HF Hub round-trip; the heavy-I/O path lives behind `pytest.mark.integration` and is documented in the test docstring.
- The registry `role: retrieval` tag stays pinned to `finbert_fed_adjacent_xbank_dapt` until the recall@k sweep ratifies the rebuild. The wiki §6.16 verdict is the audit trail for the eventual promotion decision; if the rebuild nulls, the same section records the negative result.
- This ADR does not touch wiki `12_Architecture_Decision_Records.md`. The index update is a follow-up so this PR's wiki diff stays narrow.
