# ADR 0009 — Multi-axis stance / factor / certainty / topic labels

Status: accepted, in production.
Date: 2026-05-16.
Supersedes: the previous single-axis ``hawkish | dovish | neutral`` schema (still emitted for backward compatibility).
References:
- `data/schema/labels.yaml` — the canonical label schema.
- `backend/app/data/label_schemas.py` — Python representation.
- `backend/app/data/normalize_labels.py` — mapping from source labels.
- `frontend/components/analyze/MultiAxisCards.tsx` — UI rendering.
- `../../fed-pulse.wiki/06_Deep_Learning_Roadmap.md §"Multi-axis"` — empirical motivation.

## Context

The v0 dashboard collapsed every FOMC excerpt into a single hawkish /
dovish / neutral stance. That works for a headline read but loses three
signals that the reviewer asked the project to expose:

- **Direction** — hawkish vs dovish — is a useful headline but does not
  by itself convey *how much* hawkishness, or *how confidently* the
  language commits to it.
- **Source of the lean** — is the hawkish read driven by inflation
  language or growth language? Two communications can both score
  hawkish-0.8 and respond differently to a CPI surprise.
- **Confidence-of-the-confidence** — a "tentative" hawkish read and a
  "decisive" hawkish read should not aggregate the same way in a
  multi-meeting average.

## Decision

**Adopt a four-axis schema:**

| Axis      | Type                            | Range / values                      |
|-----------|---------------------------------|-------------------------------------|
| stance    | categorical                     | `hawkish` / `dovish` / `neutral`    |
| factor    | continuous signed magnitude     | roughly `[-1, 1]`; sign matches stance, magnitude marks how strongly the language commits |
| certainty | categorical                     | `tentative` / `measured` / `decisive` |
| topic     | categorical (with secondaries)  | `inflation`, `growth`, `employment`, etc. |

Each axis is emitted with a confidence in `[0, 1]`. The schema lives in
``data/schema/labels.yaml`` and is mirrored in
``backend/app/data/label_schemas.py``; renderers consume the typed
``MultiAxisResponse`` defined in ``frontend/lib/analyze/types.ts``.

The single-axis ``stance`` field continues to be emitted at the top
level of the analyze response for backward compatibility with the v0
history table and chip rendering.

## Consequences

- Every NLP baseline / fine-tune / LLM-judge run that lands in the
  reporting pack emits the four-axis labels. Sources that only carry
  ``hawkish | dovish | neutral`` (TDW, kaggle scrape) populate the
  stance axis only; the other three are emitted as missing rather than
  spoofed.
- The compare page in the dashboard renders a per-axis delta column so
  reviewers can see at a glance "A is more decisive about inflation than
  B is."
- Future fine-tunes that learn the multi-axis schema directly (rather
  than mapping from single-axis labels) should produce a measurable
  improvement in factor / certainty. That improvement is the headline
  for the second-half reporting pack and is tracked in
  ``../../fed-pulse.wiki/06_Deep_Learning_Roadmap.md``.
- Label leakage: the multi-axis fine-tune cannot be trained on the
  reporting holdout. The training package builder enforces this at
  fold-construction time.
