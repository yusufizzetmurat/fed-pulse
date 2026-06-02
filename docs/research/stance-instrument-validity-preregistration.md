# Pre-registration — does the stance instrument track what the Fed actually did?

**Date:** 2026-06-02 · **Status:** COMPLETE — VALID but weak & asymmetric (a "hike detector"). See Result.

## Result (2026-06-02)

n=130 meetings 2010–2026 (cut 9 / hold 101 / hike 20).

| test | value | read |
|---|---|---|
| **PRIMARY** Spearman ρ(s, Δff) | **+0.283**, perm-p **0.0006**, CI95 **[0.081, 0.459]** | significant → **VALID**, but *weak* (0.2–0.4) |
| (a) AUC hike-vs-cut | **0.800**, CI95 [0.62, 0.94] | **strong** discrimination at the extremes |
| (b) ordinal ρ(s, {cut<hold<hike}) | +0.276 | monotone but weak |
| (c) leading ρ(sₜ, Δff₊₁) | +0.212, p 0.008 | weak forward alignment (survives Bonferroni) |
| diagnostic: within-holds ρ(sₜ, Δff₊₁) | +0.047 | **no** signalling inside the hold regime |
| mean s by action | cut **−0.84** / hold **−0.82** / hike **+0.03** | the tell ↓ |

**Verdict: the stance instrument is a VALID measure of policy hawkishness
(primary significant, both decision-rule conditions met) — but weak and
asymmetric.** It is essentially a **hike detector**: it separates hikes from cuts
well (AUC 0.80), but **cannot distinguish a hold from a cut** (mean s −0.82 vs
−0.84, nearly identical) and is **dovish-skewed in absolute scale** (even hikes
barely clear neutral). Its validity is carried by the hawkish extreme; the dovish
end is undifferentiated, and it carries no forward guidance within the long hold
regime (within-holds lead ≈ 0).

**What this sharpens (honest descriptive scope):** the dashboard's stance surface
can credibly flag *hiking-hawkish* statements, but should NOT be read as a
fine-grained hawk–dove thermometer on the dovish side, nor imply "more dovish
than the last hold." Concrete instrument-improvement leads: (1) the dovish-end
resolution (hold vs cut) is where the head is blind — a targeted re-label/training
focus there would add the most; (2) the absolute scale is mis-centred (dovish
bias) and would benefit from recalibration. Artifact:
`data/artifacts/stance_instrument_validity/result.json`.

## Motivation (adventure 1 — text as measurement)

Text-as-*prediction* is now closed in both directions (corners A–E, intraday,
late-fusion; reverse market→Fed null). What remains legitimate is text-as-
*description*: the multi-axis stance instrument `s = P(hawk) − P(dove)` powers the
dashboard's descriptive layer. But the project has never **validated** that
instrument against an objective external criterion — the "clean-eval impasse" is
that every stance *label* set is contaminated by training. This sidesteps that
entirely: validate `s` against the Fed's **actual policy action**, which the
classifier never saw. Construct/criterion validity, not label-ranking.

## Anchor (objective, never in training)

Per scheduled FOMC meeting, the realised policy move `Δff_bps` = step in the FRED
`DFEDTARU` funds-rate upper target around the meeting date (value 2 days after −
value 1 day before, ×100). 2010–2026 distribution (n=130): **101 hold, 20 hike,
9 cut** (−50→1, −25→8, 0→101, +25→14, +50→2, +75→4). Categories
`{cut, hold, hike}` from `sign(Δff)`. (The `mp_surprises.parquet` ff_target
fields are degenerate — only 2 non-zero moves — so DFEDTARU is the anchor.)

## Frame and target

One row per scheduled FOMC meeting 2010–2026 (n≈130), `s` read from
`stance_daily.parquet` on the meeting date. Test the relationship between the
text's hawkishness `s` and the concurrent (and next) policy action.

## Tests

- **PRIMARY (concurrent criterion validity):** Spearman ρ(`s`, `Δff_bps`) across
  all meetings. Permutation test (10,000 shuffles of `s`, one-sided H1: ρ>0) for
  p; pairs-bootstrap 95% CI on ρ.
- **SECONDARY (Bonferroni ×3):**
  (a) **Discrimination:** AUC of `s` separating hike (n=20) vs cut (n=9) meetings
      (rank-based = Mann–Whitney U/(n₁n₂)); bootstrap CI.
  (b) **Ordinal trend:** mean `s` by `{cut, hold, hike}` and a monotonicity check
      (Spearman of `s` vs ordinal category); expect cut < hold < hike.
  (c) **Leading / forward-guidance:** Spearman ρ(`s_t`, `Δff_{t+1}`) — does
      today's tone align with the **next** meeting's action (signalling)?
- **DIAGNOSTIC (descriptive, no decision weight):** within hold meetings only,
  does `s_t` rank-correlate with the next move `Δff_{t+1}`? (Is the instrument's
  hawkishness carrying forward guidance the current action doesn't?)

## Decision rule (committed now)

**The stance instrument is a valid concurrent measure of policy hawkishness** iff
the PRIMARY Spearman ρ > 0 with permutation p < 0.05 **and** bootstrap 95% CI
strictly > 0. Then characterise *strength* (|ρ|: <0.2 negligible, 0.2–0.4 weak,
0.4–0.6 moderate, >0.6 strong) and *where it breaks* via the diagnostics — these
sharpen what the descriptive layer can and cannot honestly claim. If the primary
is not significant, report that the instrument does **not** track realised policy
(a stronger caveat the dashboard would then need). No post-hoc test selection;
the four tests above are the complete committed set.

## Baseline result (snapshot for retrain gating)

Reproducible artefact pinned alongside this pre-reg as
[`stance-instrument-validity-baseline-result.json`](./stance-instrument-validity-baseline-result.json).
Headline numbers from that snapshot (n=122 meetings, 2011-01-26 to 2026-04-29):

- PRIMARY Spearman(s, Δff) = **+0.284** (perm p = 0.001, CI95 [+0.069, +0.468])
- mean s by action: cut −0.533, hold −0.584, hike +0.420
- AUC hike-vs-cut **0.778** (CI95 [0.594, 0.944])
- ordinal Spearman +0.262, leading s_t vs Δff_{t+1} rho −0.034 (p 0.658)

**Gate for Lead-1 retrains:** rebuild `stance_daily.parquet` via
`scripts/build_stance_daily.py` and re-run `scripts/stance_instrument_validity.py`
after each retrain. A retrain "wins" against this baseline iff
`mean(s|cut) < mean(s|hold)` and AUC(s, cut-vs-hold) > 0.5 — the dovish-end
resolution finding the Lead-1 loss knobs target.
