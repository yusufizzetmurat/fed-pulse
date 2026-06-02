# Pre-registration — does the stance instrument track what the Fed actually did?

**Date:** 2026-06-02 · **Status:** pre-registered before measuring the
stance↔action relationship. (Anchor distribution checked for availability only.)

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
