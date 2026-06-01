# Corner B — Pre-registration: does FOMC policy-stance text predict short-end yield reactions?

**Status:** pre-registered 2026-06-01, *before any result was inspected* (feasibility checked
data adequacy only — feature variation, overlap, Δy variation — NOT the stance↔yield
relationship). Committed before the experiment runs.

## Background and motivation

This is the **last** economically-motivated text-alpha avenue. The directional/level equity
nulls (late-fusion rebuild, intraday pivot) and the second-moment null (Corner A, text→RV)
leave one channel un-killed: **rates**. The Fed mechanically controls the short rate and steers
the front of the curve through forward guidance — the text *is* the policy signal for the
1Y–2Y much more than for equities. Supporting priors:
- the drift study found a weak (multiplicity-uncorrected) 2Y-yield lead from FOMC text;
- the external literature reports central-bank-sentiment gains specifically on **bond yields**
  (e.g. Hilscher–Nabors–Raviv 2024: RMSFE −1.3% on bonds, DM-significant) — a yields result,
  not an equities one.

So we give the **front of the curve** its own pre-registered shot, with the strongest single
text feature (policy stance) and the correct framing (event reaction, not a stale daily level).

## Hypotheses

- **H1:** a meeting's policy-stance signal predicts the subsequent short-end Treasury-yield
  reaction incrementally over a pre-meeting market baseline.
- **H0:** the stance feature adds nothing (ΔMSE = 0).

## Data

- **Yields:** `data/external/fred/DGS2.json` (2Y) and `DGS1.json` (1Y), daily, 2009-12→2026-05
  (4112 obs each). Short end = the forward-guidance-sensitive tenors (1Y/2Y price the expected
  funds-rate path; 5Y+ is term-premium/growth-driven, so deliberately excluded).
- **Statements:** FOMC statements from `events.parquet` (one per meeting date, fullest text);
  **151** fall inside the yield window.
- **Text feature:** stance `s = P(hawkish) − P(dovish)` from the served multi-axis classifier
  (std 0.61 across the sample; hawkish → expect yields up). This is the policy-direction
  (forward-guidance) content. Leak-safe: `s_i` uses only statement `i`'s own released text.

## Targets (event reactions)

For each statement `i` released on business day `t_i`, the short-end yield reaction:
- **1-day:** `r1 = y(t_i) − y(t_i − 1bd)` in bps (the announcement-day move).
- **5-day:** `r5 = y(t_i + 5bd) − y(t_i − 1bd)` in bps (the post-meeting adjustment).

for each tenor (2Y, 1Y). → **2 horizons × 2 tenors = 4 tests.**

## Models (paired, walk-forward by event)

- **Baseline:** OLS on pre-meeting, as-of-`t_i−1` features — trailing 5-day Δy momentum,
  yield level, trailing 10-day realized yield vol. (Yield changes are near-martingale, so this
  is a genuinely hard-to-beat baseline; "predict 0" is reported as a second floor.)
- **Treatment:** baseline `⊕ [s_i]`.
- **Walk-forward:** events in time order, expanding train head (burn-in 40 events), predict the
  next event's reaction OOS. OLS coefficients fit on the train events only. Strictly leak-free.

## Metric, significance, multiplicity (fixed)

- **Metric:** OOS mean-squared error of the reaction (bps²). Secondary: directional accuracy
  (sign of the reaction) and MSE vs the predict-0 floor.
- **Test:** Diebold–Mariano on the per-event squared-error differential
  `d = e²_baseline − e²_treatment`, Newey–West HAC variance (lag 5), two-sided.
- **Multiplicity:** 4 tests → Bonferroni at family α = 0.10 → per-test threshold **p < 0.025**.
- **Pre-registered hit:** DM `p < 0.025` **and** treatment MSE < baseline MSE, in **≥ 1** cell.

## Decision rule (fixed)

- **Hit →** the rates/forward-guidance channel carries real text signal — report it; it becomes
  the project's second (and only text-based) forecasting edge.
- **No hit →** text is **fully closed** for forecasting across every channel we can motivate
  (equity direction, magnitude, volume, volatility, and now rates). Report the null and stop.
  No iteration on feature transform, window, or tenor to manufacture significance.

## Honest ceiling

Even a hit is a low-frequency (~8 meetings/yr), front-end-rates result — a clean thesis finding
and a risk/positioning signal, not a high-capacity trading edge.

## Artifacts

- `data/artifacts/corner_b_text_rates/result.json` (per-cell baseline/treatment MSE, ΔMSE,
  DM stat + p, Bonferroni threshold, directional accuracy, predict-0 floor, n).
- Committed before that file exists.

---

## Result (run 2026-06-01, walk-forward by event, n=150 statements / 110 OOS) — **NULL**

| cell | MSE base | MSE +stance | MSE predict-0 | ΔMSE | DM p | dir-acc base→+s | hit |
|------|---------:|------------:|--------------:|-----:|-----:|----------------:|-----|
| 2Y / 1-day | 48.29 | 48.80 | 50.65 | −0.51 | 0.68 | 0.45 → 0.45 | no |
| 2Y / 5-day | 185.96 | 188.39 | 166.26 | −2.44 | 0.63 | 0.48 → 0.46 | no |
| 1Y / 1-day | 27.99 | 28.59 | 23.62 | −0.61 | 0.19 | 0.41 → 0.44 | no |
| 1Y / 5-day | 152.51 | 154.30 | 100.42 | −1.79 | 0.67 | 0.50 → 0.50 | no |

**0 / 4 cells hit.** Adding stance makes the forecast **worse** in every cell (ΔMSE < 0); no DM
test is significant and the sign is the wrong way. Directional accuracy hovers at ~50% (coin
flip) with and without text. **Verdict: NULL.**

**Well-powered, not a weak test:** the stance feature varies strongly (std 0.58, full
dovish↔hawkish swing) and there are 110 OOS events. The null is informative. Note the OLS
baseline itself loses to "predict zero" in 3 of 4 cells — short-end yield reactions are
essentially unforecastable from pre-meeting features, and stance adds nothing.

**Caveat (honest):** we tested the stance *level*, not the stance *surprise* (vs a market
expectation we don't have). But the announcement-day reaction `r1` should respond to the
surprise component, and it shows no signal in either sign — consistent with the
efficient-information story: by the time the statement text is public, the reaction is already
priced. Per the pre-registered rule, we do **not** iterate into a surprise-construction variant.

**Decision (per pre-registration):** null → **text is fully closed for forecasting** across
every economically-motivated channel (equity direction, magnitude, volume, volatility, rates).
Text remains valuable as *description* (stance/certainty/time labels, semantic diff) — not as a
market predictor. The one forecasting edge in the project stays the QLIKE-DLq RV ensemble.
