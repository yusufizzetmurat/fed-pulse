# ADR 0045 — Intraday-window reaction pivot: direction-first de-risk result

Status: accepted, CLOSED. Full null: FOMC text predicts neither the direction nor the magnitude of the intraday announcement reaction at n=110 (2013→present). Pivot does not proceed to serving. See Update 2 for the magnitude result and final decision.
Date: 2026-05-31 (updated same day with the n=110 scale-up result — see "Update" below).
References:
- Round 6 / Path 2 from wiki `06_Deep_Learning_Roadmap.md §Path 2 contingency`; result summarized in wiki `17_Intraday_Pivot_Negative_Result`.
- Data: Polygon/Massive Starter intraday SPY backfill (`backend/app/data/polygon_spx.py`) → `data/external/polygon/spx_intraday_fomc_days.parquet` (41 FOMC statement days, 2021-06-16…2026-04-29, full 91-bar 13:30–15:00 ET windows).
- Dataset: `backend/app/data/intraday_event_builder.py` → `intraday_events.parquet` (41 events, both target windows).
- Harness + result: `backend/app/data/intraday_direction_train.py` → `data/artifacts/intraday_direction/tp_v3_full_rebuild_2026_05_30/result_{immediate,delayed}.json`.

## Context

The result-improvement programme plateaued at the SNR ceiling for forward-10-day-vol regime (strict-forward Transformer re-baseline 0.4308). The literature (Cieslak-Schrimpf 2019, Lucca-Moench 2015, Boguth et al. 2019) locates FOMC text's predictive power in the **intraday announcement window**, not 10-day vol. This pivot replaces the target with the SPX reaction around the 2:00pm ET statement release and asks, on a cheap de-risk pass, whether the **direction** of that reaction is predictable from a pre-announcement market sequence + FinBERT text.

The de-risk deliberately used only what was free-to-cheap: the Polygon Starter plan (rolling 5-year history) yields **41 FOMC statements** (2021-06+), far short of the ~151 available since 2010. The existing `MultiModalForecasterModel` (gated-InfoNCE fusion, classification head) was trained `n_classes=2` with a 31-step pre-announcement 1-min bar sequence as the market modality and the mean-pooled `finbert_fed_adjacent` CLS embedding as text. Two target windows were built and evaluated — immediate (14:00→14:30) and delayed (14:30→15:00) — with 4 expanding walk-forward folds, pooled out-of-fold directional accuracy, and a 90% bootstrap CI, against majority-class and market-only (text-ablated) baselines. Seed 11, CPU.

A pre-existing repo bug surfaced during the run: the default sentiment model id `gtfintechlab/fomc-roberta-any-exp` 404s on HF and silently falls back to `distilbert-sst-2`. The de-risk was re-run with `FED_PULSE_SENTIMENT_MODEL=yusufizzetmurat/finbert-fed-adjacent` (the registry's canonical encoder) to get a valid text channel. The repo default should be re-pointed separately (out of scope here).

## Result

| window | full | market-only | majority | text lift | 90% CI | per-fold |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| **immediate 14:00→14:30** | **0.656** | 0.531 | 0.594 | **+0.125** | [0.500, 0.783] | 0.625 / 0.500 / 0.625 / 0.875 |
| delayed 14:30→15:00 | 0.469 | 0.500 | 0.562 | −0.031 | [0.312, 0.594] | 0.375 / 0.500 / 0.625 / 0.375 |

## Decision

1. **The canonical window is the immediate 14:00→14:30 reaction.** It is the only window with positive signal: the full model beats both the majority baseline (+0.062) and the market-only ablation (+0.125), and accuracy *rises across folds* (0.875 on the most recent, largest-train fold) — consistent with a real, data-hungry signal. The delayed window shows nothing (full below both baselines); it is dropped. This resolves the spec's "build both, pick by result" decision.

2. **The result is suggestive, not significant — the pivot stays gated on a corpus scale-up.** At n=41 the 90% CI lower bound (0.500) sits below the majority baseline (0.594), so the signal is not yet statistically distinguishable from chance. This is the predicted underpowered outcome, not a null: the point estimate, the +0.125 text-over-market lift, the across-fold improvement, and the agreement with the canonical CVJ/Lucca-Moench window all point the right way.

3. **Next step is the Alpha Vantage scale-up to ~151 events, then re-run.** Subscribe to Alpha Vantage Standard (~$50, one month then cancel), extend `alphavantage_spx.py` to persist raw bars in the same schema `load_intraday_bars` reads (so Phases 2–3 run unchanged), backfill 2011→present, and re-run the harness. Expectation: the CI tightens; the immediate-window signal either confirms (clears the baseline) or resolves to a clean, adequately-powered null. Either is a defensible thesis result.

## Consequences

- No serving/frontend integration until the scaled re-run confirms signal (Approach A gate holds).
- The immediate-window framing carries forward: feature window 13:30→14:00, target window 14:00→14:30, direction = sign(return).
- Magnitude-regression and the InfoNCE-λ contrastive term remain deferred to the widening phase; the de-risk used CE on direction only.
- The Polygon Starter 5-year limit is the binding reason coverage stopped at 41 events; the scale-up moves the data source to Alpha Vantage (cheaper full history) — Polygon is not upgraded.

## Update (2026-05-31) — n=110 scale-up: direction is a null

Alpha Vantage Standard backfilled 1-min SPY for every FOMC statement 2013→present (the 2:00pm-release era, where the 14:00 boundary is valid): **110 events** (2013-01-30…2026-04-29), full 91-bar windows, via `alphavantage_spx.backfill_fomc_days_raw_bars` → `data/external/alphavantage_bars/` (same schema; `load_intraday_bars` + the builder consumed it unchanged). Dataset rebuilt to 110 events; direction balance is near-even (immediate majority 0.527). Same harness, seed 11, FinBERT `finbert_fed_adjacent`.

| window | full | market-only | majority | 90% CI | per-fold (n_test=22 each) |
| --- | ---: | ---: | ---: | --- | --- |
| immediate | 0.545 | 0.500 | 0.500 | [0.455, 0.636] | 0.727 / 0.682 / 0.409 / 0.364 |
| delayed | 0.534 | 0.523 | 0.568 | [0.455, 0.625] | 0.591 / 0.455 / 0.636 / 0.455 |

**Verdict: the n=41 immediate-window signal (0.656) does not survive.** At n=110 the immediate window is statistically indistinguishable from chance (point 0.545, CI straddles 0.50, text adds nothing over a 0.500 market-only baseline). The per-fold pattern *reversed* — the recent, larger-train folds are now the weakest (0.41, 0.36) vs the strongest at n=41 — confirming the earlier result was small-sample overfitting, not a data-hungry trend. The delayed window sits below its majority baseline. **Documented null for the direction target.**

**Scope of the null (what it does and does not say):** this is a null for *2-class reaction direction* under this configuration (CE-only, no InfoNCE-λ term, small GRU, 3 scale-free bar features, mean-pooled FinBERT, 110 events). It is consistent with market efficiency on the *sign* of the announcement reaction. It does **not** test reaction *magnitude/volatility* — which is where the CVJ/Lucca-Moench literature actually locates text predictability, and which the spec deferred to the widening phase. The honest next question is magnitude regression, not more direction data (110 events already pin direction at ~chance; more data will not move a CI centred near 0.50).

**Decision:** do not advance the direction target to serving/frontend. Either (a) write up the direction null as the pivot's result, or (b) open a magnitude-regression follow-up (new regression head + InfoNCE-for-regression) as the remaining legitimate test of the pivot thesis. No further data spend is warranted for the direction question.

## Update 2 (2026-05-31) — magnitude regression: text is a full null

Ran the deferred magnitude test (`intraday_magnitude_train`): predict `mag_{window}` = |return| with the multimodal model as a regressor (n_classes=2 with MSE on output unit 0 — the model enforces n_classes≥2, so a 1-unit head is not available), walk-forward out-of-sample R² / RMSE / Spearman vs a per-fold mean-magnitude baseline, full vs market-only, both windows, n=110, seed 11, FinBERT `finbert_fed_adjacent`.

| window | full R² (CI90) | market-only R² | full Spearman | RMSE vs base |
| --- | --- | ---: | ---: | --- |
| immediate | −0.062 (−0.123, −0.014) | **+0.080** | −0.116 | 0.00250 vs 0.00243 |
| delayed | −0.029 (−0.051, −0.008) | −0.129 | −0.022 | 0.00431 vs 0.00425 |

**Verdict: the FOMC-text thesis is a full null on the intraday pivot.** The full (text+market) model has *negative* out-of-sample R² on both windows — worse than predicting the mean reaction size. Text does not predict reaction magnitude any more than it predicted direction; on the immediate window adding text actively *destroys* a signal the market channel had on its own (+0.080 → −0.062).

**The one positive result is not about text:** market-only immediate-magnitude R²=+0.080, Spearman=0.256 — pre-announcement price action weakly predicts reaction size. This is consistent with **volatility clustering** (ARCH-type autocorrelation), a known market-microstructure effect independent of FOMC communication, and it is small (n=110, no CI computed for the ablation). It is not evidence for the pivot's hypothesis.

**Final decision: close the intraday pivot as a documented negative result.** Across a clean, adequately-powered design (110 FOMC statements, 2013→present, the canonical 2:00pm-release era), FOMC text — encoded with the project's own `finbert_fed_adjacent` — does not predict the intraday announcement-window reaction in either direction or magnitude beyond a market-only baseline. This is the honest answer to the wiki's "Path 2 is the next durable lever" hypothesis: at this corpus scale and feature/model setup, it is not. No serving/frontend; no further data spend. Deferred, genuinely-untested levers remain (InfoNCE-λ contrastive term, options/VIX pre-announcement features, the full ~30-year corpus via a higher Polygon tier) but none is on the critical path and each is lower-expected-value than the work already shipped.
