# JOB 2 — #217 D2 architecture ensemble (5 archs × 5 samples × 5 seeds × 4 folds)
Archs: gru, lstm, lstm_attn, tcn, transformer. Direct script + ensemble_aggregator (docker bypassed).

## Ensemble (pooled over 2205 rows)
- mean_logit: macro_f1 = 0.4922, 95% CI [0.4595, 0.5235]
- mean_softmax: ~0.49 (see JSON)
- (plurality_vote in JSON)

## Decision input — compute single-arch POOLED baseline laptop-side
The aggregator emits the ENSEMBLE pooled macro_f1 but NOT per-arch pooled. Do NOT use the
per-fold max macro_f1 (tcn 0.7743, transformer 0.7258, gru 0.7170, lstm 0.7002, lstm_attn
0.6753) as the lift baseline — those are max-over-folds, inflated by single-class/degenerate
test folds, and not comparable to the pooled-over-2205 ensemble number. To apply the #217
rule (ensemble lift >+0.03 with overlapping CI), pool each arch's per-(fold,seed) predictions
over the same 2205 rows and compare to 0.4922 [0.4595,0.5235]. Raw per-arch
forecaster_sweep_results.json are in arch_sweep_raw/canonical/<arch>/ for that computation.
