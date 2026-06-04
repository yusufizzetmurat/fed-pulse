# Dense FOMC-text marginal value: text over the dense backbone

**Question:** on FOMC announcement days, does the statement text explain
realized-vol / abnormal-volume residual that the HAR/AR baseline leaves on the
table? (`app.data.dense_fomc_text`)

## Design

Two-stage, leakage-safe, per 5-fold walk-forward (embargo 10). Stage 1: the
HAR/AR baseline (`dense_forecast_train`) predicts every row; the residual
(true − baseline) is the signal the backbone misses. Stage 2: on FOMC rows only,
ridge that residual on the statement-text features, fit on train-FOMC, scored on
test-FOMC. Targets are log realized vol at h1/h3/h5/h10 plus abnormal volume
(`av`); R² is OOS vs the per-fold train-mean. Restricting to FOMC rows holds
calendar proximity fixed, so the marginal Δ = R²(baseline + text-residual) −
R²(baseline) isolates text. Text encoding is per-fold whitened PCA of the
mean-pooled FinBERT statement embedding plus a statement-delta scalar
(1 − cosine to the prior statement). n_fomc_test = 262.

Three arms:
- **text_alone** — baseline swapped for the per-fold train mean, so text must
  carry the demeaned target alone (standalone signal; r2_notext ≈ 0 by design).
- **delta_only** — HAR/AR baseline + the single statement-delta scalar (`pca_k=0`).
- **tp_v3_full_rebuild_2026_05_30** — HAR/AR baseline + full text block (PCA-16 + delta).

## Results (R² text vs no-text, Δ with 90% bootstrap CI)

text_alone (vs train-mean baseline):

| target | r2_notext | r2_text | Δ | Δ CI90 |
|---|---:|---:|---:|---|
| rv_1  | ~0 | −0.425 | −0.425 | [−0.666, −0.225] |
| rv_3  | ~0 | −0.322 | −0.322 | [−1.030, 0.107] |
| rv_5  | ~0 | 0.077 | 0.077 | [−0.079, 0.194] |
| rv_10 | ~0 | −0.584 | −0.584 | [−1.766, 0.079] |
| av    | 0 | −0.192 | −0.192 | [−0.386, −0.044] |

delta_only (vs HAR/AR baseline):

| target | r2_notext | r2_text | Δ | Δ CI90 |
|---|---:|---:|---:|---|
| rv_1  | 0.224 | 0.237 | +0.013 | [−0.020, 0.048] |
| rv_3  | 0.425 | 0.321 | −0.104 | [−0.322, 0.017] |
| rv_5  | 0.553 | 0.561 | +0.008 | [−0.005, 0.019] |
| rv_10 | 0.581 | 0.294 | −0.287 | [−0.812, 0.003] |
| av    | 0.336 | 0.206 | −0.129 | [−0.289, −0.023] |

tp_v3_full_rebuild_2026_05_30 (full text block vs HAR/AR baseline):

| target | r2_notext | r2_text | Δ | Δ CI90 |
|---|---:|---:|---:|---|
| rv_1  | 0.224 | −0.256 | −0.480 | [−0.706, −0.289] |
| rv_3  | 0.425 | −0.018 | −0.444 | [−1.103, −0.070] |
| rv_5  | 0.553 | 0.419 | −0.134 | [−0.262, −0.043] |
| rv_10 | 0.581 | −0.083 | −0.663 | [−1.750, −0.060] |
| av    | 0.336 | 0.131 | −0.205 | [−0.393, −0.074] |

**Verdict:** text adds no marginal value over the dense HAR/AR backbone. Every
arm's Δ is ≤ 0 or CI-straddles 0 except where it is significantly negative: the
full block drags every target with CIs clearing 0 below, and `av` drags in all
three arms. Text-alone cannot beat a train-mean baseline. A null result inside
a backbone that demonstrably works.

See also [text-alone-forecast-result.md](text-alone-forecast-result.md).
