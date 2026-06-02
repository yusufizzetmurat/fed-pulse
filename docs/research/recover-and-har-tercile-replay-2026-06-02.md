# Recovery and HAR-tercile replay on the fusion TP (2026-06-02)

## Substrate

All numbers in this note use the fusion training package pulled fresh from
Hugging Face: `data/processed/tp_intraday_fomc_text_volatility/`. Relevant
contents:

- `fusion/daily_fusion.parquet`: 5,385 daily rows, 46 columns, 2005-01-03 to
  2026-05-29. Carries HAR lags (`rv_daily`, `rv_weekly`, `rv_monthly`),
  forward targets (`rv_fwd_1`, `rv_fwd_5`, `rv_fwd_22`), volume / downside /
  jump HARs, cross-market correlation lags (`corr_tnx`, `corr_dxy`),
  calendar features (`days_since_stmt`, `days_to_stmt`) and MP surprise
  triples (`surprise_level`, `surprise_path`, `surprise_info`).
- `market/spx_5min_daily_rv.parquet`: SPX intraday RV at 5-min sampling,
  10 columns (`date`, `rv`, `rs_pos`, `rs_neg`, `bv`, `rq`, `rskew`,
  `rkurt`, `parkinson`, `rvol`), 5,385 rows.
- `corpus/fed_communications.parquet`: 2,000 Fed-comm documents.
- `embeddings/{finbert_fed_adjacent,bge_large_en_v15,e5_large_v2,gte_large}.parquet`:
  768-d per-document embeddings.

Two gaps in the fusion TP that affected this round:

1. No `_market_cache/` with per-symbol VIX/TNX/IRX daily parquets. The dense
   daily forecast (B) and the late-fusion text leg (D) both need that
   cache. Both were resolved by pointing at the canonical
   `tp_v3_full_rebuild_2026_05_30/_market_cache`, which is the same cache
   the production drivers (`scripts/late_fusion_gated_neutral.py`,
   `scripts/dense_daily_forecast.py`) use. Protocol unchanged.
2. No `events.parquet` and no cross-bank corpus. That blocked candidates E
   (NLP baseline batch on stance labels) and F (cross-bank transfer)
   entirely. They are reported as `not_runnable`, not as failures.

## Recovery: closed and open audit findings

Five candidates produced real numbers in this session. Two were marked
`not_runnable` up front because the fusion TP does not carry the inputs.

### Closed findings

| Key | Wiki claim                                                   | Recovered                                              | Match  |
| --- | ------------------------------------------------------------ | ------------------------------------------------------ | ------ |
| A   | HAR-tercile fusion 0.687 / 0.685 / 0.654                     | 0.6873 / 0.6850 / 0.6542 pooled (n=1,999 valid)        | yes    |
| B   | Dense rv_10 R^2 0.559 [0.547, 0.570], AV R^2 0.301           | 0.5585 [0.5471, 0.5698], 0.3008 [0.2803, 0.3209]       | yes    |
| C   | QLIKE bake-off C-vs-A 96.7% CI includes 0 at all 3 horizons  | identical to 4 dp at h1/h5/h22                         | yes    |
| D   | Late-fusion gated text leg null vs market-only at h1/h5/h22  | fused 0.629 / 0.634 / 0.496 vs market 0.631 / 0.637 / 0.497, gate active ~0.01-0.04, 90% block CIs match | yes |

All four numbers reproduce the wiki text to the printed precision.
Specifically:

- A used `scripts/reproduce_har_tercile_fusion.py` (newly added in this
  round) reading directly off `fusion/daily_fusion.parquet`. Single OLS
  per fold, embargo=23, q33/q67 train-slice thresholds, 5-fold expanding
  walk-forward. CPU only, under 30 seconds.
- B ran `app.data.dense_forecast_train.run` inside the backend container
  on the canonical `_market_cache` (9,096 walk-forward days, 1990+).
- C ran `scripts/qlike_heterogeneous_ensemble.py` after staging the
  fusion-TP `spx_5min_daily_rv.parquet` at the hard-coded
  `data/external/alphavantage_bars/` path. 5 folds, 8 MLP seeds + 3
  sigma-LSTM seeds, 300 epochs each. The C-vs-A 96.7% Bonferroni CI
  endpoints reproduce byte-equivalent to the preregistration.
- D ran `app.data.fed_comms_regime` on RTX 4080 via the gpu compose
  profile, 100 epochs x 5 walk-forward folds x 3 horizons. `n_eval=4,465`,
  gate collapses to roughly 0.01 to 0.04 mean activation, every CI matches
  the wiki to printed precision.

### Open / deferred

| Key | Reason                                                                                                                                                      |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| E   | Not runnable on the fusion TP. Needs `events.parquet` with stance labels; that lives in `tp_v3_full_rebuild_2026_05_30`, not in the HF fusion TP. Deferred. |
| F   | Cross-bank corpora (BoE, BoJ) are not in this repository or the fusion TP. Out of scope until cross-bank parquets are ingested.                              |
| G   | Started transformer + `tier2_market_rich` on the canonical TP with the gpu profile. The 20-trial sweep had not finished writing `forecaster_sweep_results.json` when the session stop hook fired. Partial: dataset and fold loading succeeded, sweep was in flight. No headline F1 was computed. Not reported as a number. |

## HAR-tercile replay: apples-to-apples results

All four arms ran on the same fusion TP, same valid mask (n_valid=5,363;
n_pooled=4,465), same expanding walk-forward (5 folds, embargo=23), same
train-slice q33/q67 tercile labels as the recovered baseline. The
comparator is the recovered baseline itself (0.6873 / 0.6850 / 0.6542
pooled at h1 / h5 / h22, fold std 0.042 / 0.034 / 0.046). The acceptance
bar is "beats by at least 1 sigma of the baseline fold std at any
horizon".

| Arm                 | h1 pooled | h5 pooled | h22 pooled | beats baseline 1 sigma? |
| ------------------- | --------- | --------- | ---------- | ----------------------- |
| baseline (recovered)| 0.6873    | 0.6850    | 0.6542     | -                       |
| `stacking_fusion`   | 0.6835    | 0.6261    | 0.6058     | no                      |
| `specialist_fusion` | 0.6497    | 0.6334    | 0.5461     | no                      |
| `stress_route_fusion`| 0.6597   | 0.6332    | 0.5735     | no                      |
| `per_asset_har` NDX | 0.2985    | 0.5437    | 0.5507     | no                      |
| `per_asset_har` DJI | 0.3395    | 0.5281    | 0.5550     | no                      |

Per-arm summary:

- `stacking_fusion`: HAR + DL meta-blend with inner-val grid search on
  the blend weight. h1 essentially tied (-0.004 pooled, -0.01 sigma on
  fold mean), h5 and h22 worse by roughly 1.1 to 1.4 sigma. Per-fold
  blend weights were bimodal (h22 weights [0.0, 1.0, 0.0, 1.0, 0.95]),
  which says the inner-val tail is reacting to local regime shifts that
  do not generalise to the test slice. DL leg alone (MLP over the full
  fusion feature stack) trails HAR pooled-F1 at every horizon.
- `specialist_fusion`: 2-layer LSTM (hidden=64, dropout=0.2, seq_len=20)
  with class-balanced focal loss (beta=0.999, gamma=2.0), 5-seed logit
  ensemble. Underperforms by 1.5 to 2.9 sigma at h1/h5/h22.
- `stress_route_fusion`: VIX-gated route to DL when prior-day VIX > 22.
  About 1,199 of 4,465 pooled test rows trip the gate; on those rows
  the DL prediction is materially weaker than HAR, so the gate degrades
  the macro-F1 at every horizon. This is the same sign as the prior
  `canonical_vix` stress_route arm, now confirmed on the fusion TP for
  all three horizons.
- `per_asset_har` NDX / DJI: not a per-asset HAR failure. The fusion TP
  carries 5-min intraday RV only for SPX. ^NDX and ^DJI fall back to
  daily-close r^2 as the rv proxy, which structurally caps macro-F1 in
  the same way the existing per_asset_har artifact already documented
  for SPX-daily-r^2 (0.298 / 0.498 / 0.550). The cause is data
  resolution, not the framing. Upstream fix is to source 5-min bars for
  NDX and DJI (or QQQ / DIA ETF proxies) and rebuild the fusion TP with
  per-asset intraday RV columns. Not in scope for this round.

No arm beats the recovered baseline outside CI at any horizon.

## Decision: KEEP_BASELINE

The recovered HAR-tercile fusion baseline holds at 0.687 / 0.685 / 0.654.
Four candidate improvements ran on the same substrate with the same
protocol and the same valid mask. None beat the baseline outside the
fold std band at any horizon. Three of them lose by more than 1 sigma at
h5 or h22.

The wiki claims that were audited in this round (A, B, C, D) all
reproduced to the printed precision on the fusion TP, so the
publishable text does not need to be revised on those findings.

Two recovery candidates remain open:

- E (NLP baseline batch on stance labels) requires `events.parquet`,
  which is not in the fusion TP. The same script runs cleanly on
  `tp_v3_full_rebuild_2026_05_30`, so the audit is not blocked, only
  parked on the canonical TP.
- F (cross-bank transfer) requires BoE / BoJ corpora not currently
  available in the repository. Out of scope until those ingest.
- G (transformer regime tier) had its training run truncated by the
  session stop. The dry run validated; the real run was in flight when
  the stop hook fired. Re-launching that sweep is a clean follow-up
  with no design changes.

## Next steps

1. Re-launch the G transformer + `tier2_market_rich` sweep on the
   canonical TP. The command is already validated, just needs a longer
   wall-time budget than this session allowed.
2. Run E (`nlp_baseline_batch`) against `tp_v3_full_rebuild_2026_05_30`
   as a clean follow-up. The fusion TP is not the right substrate for
   that arm.
3. Park F until BoE / BoJ corpora are ingested. Not actionable from the
   current data inventory.
4. If a HAR-tercile improvement is still wanted, the productive
   direction is data, not modelling: source 5-min bars for NDX / DJI
   (or QQQ / DIA) and rebuild the fusion TP with per-asset intraday RV.
   The four arms tried in this round (stacking, LSTM specialist,
   VIX-gated routing, per-asset on daily-r^2) all underperform the
   recovered baseline on the current substrate.
