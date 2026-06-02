# HAR-tercile deep-improvement study, 2026-06-02

## Premise

A recent arXiv preprint (2604.10402, "Asymmetric losses and longer lookbacks for
volatility-regime classification") argues that deep classifiers can beat HAR-RV
persistence on tercile volatility regimes once two design choices are made:
class-aware asymmetric loss for the high-volatility class, and longer lookback
windows (60 bars) so the network can pick up regime turns the daily HAR
mis-times. The paper reports per-asset gains in the 5 to 12 percent range on
SPY, QQQ and IWM, with the largest lifts in stress slices defined by elevated
VIX.

The fed-pulse HAR-tercile baseline (wiki section 21, daily-fusion pipeline)
reports macro-F1 of 0.687 / 0.685 / 0.654 at h=1/5/22. That number is the bar
the paper's claims have to clear on our data. The four arms below were designed
to test, in order of expected strength, whether any combination of stacking,
asymmetric loss, stress routing, or per-asset re-fitting moves macro-F1 above
that bar.

## Arms tested

Pre-registered plan and per-arm result JSONs are colocated at
`docs/research/har-tercile-arm-<key>-result.json`.

### Arm A: soft-prob stacking of HAR-tercile and dual-head DL (`stacking_har_dl`)

A per-fold logistic blend of HAR-tercile soft-probs and a torch MLP surrogate of
the dual-head DL classifier, with temperature scaling on the DL leg before
blending. Three seeds, five folds, three horizons (h=1, 5, 22). The HAR-tercile
leg was re-derived on the canonical training package
`tp_v3_full_rebuild_2026_05_30` because the `fed_comms_fusion` artifact tree the
wiki baseline depends on is not in the checkout.

Headline pooled macro-F1 (mean across 15 cells):

| Horizon | HAR-only (replay) | DL surrogate | Blend  | Blend delta vs HAR | Median p-value |
|---------|-------------------|--------------|--------|--------------------|----------------|
| h=1     | 0.298             | 0.350        | 0.326  | +0.029             | 0.134          |
| h=5     | 0.461             | 0.384        | 0.327  | -0.134             | 0.003          |
| h=22    | 0.372             | 0.475        | 0.379  | +0.008             | 0.108          |

Learned blend weights: `w_har` mean 0.27 to 0.54, `w_dl` mean 0.68 to 0.92. The
blend did not collapse to either leg, but the DL surrogate's errors were not
cleanly orthogonal to HAR's: at h=5 the blend loses by 13 percentage points
with median paired-bootstrap p of 0.003. Two horizons are statistical ties.

### Arm B: per-horizon specialist LSTM with focal + asymmetric loss (`specialist_h1_asym`)

Five-seed expanding-walk-forward run with `--regime-loss focal --focal-gamma
3.0` and a new `FED_PULSE_REGIME_UNDER_PENALTY=1.5` hook in
`backend/app/training/loop.py` that multiplies focal-CE loss by 1.5x when
true=high and argmax!=high. Sequence length stayed at 20 because the canonical
training package carries 20-bar `prior_bars_json` windows; the 60-bar
variant was the arm's documented fallback case. h=22 was proxied by
`--target-horizon 20`.

| Horizon | Macro-F1 mean ± std | Delta vs HAR (wiki 0.687 / 0.685 / 0.654) |
|---------|---------------------|-------------------------------------------|
| h=1     | 0.344 ± 0.023       | -0.343                                    |
| h=5     | 0.423 ± 0.025       | -0.262                                    |
| h=22*   | 0.434 ± 0.025       | -0.220                                    |

(*h=22 proxied by h=20)

Class-2 (high) F1 was the strongest cell at every horizon (0.49 to 0.58),
confirming the asymmetric loss did the expected thing mechanically. Classes 0
and 1 collapsed in trade, dragging macro-F1 well below HAR.

### Arm C: VIX-stress-gated routing (`stress_route`)

Per-row gate: route to dual-head DL when `vix_t_minus_1 > 22`, otherwise to
HAR-tercile. `VIX_3m_over_1m_slope` is fully NaN on this training package, so
the single-threshold fallback was used as the arm spec anticipated.

The dual-head DL classifier targets `forward_realized_vol_10d` only, so this
arm has no h=1/h=5/h=22 split; the comparison is pooled macro-F1 on the
canonical_vix surface.

| Slice                       | Routed | HAR-only | DL-only | n cells |
|-----------------------------|--------|----------|---------|---------|
| Pooled                      | 0.442  | 0.424    | 0.441   | 25      |
| Stress slice (VIX>22)       | 0.293  | 0.339    | 0.293   | 15      |

Paired block-bootstrap on the stress slice: routed minus HAR-only macro-F1 has
a 90 percent CI of `[-0.076, -0.017]` with mean -0.046. The CI excludes zero on
the negative side. The paper's qualitative claim that "high-vol is where the DL
model matters" is rejected on this evaluation surface: DL is significantly
worse than HAR on exactly the stress rows the paper says DL should win.

### Arm D: per-asset HAR-tercile on ^NDX and ^DJI (`per_asset_har`)

Not a deep-learning arm. Fits HAR(1,5,22) OLS per fold on a daily-r² RV proxy
(yfinance only carries daily closes) for NDX, DJI, and SPX as an apples-to-apples
daily comparator. The arm exists to fill the per-asset baseline gap and to
patch a known SPX-only routing bug in `services.har_tercile.predict_har_regime`
in a later commit (the patch itself is out of scope for the read-only plan
phase).

Pooled macro-F1 (single deterministic fit; n_seeds=1):

| Asset    | h=1   | h=5   | h=22  |
|----------|-------|-------|-------|
| SPX wiki | 0.687 | 0.685 | 0.654 |
| SPX daily-r² comparator | 0.298 | 0.498 | 0.550 |
| ^NDX     | 0.353 | 0.580 | 0.613 |
| ^DJI     | 0.317 | 0.525 | 0.586 |

On a like-for-like comparator (SPX under the same daily-r² proxy), NDX clears
SPX-daily by 5 to 8 percentage points and DJI by 2 to 4 percentage points,
replicating the per-asset variation the paper claims for equities. Raw NDX/DJI
pooled macro-F1 still trails the wiki SPX baseline because intraday RV is not
available off yfinance.

## Summary table

Wiki HAR-tercile baseline: h=1 0.687, h=5 0.685, h=22 0.654.

| Arm                  | h=1 macro-F1  | h=5 macro-F1  | h=22 macro-F1 | Best delta vs wiki | Beats wiki by 1σ? |
|----------------------|---------------|---------------|---------------|--------------------|-------------------|
| A. stacking_har_dl   | 0.326 ± 0.051 | 0.327 ± 0.120 | 0.379 ± 0.132 | -0.275 (h=22)      | No                |
| B. specialist_h1_asym| 0.344 ± 0.023 | 0.423 ± 0.025 | 0.434 ± 0.025 | -0.220 (h=22)      | No                |
| C. stress_route      | n/a*          | n/a*          | n/a*          | -0.245 (pooled)    | No                |
| D. per_asset_har NDX | 0.353 ± 0.053 | 0.580 ± 0.068 | 0.613 ± 0.064 | -0.041 (h=22)      | No                |
| D. per_asset_har DJI | 0.317 ± 0.069 | 0.525 ± 0.080 | 0.586 ± 0.065 | -0.068 (h=22)      | No                |

(*Arm C runs on the canonical_vix single-horizon surface; pooled macro-F1
0.442 vs wiki 0.687 daily-fusion).

No arm beats the wiki HAR-tercile baseline at any horizon. The closest call is
arm D NDX at h=22 with a -0.041 gap, and that arm is HAR-tercile re-fitted on a
weaker RV proxy, not a deep model.

## Decision

Keep HAR-tercile. The wiki HAR-tercile baseline holds at h=1, h=5 and h=22.
The deep-learning arms (A, B, C) all underperform it materially; the per-asset
arm (D) is a correctness fill rather than a competitor and lands inside the
expected band for a daily-r² RV proxy.

Recommended next step: file a documented-null-result note plus the
correctness-fill code change (per-asset HAR symbol arg in
`services.har_tercile.predict_har_regime`) as one PR. The four arms become
negative-result citations in the writeup. Do not promote any deep-learning arm
to the serving path.

## Caveats and threats to validity

1. Arm A's HAR baseline replay sits at 0.298 / 0.461 / 0.372 macro-F1 on the
   canonical training package, well below the wiki 0.687 / 0.685 / 0.654. The
   wiki baseline depends on the `fed_comms_fusion` daily-fusion artifact tree
   which is absent from the current checkout. The blend delta in arm A is
   therefore measured against a different row population and a different HAR
   feature source than the wiki number. Reconstructing the daily-fusion
   artifact would be required for an apples-to-apples stacking result.
2. Arm B could only test the asymmetric-loss half of the paper's claim because
   `prior_bars_json` carries 20-bar windows; the 60-bar lookback half requires
   regenerating the data slice. h=22 was proxied by h=20 because 22 is not in
   the supported target-horizon set. Both substitutions weaken the claim that
   the paper's design choices were fully replicated.
3. Arm C's dual-head DL classifier is single-horizon
   (`forward_realized_vol_10d`); the canonical_vix surface used for HAR-only
   comparison is statement-level and has smaller per-fold n than the
   daily-fusion surface the wiki baseline lives on. The 0.687 number is not
   reachable from this surface even by the HAR-only leg.
4. Arm D uses daily-r² RV instead of intraday 5-min RV (no intraday data on
   yfinance for NDX/DJI), so the raw pooled macro-F1 is not directly
   comparable to the wiki SPX number. The like-for-like SPX daily-r²
   comparator was reported to make the per-asset gap interpretable, but the
   underlying RV process is noisier than the wiki baseline's.
5. Fold-1 RV distributions differ markedly from later folds on NDX/DJI
   (KS up to 0.69 on DJI h=22 fold 1), pulling fold-1 macro-F1 down sharply.
   This is a known weakness of the OLS HAR fit on early-history equity
   indices.
6. The asymmetric-loss arm (B) successfully boosted class-2 F1 to 0.49 to 0.58
   but at the cost of class-0 and class-1 recall. This is the predicted
   failure mode for asymmetric loss on a balanced macro-F1 metric and should
   be cited as a counter-example to the paper's gain claim on macro-averaged
   evaluation.

## Files

- Research note (this file): `docs/research/har-tercile-deep-improvement-2026-06-02.md`
- Per-arm result JSONs:
  - `docs/research/har-tercile-arm-stacking_har_dl-result.json`
  - `docs/research/har-tercile-arm-specialist_h1_asym-result.json`
  - `docs/research/har-tercile-arm-stress_route-result.json`
  - `docs/research/har-tercile-arm-per_asset_har-result.json`
- Existing per-asset writeup (arm D): `docs/research/har_tercile_per_asset.md`
- Scripts (worktree only, not committed to main):
  - `scripts/run_har_dl_stacking.py`
  - `scripts/run_stress_route_eval.py`
  - `scripts/fit_per_asset_har_tercile.py`
- Code change (out of scope for this note, queued for follow-up):
  per-asset symbol arg in `backend/app/services/har_tercile.py`.
