# ADR 0027 — FOMC-attributable rates target (surprise decomposition)

The rates heads landed in #292 supervise three forward 5-day yield changes (`yield_2y_change_5d`, `yield_5y_change_5d`, `terminal_rate_change_5d`) in basis points. The supervised target is the raw observed move over `[T, T+5]` — every bp of yield change between the FOMC announcement close and five trading days later, regardless of what drove it.

That raw quantity blends two signals: the FOMC-attributable component (the part that responds to the policy decision and its surprise relative to the strictly-prior implied expectation), and the non-FOMC component (data releases between `T+1` and `T+5` like jobs / CPI / PPI, cross-asset shocks, position unwind, noise). For a forecaster sitting at `T-0` reading the FOMC statement, only the first is plausibly predictable from FOMC features. The second is orthogonal to anything the model can read off the statement and the strictly-prior market state. Training against the raw move floors the achievable R²: even a perfect FOMC-reaction model can only explain the FOMC-attributable share of the variance.

The literature pattern is to decompose the rates response into a surprise-driven component and a residual. Kuttner (2001) does this for the next-meeting Fed Funds futures contract; Gürkaynak-Sack-Swanson (2005) extends it across the curve with a two-factor PCA on the surprise window; Cieslak-Vissing-Jørgensen (2021) projects the response onto a policy-surprise axis to isolate the "Fed put" channel. The common construction projects the observed move onto a policy-surprise direction estimated from the OIS / Fed Funds futures surprise at the announcement and reports the projection scalar.

#291 shipped the pre-meeting expectation feature block (`pre_meeting_implied_next_move_bps`, `pre_meeting_implied_hike_prob`, etc.) that supplies the strict-prior expected leg. #350 (ADR 0024) reformulated `mp_surprise_level` against that expected leg so the surprise itself is leak-clean. The two prerequisites land the supply side for a Kuttner-style projection.

## What lands

A per-event-row supervised target

```
attributable_bps = observed_move_bps * sign(mp_surprise_level)     if |mp_surprise_level| >= 1.0 bp
                 = MISSING                                          otherwise
```

for each of the three rates heads. This is the 1-D Kuttner projection: observed move `m` projected onto the unit direction `u = surprise / |surprise|`, with the scalar `m · u = m * sign(surprise)` as the supervised target. The sign convention follows the policy-surprise direction so a positive target means the observed move agreed with the surprise (hawkish surprise → yields up, dovish surprise → yields down).

The surprise direction is the strict-prior `mp_surprise_level` from `mp_surprises.parquet`, post-ADR 0024:

```
mp_surprise_level_bps = (ff_target_after − ff_target_prior) * 100 − pre_implied_next_move_bps
                      = actual_target_change − strict_prior_expected_change
```

The expected leg is observable strictly before `event_date`; the actual leg is the announcement itself (the decision the surprise is defined against, not a market read of it). The direction is leak-clean: the surprise is defined at the moment of the announcement, and `sign(surprise)` carries no information from the `[T, T+5]` window the target measures.

Pause / no-change meetings where the FOMC matches the strictly-prior expectation exactly have `mp_surprise_level ≈ 0`. The projection direction is ill-defined and any signed projection would be a coin flip. We gate at `|mp_surprise_level| < 1.0 bp` (`SURPRISE_DIRECTION_EPSILON_BPS`) and mark the target missing rather than zero. Coercing to zero would inject a fake "no FOMC-attributable component" label on every pause meeting and bias the regression toward the origin; the existing `bps_mask` machinery in `build_partition_rates_targets` already handles missing rows row-by-row so masking is free. The 1-bp threshold is well below the standard 25-bp move and an order of magnitude above floating-point noise in the strict-prior implied-move proxy.

## Wiring

`ModelConfig.rates_target_mode` selects the derivation. Values: `"raw"` (default, byte-identical to pre-#305) and `"fomc_attributable"`. The mode applies uniformly to every mounted rates head; per-head mode-mixing was deferred so the CLI stays one knob deep.

Three new `FeatureVector.target_yield_*_change_5d_fomc_attributable` fields carry the projected target per event row, populated by the training-package loader alongside the existing raw targets. The loader computes the projection once per event from the strict-prior `mp_surprise_level` and writes the projected scalar (or `None`) onto every per-bar vector in the supervised sequence; only the target row is read downstream.

`_rates_field_for(head, target_mode=...)` returns either `target_yield_<tenor>_change_5d` or its `_fomc_attributable` sibling; `build_partition_rates_targets(..., target_mode=...)` plumbs the mode through to the gather. The per-fold standardiser (`fit_rates_scaler`) fits on the projected values when the mode flips, and val/test partitions reuse the train-fitted scaler so no look-ahead leaks. A new `--rates-target-mode` CLI flag forwards the choice into `ModelConfig`; default `"raw"` keeps the pre-#305 path byte-identical for every caller that doesn't opt in.

The evaluation surface (per-head `mae_bps` / `dir_acc` / `R²` with block-bootstrap CIs from `app.evaluation.regression_metrics`) doesn't change — metrics still measure prediction against supervised target in bps, but the target is the FOMC-attributable component when the new mode is active.

## Why not the alternatives

Keeping the raw target and decomposing the prediction at inference time is identity-on-the-output: the head's loss surface and gradient flow stay tuned to the noisy raw move, and the model has no incentive to focus on the FOMC-attributable component during training. The whole point of #305 is to change the loss, not the reporting frame.

Mounting two heads per tenor (raw + projection) and training them jointly doubles the rates parameter count and the loss-mixing surface (every head gets its own alpha mix with the joint loss), with unclear methodology lift over picking one target. The comparison sweep against `--rates-target-mode=raw` gives the side-by-side numbers without the parameter blow-up.

A 2-D projection onto `(mp_surprise_level, mp_surprise_path_factor)` has no second axis to project onto because the observed move is a 1-D scalar per head. The natural multi-dimensional extension is the GSS-style "across the curve" projection — one head per surprise factor rather than per tenor — which is a larger architectural change and is parked.

Using the standard CME-implied surprise window `(post − pre)` over a 30-minute window around the announcement reintroduces a `T+ε` read into the surprise quantity and undoes the strict-prior contract ADR 0024 just closed.

Signing the projection by `mp_surprise_path_factor` is plausible — path captures curve-shape surprise, level captures the immediate-policy surprise. We pick level because Kuttner anchors there and because the 5-day forward horizon is dominated by the near-end of the curve where the level surprise has the bigger reaction. The path-anchored variant is a follow-up if the level variant ships.

## Methodology shift

The framing changes honestly. Pre-#305: "the model predicts the post-FOMC 5-day rates move." Post-#305 with `--rates-target-mode=fomc_attributable`: "the model predicts the FOMC-attributable component of the 5-day rates move — the part of the move policy-watchers actually care about." The second framing is the one the literature uses and is the one a Fed-watcher product surface should lead with.

R² numbers across the two modes are not directly comparable. The fomc-attributable target has a smaller raw variance than the unprojected move (because the projection collapses non-FOMC variance into a residual we discard), so R² may be higher in the new mode without any actual lift in predictive power — the denominator changed. `mae_bps` and `dir_acc` stay comparable across modes (both in raw bps), but the interpretation shifts: `dir_acc` in fomc-attributable mode is "did the model correctly predict whether the move agreed with the surprise direction?", not "did the model correctly predict the move sign."

The expected lift is on interpretability and theoretical motivation, not headline accuracy. The §16 comparison table reports both modes side-by-side and lets the reader judge.

## Downstream effects

Default `--rates-target-mode=raw` is byte-identical to pre-#305. Existing callers, the canonical determinism regression, and the reproducibility-smoke CI stay green without changes. Opting in requires the explicit flag. A canonical-comparison sweep against `--rates-target-mode=fomc_attributable` is a Runpod job (operator-run via `make canonical-comparison TARGET_MODE=fomc_attributable`); this PR ships the code path, not the headline cell.

The new column lands on `FeatureVector` with `None` defaults. Three new fields, but `as_rich_list` doesn't include them (they ride on the target row only), so the rich-feature input tensor shape stays at 35 dims and the per-fold RobustScaler fit is byte-identical. Pre-#305 checkpoints deserialise into `rates_target_mode="raw"` via the `ModelConfig.from_model` fallback (the new field has a default).

The strict-prior contract on the projection direction is enforced upstream by ADR 0024 — this ADR consumes it rather than duplicating it. The post-#350 `feature-provenance-audit.md` documents `mp_surprise_level` as strict-prior; the projected targets inherit that gate — the direction is determined strictly before the event, with only the post-event observed move scaled by it.

Reviewer focus is on three contracts: the projection helper marks the surprise-zero case missing (not zero); `build_partition_rates_targets` accepts `target_mode` and forwards it without changing the train/val/test scaler-reuse contract; and the default mode reads the same `target_yield_<tenor>_change_5d` field the pre-#305 build path read. Each is pinned by a unit test in `test_rates_targets.py`.

## References

- `backend/app/training/rates_targets.py`, `backend/app/training/loaders.py`
- `backend/app/models/config.py::ModelConfig.rates_target_mode`
- Issues #291, #305; ADR 0024 (#350)
- Kuttner (2001); Gürkaynak, Sack, Swanson (2005); Cieslak, Vissing-Jørgensen (2021)
