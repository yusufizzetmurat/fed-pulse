# ADR 0027 — FOMC-attributable rates target (surprise decomposition)

Status: accepted, code path live; canonical sweep deferred to operator.
Date: 2026-05-27.
References:
- Issue #305 (closes).
- Issue #291 (pre-meeting expectation features; the strict-prior implied-move proxy that anchors the surprise direction).
- Issue #350 / ADR 0024 (strict-prior reformulation of `mp_surprise_level`; the leak-clean direction this projection consumes).
- `backend/app/training/rates_targets.py` — `fomc_attributable_projection`, `RATES_TARGET_MODES`, target-mode-aware `_rates_field_for` + `build_partition_rates_targets`.
- `backend/app/training/loaders.py` — per-event projection wired into both training-package loader sites.
- `backend/app/models/config.py` — `ModelConfig.rates_target_mode`; new FeatureVector `target_yield_*_change_5d_fomc_attributable` fields.
- Kuttner (2001), "Monetary policy surprises and interest rates: evidence from the Fed Funds futures market." J. Monetary Econ.
- Gürkaynak, Sack, Swanson (2005), "Do actions speak louder than words?" Int. J. Central Banking.
- Cieslak, Vissing-Jørgensen (2021), "The economics of the Fed put." Rev. Fin. Stud.

## Context

The rates heads landed in #292 supervise three forward 5-day yield changes (`yield_2y_change_5d`, `yield_5y_change_5d`, `terminal_rate_change_5d`) in basis points. The supervised target is the *raw* observed move over `[T, T+5]` — every basis point of yield change between the FOMC announcement close and five trading days later, regardless of what drove it.

That raw quantity blends two signals:

1. The FOMC-attributable component — the part of the move that responds to the policy decision (its surprise relative to the strictly-prior implied expectation).
2. The non-FOMC component — data releases between `T+1` and `T+5` (jobs reports, CPI, PPI), cross-asset shocks, position unwind, and noise.

For a forecaster sitting at `T-0` reading the FOMC statement text, only the first component is plausibly predictable from FOMC features. The second component is, by construction, orthogonal to anything the model can read off the statement and the strictly-prior market state. Training the head against the raw move therefore floors the achievable R²: even a perfect FOMC-reaction model can only explain the FOMC-attributable share of the variance.

The literature pattern is to *decompose* the rates response into a surprise-driven component and a residual. Kuttner (2001) does this for the next-meeting Fed Funds futures contract; Gürkaynak–Sack–Swanson (2005) extends it across the curve with a two-factor PCA on the surprise window; Cieslak–Vissing-Jørgensen (2021) projects the response onto a policy-surprise axis to isolate the "Fed put" channel. The common construction is to project the observed move onto a policy-surprise direction estimated from the OIS / Fed Funds futures surprise at the announcement and report the projection scalar.

#291 shipped the pre-meeting expectation feature block — `pre_meeting_implied_next_move_bps`, `pre_meeting_implied_hike_prob`, etc. — that supplies the strictly-prior "expected" leg. #350 (ADR 0024) reformulated `mp_surprise_level` against that strict-prior expected leg so the surprise quantity itself is leak-clean. The two prerequisites land the supply side for a Kuttner-style projection.

This ADR records the decision to use the post-#350 `mp_surprise_level` as the policy-surprise direction and project each rates head's observed move onto its 1-D unit vector.

## Decision

Add a per-event-row supervised target

```
attributable_bps = observed_move_bps * sign(mp_surprise_level)     if |mp_surprise_level| >= 1.0 bp
                 = MISSING                                          otherwise
```

for each of the three rates heads (2y / 5y / terminal). The construction is the 1-D Kuttner projection: the observed move `m` is projected onto the unit direction `u = surprise / |surprise|`, and the scalar projection coefficient `m · u = m * sign(surprise)` is taken as the supervised target. The sign convention follows the policy-surprise direction so a positive target means the observed move agreed with the surprise (hawkish surprise → yields up, dovish surprise → yields down).

### Projection direction

The surprise direction is the strict-prior `mp_surprise_level` from `data/external/fred/mp_surprises.parquet`, post-#350 / ADR 0024. That construction is

```
mp_surprise_level_bps = (ff_target_after − ff_target_prior) * 100 − pre_implied_next_move_bps
                      = actual_target_change − strict_prior_expected_change
```

where `pre_implied_next_move_bps` is the FRED-only implied move proxy from `app.data.rates_event_features.implied_next_move_bps` at `T-1`. The expected leg is observable strictly before `event_date`; the actual leg is the announcement itself (the policy decision the surprise is defined *against*, not a market read of it). The direction is therefore leak-clean: the surprise is defined at the moment of the announcement, and `sign(surprise)` carries no information from the `[T, T+5]` window the target measures.

### Edge case: |surprise| < epsilon

Pause / no-change meetings where the FOMC matches the strictly-prior expectation exactly have `mp_surprise_level ≈ 0`. The projection direction is ill-defined (the unit vector divides by zero) and any signed projection would be a coin flip. We gate at `|mp_surprise_level| < 1.0 bp` (`SURPRISE_DIRECTION_EPSILON_BPS`) and mark the target *missing* rather than zero. Coercing to zero would inject a fake "no FOMC-attributable component" label on every pause meeting and bias the head's regression toward the origin; the existing `bps_mask` machinery in `build_partition_rates_targets` already handles missing rows row-by-row so the masking is free.

The 1-bp threshold is well below the FOMC's 25-bp standard move and an order of magnitude above floating-point noise in the strict-prior implied-move proxy.

### Wiring

- A new `ModelConfig.rates_target_mode` field with values `"raw"` (default, byte-identical to pre-#305) and `"fomc_attributable"` selects the derivation. The mode applies uniformly to every mounted rates head; per-head mode-mixing was deferred so the CLI surface stays one knob deep.
- Three new `FeatureVector.target_yield_*_change_5d_fomc_attributable` fields carry the projected target per event row, populated by the training-package loader alongside the existing raw targets. The loader computes the projection once per event from the strict-prior `mp_surprise_level` and writes the projected scalar (or `None` for ill-defined surprises) onto every per-bar vector in the supervised sequence; only the target row is read downstream.
- `_rates_field_for(head, target_mode=...)` returns either `target_yield_<tenor>_change_5d` or `target_yield_<tenor>_change_5d_fomc_attributable`; `build_partition_rates_targets(..., target_mode=...)` plumbs the mode through to the gather step. The per-fold standardiser (`fit_rates_scaler`) fits on the projected values when the mode flips, and val / test partitions reuse the train-fitted scaler so no look-ahead leaks into the standardisation step.
- A new `--rates-target-mode` CLI flag on `app.train_forecaster` forwards the choice into `ModelConfig`. The default `"raw"` keeps the pre-#305 path byte-identical for every caller that does not opt in.

The evaluation surface (per-head `mae_bps` / `dir_acc` / `R²` with block-bootstrap CIs from `app.evaluation.regression_metrics`) does not change — the metrics still measure the head's prediction against the supervised target in bps, but the target itself is the FOMC-attributable component when the new mode is active. Side-by-side comparisons against the raw-mode headline live on the per-head metric block; the canonical comparison is the operator-run sweep against `--rates-target-mode=fomc_attributable`.

## Alternatives considered

**Raw move + post-hoc decomposition.** Keep training against the raw observed move, then decompose the prediction at inference time against the surprise direction. Rejected: the post-hoc decomposition is identity-on-the-output, so the head's loss surface and gradient flow stay tuned to the noisy raw move; the model has no incentive to focus on the FOMC-attributable component during training. The whole point of #305 is to *change the loss*, not the reporting frame.

**Predict both heads jointly.** Mount two regression heads per tenor — one for the raw move, one for the projection — and train them jointly. Rejected: doubles the rates parameter count and the loss-mixing surface (every head now has its own alpha mixing with the joint loss). The methodology lift over picking one target is unclear, and the comparison sweep against `--rates-target-mode=raw` already gives the side-by-side numbers without the parameter blow-up.

**Project onto the full 2-D surprise vector (level + path).** Use `(mp_surprise_level, mp_surprise_path_factor)` as a 2-D surprise direction and project the observed move onto it. Considered, but the observed move is a 1-D scalar per head (`yield_<tenor>_change_5d`), so a 2-D direction has no second axis to project onto without coupling the three heads. The natural multi-dimensional extension is the GSS-style "across the curve" projection, which would mount one head per surprise factor rather than one per tenor. That is a larger architectural change and is parked.

**Use the standard CME-implied surprise window `(post − pre)` over a 30-minute window around the announcement.** Rejected: the post-#350 audit explicitly closed the `[T-1, T+1]` window in the surprise construction (ADR 0024). Using the 30-minute window would reintroduce a `T+ε` read into the surprise quantity and undo the strict-prior contract. The strict-prior version is the one a forecaster scoring at `T-0` can actually compute.

**Sign the projection by `mp_surprise_path_factor` instead of `mp_surprise_level`.** The path factor captures curve-shape surprise; the level factor captures the immediate-policy surprise. Both could plausibly anchor the direction. We pick level because the literature (Kuttner) anchors there and because the 5-day forward horizon the heads supervise is dominated by the near-end of the curve where the level surprise has the bigger reaction. The path-anchored variant is a follow-up if the level variant ships.

## Consequences

### Methodology

The headline framing shifts honestly. Pre-#305: "the model predicts the post-FOMC 5-day rates move." Post-#305 with `--rates-target-mode=fomc_attributable`: "the model predicts the FOMC-attributable component of the 5-day rates move — the part of the move that policy-watchers actually care about." That second framing is the one the literature uses and is the one a Fed-watcher product surface should lead with.

The R² numbers between the two modes are not directly comparable. The fomc-attributable target has a smaller raw variance than the unprojected move (because the projection collapses any non-FOMC variance into a residual we discard), so R² values may be higher in the new mode without any actual lift in predictive power — the denominator changed. The mae-bps and dir-acc metrics remain comparable across modes (both in raw bps units), but the interpretation shifts: dir-acc in fomc-attributable mode is "did the model correctly predict whether the move agreed with the surprise direction?", not "did the model correctly predict the move sign."

### Model + sweep

The default `--rates-target-mode=raw` is byte-identical to the pre-#305 path. Existing callers, the canonical determinism regression, and the reproducibility-smoke CI smoke all stay green without changes. Opting in requires the explicit flag.

A canonical-comparison sweep against `--rates-target-mode=fomc_attributable` is a Runpod job (operator-run via `make canonical-comparison TARGET_MODE=fomc_attributable` once code lands). This PR ships the code path, not the headline cell. The sweep is not a merge gate.

### Reproducibility

The new column lands on the `FeatureVector` dataclass with `None` defaults. The dataclass shape changes (three new fields), but `as_rich_list` does not include any of them — they ride on the target row only — so the rich-feature input tensor shape stays at 35 dims and the per-fold RobustScaler fit is byte-identical. Pre-#305 checkpoints deserialise into `rates_target_mode="raw"` via the `ModelConfig.from_model` fallback path (the new field has a default).

The strict-prior contract on the projection direction is enforced upstream by ADR 0024 — this ADR consumes that contract rather than duplicating it. The post-#350 `feature-provenance-audit.md` documents `mp_surprise_level` as strict-prior; the new projected targets inherit that gate by construction and are themselves strictly `<` event_date in the sense that the *direction* is determined strictly before the event, with only the post-event observed move scaled by that direction.

### Reviewer focus

- Does the projection helper correctly mark the surprise-zero case as missing (not zero)? The unit test `test_fomc_attributable_projection_pause_meeting_marks_missing` pins this.
- Does the per-fold standardiser fit on the train slice only when the new mode is active? Yes — `build_partition_rates_targets` accepts the `target_mode` argument and forwards it to the gather step, but the train/val/test scaler-reuse contract is identical to the raw-mode path. The unit test `test_build_partition_rates_targets_fomc_attributable_uses_train_scaler` pins this.
- Is the default `--rates-target-mode=raw` byte-identical to the pre-#305 path? Yes — the default mode reads the same `target_yield_<tenor>_change_5d` field the pre-#305 build path read, and the per-fold builder takes the same code path. The smoke test `test_train_model_smoke_rates_target_mode_raw_byte_identical` would not be cheap to write as a true byte-identical assertion (the model weights diverge with seed micro-variation), so we settle for "default mode runs to completion and the persisted ModelConfig records rates_target_mode='raw'."

### Submission-time framing

The thesis framing is honest. This is a methodology contribution: a literature-mapped re-derivation of the supervised target so the head's loss is aligned with what a Fed-watcher actually wants to predict. It is *not* expected to beat the raw target on R² (the denominators differ); the case for it is interpretability and theoretical motivation, not headline accuracy. The §16 comparison table reports both modes side-by-side and lets the reader judge.
