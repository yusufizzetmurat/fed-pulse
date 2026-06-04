# ADR 0022 — Trajectory arm parameter-count cap + lift-vs-baseline gating

Status: accepted, in production (as of merge).
Date: 2026-05-27.
References:
- Issue #332.
- `backend/app/trajectory/train.py` — `assert_parameter_count_within_cap`, `DEFAULT_PARAMETER_COUNT_CAP`, `--no-param-cap` CLI flag.
- `backend/app/trajectory/baselines.py` — `previous_stance`, `rolling_majority`, `small_lstm_baseline`, `compare_against_transformer`, `LIFT_THRESHOLD_POINTS`.
- `backend/app/services/trajectory.py` — `_load_lift_check`, `_TrajectoryState.lift_vs_baseline / delta_dir_acc / baseline_used`.
- `backend/app/schemas.py` — `TrajectoryResponse.lift_vs_baseline / delta_dir_acc / baseline_used`.
- `tests/unit/test_trajectory_baselines.py` — synthetic high-autocorrelation fixture + cap-trip guard.
- `fed-pulse.wiki/06_Deep_Learning_Roadmap.md §6.16` — trajectory baselines + parameter-count cap row.
- PR #296 — the trajectory arm this ADR amends.

## Context

The trajectory head shipped in #296 runs at 4 layers x 64 d_model x 4 attention heads against ~250 historical FOMC statements. At the canonical 768-d DAPT encoder embedding, the configuration lands around 250k trainable parameters. The data:parameter ratio is poor by construction, and the issue surfaced two compounding concerns.

1. No published baseline comparison. FOMC stance sequences carry strong autocorrelation; meetings cluster in regime runs (a dovish meeting is overwhelmingly followed by another dovish meeting, and the modal class accounts for a non-trivial share of the corpus). Naive predictors that echo the last stance or take the rolling majority over the last few meetings score well by default. Without those baselines published on the same fold protocol, the Transformer arm's absolute directional-accuracy number tells the reviewer nothing about whether the arm extracted real signal versus tracking the autocorrelation a `previous_stance` predictor would have matched for free.
2. Capacity overshoot. A 250k-parameter Transformer over 250 statements is the textbook "more parameters than rows" configuration. The model can fit the training noise even on a strict-forward walk-forward split, then read clean on the validation slice purely because the test labels share the autocorrelation structure of the train labels. A smaller arm (2 layers x 32 d_model, ~50k params at the canonical embedding) is the configuration the data warrants; the larger arm should be admitted only when a clean lift over the strongest naive baseline justifies the additional capacity.

Two options were on the table:

- **Option A — soft note.** Document the parameter-count concern in the §6.7 cell and leave the 4x64 default in place. Surface the baseline comparison as a follow-up paragraph in the wiki only. No risk of breaking existing bundles; no structural change to the codebase.
- **Option B — hard cap with override.** Cap the Transformer arm at the configuration the data warrants (2x32) by enforcing a parameter-count assertion in the trainer entry point, publish the three naive baselines (`previous_stance`, `rolling_majority(n=3)`, `small_lstm_baseline`) on the canonical fold protocol, ship a `lift_vs_baseline` badge on `/analyze/trajectory` so the UI can render the verdict, and document the >= 5pp directional-accuracy lift threshold the override requires. Bundles trained before this ADR continue to load (the new schema fields default to `None`/`False`).

Option A leaves the same gap the issue surfaces. Option B is the path the issue spells out, and is the option this ADR records.

## Decision

Option B. The trajectory arm ships with:

1. **Parameter-count cap.** `app.trajectory.train.assert_parameter_count_within_cap` runs in `train_and_persist` before the fit loop starts. The default cap is `DEFAULT_PARAMETER_COUNT_CAP = 75_000`, sized to admit a 2-layer x 32-d_model Transformer at the canonical 768-d DAPT embedding (~50.6k params) with headroom while rejecting the historical 4-layer x 64-d_model default (~250k params). The CLI exposes the override as `--no-param-cap`; the function signature carries `enforce_param_cap: bool = True` so callers from inside the package can switch the cap off explicitly when they have a documented reason (the trainer-wiring tests pass `enforce_param_cap=False` because the 4-d toy embedder pushes the historical default out of the cap budget even when the issue's concern about real-world capacity overshoot does not apply).
2. **Three baselines on the canonical fold protocol.**
   - `previous_stance(history)` predicts the last observed stance label. The lower bound of "did the Transformer extract anything?".
   - `rolling_majority(history, n=3)` predicts the modal label over the last three real meetings, with ties broken by recency. The standard moving-average naive baseline for autocorrelated classification.
   - `small_lstm_baseline(history)` is a 1-layer x 16-hidden LSTM trained on stance-only sequences. The "small honest baseline" the issue references, capped at <= 5k parameters by `assert_param_count_within_cap(cap=DEFAULT_LSTM_PARAM_CAP)` so the comparison is unambiguously between the Transformer arm and a structurally-smaller LSTM, not between the Transformer arm and a Transformer-lite.

   Each baseline emits a `BaselineResult` dataclass carrying directional accuracy + a 3x3 confusion matrix. `evaluate_previous_stance` / `evaluate_rolling_majority` / `evaluate_small_lstm` are the convenience helpers the trainer uses; `compare_against_transformer` assembles the lift / no-lift verdict.

3. **Lift threshold.** `LIFT_THRESHOLD_POINTS = 0.05`. The Transformer arm clears the threshold iff it beats the strongest baseline (max directional accuracy across the three) by `>= 5pp`. The verdict lands as the boolean `lift_vs_baseline` plus the numeric `delta_dir_acc` (Transformer dir-acc minus best-baseline dir-acc) and the string `baseline_used` (the name of the strongest baseline); all three persist to the bundle's `metrics.json` under the `lift_check` block.
4. **API surface.** `TrajectoryResponse` carries the three lift fields (`lift_vs_baseline: bool`, `delta_dir_acc: float | None`, `baseline_used: str | None`). The runtime singleton (`_load_lift_check`) reads the verdict from the bundle's `metrics.json` and surfaces it through `_TrajectoryState`. All three default to `None` / `False` for bundles that predate this ADR so the schema extension is back-compatible.

## Consequences

- New trajectory training runs targeting the canonical 768-d DAPT embedding default to the 2x32 Transformer arm (the only configuration that fits the 75k cap without an explicit override). The historical 4x64 default is reachable only via `--no-param-cap`, and the documented gate for using that flag is a downstream sweep demonstrating the `>= 5pp` lift threshold has already been cleared.
- `/analyze/trajectory` ships a stable `lift_vs_baseline` badge for every bundle, including legacy bundles (default-False until a re-train under #332 lands the `metrics.json` `lift_check` block).
- The baselines module is reusable by future trajectory follow-ups; `evaluate_*` and `compare_against_transformer` are stand-alone helpers that do not assume the trainer's call graph.
- The three trainer-wiring tests that previously exercised the 4x64 Transformer with a 4-d toy embedder now pass `enforce_param_cap=False` with an inline comment explaining the carve-out. The behaviour the cap exists to guard (real-world capacity overshoot at the 768-d DAPT embedding) is exercised by `test_oversized_4x64_transformer_trips_cap` in `tests/unit/test_trajectory_baselines.py`.
- The `LIFT_THRESHOLD_POINTS` constant + the override path through `--no-param-cap` together let a future downstream sweep (e.g. ensembling, distillation) lift the cap when the lift evidence justifies it. The ADR does not bake "2x32 forever" into the codebase; it bakes "2x32 by default, 4x64 only with evidence" into the codebase.
- The comparison table cells in `fed-pulse.wiki/06_Deep_Learning_Roadmap.md §6.16` are placeholders (`_pending sweep_`) at this commit. The next trajectory GPU sweep fills in the directional-accuracy + confusion-matrix cells from the bundle's persisted `metrics.json` block.
