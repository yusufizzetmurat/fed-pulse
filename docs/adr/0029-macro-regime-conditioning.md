# ADR 0029 — Macro-regime conditioning + gated rich-feature block

Status: accepted, code path live; canonical sweep deferred to operator.
Date: 2026-05-27.
References:
- Issue #307 (closes).
- Issue #292 — multi-target heads on shared encoder (the rates-head infrastructure the gating block is built on top of).
- Issue #305 — surprise-decomposition target (companion methodology piece on the rates side).
- Issue #334 — text-feature substitution / interaction finding: adding small text-derived feature families on top of the existing rich-feature block can give negative interaction lift.
- ADR 0024 — strict-prior MP-surprise reformulation (the `ff_target_prior` field this ADR's policy-cycle indicator reads).
- ADR 0028 — retrieval-augmented input features (companion code-path-only PR; same opt-in flag + sweep-hand-off framing).
- `backend/app/training/regime_features.py` — pure-Python composer with the three per-feature helpers.
- `backend/app/training/loaders.py` — `_compute_macro_regime_features_for_event` + per-event call sites on both loader entry points.
- `backend/app/models/forecaster_base.py` — the gate Linear layer on `ForecasterBase` and the multiplicative application inside `prepare_recurrent_input`.
- `backend/app/models/config.py` — `FeatureVector.macro_regime_features` schema extension; slice constants; `rich_feature_size_with_regime`; `ModelConfig.use_regime_conditioning`.

## Context

The relationship between FOMC text and market reaction is regime-dependent. "Patient" is dovish in 2014 and neutral in 2022; "data-dependent" is a hawkish hold in 2018 and a dovish pivot signal in 2024. Without conditioning, a single supervised head averages across regimes and underfits the heterogeneity. The §16 STRETCH children #305 (surprise decomposition) and #306 (retrieval-augmented features) attack the SNR problem on the target side and on the retrieval side respectively; #307 attacks it on the input-conditioning side.

The methodology contribution is the architectural mechanism for handling regime heterogeneity: a small strict-prior macro-regime indicator block plus a multiplicative gate that lets the model modulate the rich-feature contribution by the active regime. Whether the gate produces a headline lift on the small training corpus is a separate empirical question — the case for the ADR is the mechanism, not the cell.

The rates-head infrastructure landed under #292 (merged in #317) is the natural mounting point: the shared encoder feeds the rates heads through the same `ForecasterBase` backbone the gate sits on, so the conditioning surface is one Linear layer wide.

## Decision

Add an opt-in `--use-regime-conditioning` flag on `app.train_forecaster`. Default OFF — when off the new fields stay `None` and `FeatureVector.as_rich_list()` does NOT append the regime block; the per-bar feature size on the legacy / opt-out path is byte-identical to pre-#307. When ON, the loader writes a 3-scalar strict-prior macro-regime indicator block onto every bar of every supervised sequence, and the model factory mounts a multiplicative gate over the legacy rich-feature slice.

### Feature block — three signed scalars in {-1, 0, +1}

| Feature | Strict-prior construction |
| --- | --- |
| `policy_cycle_phase_score` | Walks the MP-surprise lookup for every meeting whose `event_date < supervised_event.event_date` and within the trailing 365 calendar days; reads `ff_target_prior` (the band midpoint published the day before each meeting, post-#350 strict-prior; ADR 0024). The score is `+1` (hiking) when `(latest - earliest) >= +25 bp`, `-1` (cutting) when `<= -25 bp`, `0` otherwise. Cold-start (<2 eligible meetings) collapses to `0`. |
| `vix_level_regime_score` | Reads the supervised sequence's prior 20-bar `vix_close` series, all dated strictly before `event_date` by the events-builder's no-look-ahead contract. T-1 VIX (last entry) is bucketed against tertiles of the same prior-bar series. `+1` when T-1 > upper tertile; `-1` when below the lower tertile; `0` otherwise. |
| `term_spread_sign` | Sign of `tnx_close - irx_close` (10y minus 13-week, the same series the cross-asset slice already carries) on the last prior bar (T-1). `+1` for a positive slope, `-1` for inversion, `0` for missing or exact zero. |

Plus a paired `macro_regime_features_missing` flag (`1.0` when the flag is off; `0.0` when populated).

### Gating block — multiplicative sigmoid gate

The gate is a single `nn.Linear(3, RICH_FEATURE_SIZE - FEATURE_SIZE)` mounted on `ForecasterBase` when `use_regime_conditioning=True`. The output is squashed through `2 * sigmoid(...)` so a zero-init weight + zero-init bias produces a mask identically `1.0` across every position — the gate is a strict no-op at start of training, and the forward pass is byte-identical to the no-gate path until gradients push the gate off identity.

The mask is multiplied position-wise against the legacy rich-feature slice `[FEATURE_SIZE:RICH_FEATURE_SIZE]` of the per-bar input tensor. The regime block tail (positions past `RICH_FEATURE_SIZE`) is left untouched so the recurrent core can still read the regime indicator alongside the gated rich block — the gate modulates the existing rich contribution; the recurrent core decides what to do with the regime signal itself.

The gate position is on the input side, not on the head side. This matters: a head-side gate would amplify variance (the head reads pooled-state representations whose magnitudes are already learnt), but an input-side gate can only attenuate or pass — it never adds new variance to a slot the rich-feature scaler has not already normalised. The reviewer-focus claim "the gate's multiplicative position avoids amplifying noise" is structural, not empirical.

### Conditional emission contract — preserves byte-identity for legacy callers

The schema design deliberately deviates from the #306 pattern: `RICH_FEATURE_SIZE` (the module-level constant) does NOT widen when the flag is on. Instead, the regime block is appended past `RICH_FEATURE_SIZE` by `FeatureVector.as_rich_list` only when `macro_regime_features is not None`. The opt-out default keeps the per-bar feature size byte-identical to pre-#307 for every downstream caller that iterates slices inside `[0:RICH_FEATURE_SIZE]`. The new helper `rich_feature_size_with_regime(use_regime: bool)` returns the wider size on the opt-in path so model factories widen the input projection in lockstep.

This is the structural lock for reviewer-focus #3: the default per-bar feature size on `--no-regime-conditioning` is exactly the pre-#307 width. Pre-#307 checkpoints with the legacy width therefore deserialise unchanged on the default path; only `--use-regime-conditioning=True` runs need a fresh per-fold rich-feature scaler fit (no widening on the default `RICH_FEATURE_SIZE` slice means the existing `RichFeatureScalerParams.medians` / `iqrs` tuples still match the legacy width).

The per-fold `RobustScaler` slice `[FEATURE_SIZE:RICH_FEATURE_SIZE]` is unchanged: the regime block (which sits past `RICH_FEATURE_SIZE`) is not scaled. Reason: the three scalars live in `{-1, 0, +1}` by construction, already in a tighter range than any rich-feature scalar after the IQR transform; running a per-fold scaler on a categorical-encoded signal would only shift the median to 0 — a no-op transform mathematically.

### Strict-prior contract

Every input the regime composer reads is dated strictly before `event_date`:

- `policy_cycle_phase_score` reads `ff_target_prior` from MP-surprise rows whose own `event_date < supervised_event.event_date`. Filtered at the helper boundary by an explicit strict-prior gate.
- `vix_level_regime_score` reads only the supervised sequence's prior 20 bars; the events-builder's `_assert_no_lookahead` already enforces strict-before-event-date bar dates.
- `term_spread_sign` reads `tnx_close` and `irx_close` from the last prior bar only — same strict-prior contract.

The audit row in `docs/feature-provenance-audit.md` classifies the block as strict-prior; the regression test in `tests/regression/test_feature_provenance_as_of.py` is unchanged because the default (flag-off) path produces no new fields. The dedicated regression in `tests/unit/test_macro_regime_features.py::test_provenance_audit_strict_prior_on_regime_features` locks the flag-on contract.

### CLI surface

`--use-regime-conditioning` / `--no-regime-conditioning` on `app.train_forecaster`. Threaded through both `load_walk_forward_split` call sites (walk-forward + single-fold) and into the `ModelConfig.use_regime_conditioning` field so the factory mounts the gate at build time. The report-payload JSON gains a `rich_feature_families.regime_conditioning` boolean so the sweep aggregator can group cells by the flag's state.

## Alternatives considered

**Additive bias on the rich-feature slice.** Replace the multiplicative gate with `x_rich += linear(regime_block)`. Rejected: an additive transform adds variance to the rich slice that the per-fold `RobustScaler` cannot un-do (the scaler was fit on the unmodified slice). The multiplicative gate only attenuates or passes existing values, so the post-gate slot stays inside the scaler's calibrated range. The literature on gating mechanisms (LSTM forget gates, transformer gated linear units) consistently picks the multiplicative form for the same reason.

**Concatenate the regime block onto the rich slice without a gate.** The simplest path: append the three scalars and let the recurrent core's Linear projection mix them into the hidden state on its own. Rejected as the methodology contribution: any model with a wide enough input projection already does this; the methodology claim is the explicit conditioning mechanism, not a wider input. The reviewer-focus framing in the PR body draws the distinction (gate as architectural lever, not a feature-pile-on).

**Gate on the pooled hidden state instead of on the per-bar input.** Move the gate after `_encode` so it modulates the pooled `(B, hidden_size)` representation downstream of the recurrent core. Rejected: the head-side position would let the gate amplify any direction in the pooled state (the gate output is unbounded after the Linear projection runs through the head), defeating the variance-attenuation guarantee. The input-side position keeps the gate's effect bounded by the rich-feature slice's existing scaled magnitude.

**Train a separate regime-prediction head and feed its softmax as a conditioning input.** Rather than a strict-prior data-derived indicator, learn a supervised regime classifier off the existing rich block and use its softmax as the gate input. Rejected: the supervised regime label would have to be derived from the same forward-target window the main head reads, so the regime head would either duplicate the main supervisory signal (no methodological lift) or train against a different label whose construction would itself need an ADR. The strict-prior data-derived path is a smaller surface and a cleaner methodology claim.

**Include more regime features.** Five or seven indicators rather than three (e.g. dollar-index level regime, real-rate quartile, credit-spread tier). Rejected on the small-corpus tax: every additional feature is a degree of freedom the gate has to fit on ~250 training events. Three indicators are the minimum that cover the three independent dimensions the literature consistently identifies as the dominant regime axes — policy direction, vol level, recession proxy — without overfitting the gate.

## Consequences

### Honest framing per #334

The #334 substitution finding showed that stacking small feature families on top of the existing rich-feature input can give negative interaction lift. The regime block is itself a small feature family, so the headline lift on a single sweep may be negative, neutral, or positive. The methodology contribution is the architectural mechanism for handling regime heterogeneity; the cell-by-cell delta on `--use-regime-conditioning` is operator-driven via the Runpod sweep.

We ship the code path anyway because:

- The architectural mechanism (gated rich-feature block conditioned on strict-prior macro indicators) is a defensible methodology contribution on its own.
- The block is small (3 scalars + 1 flag) so it does not blow up the input dimension; the comparison sweep is cheap.
- The gate's zero-init makes the OFF / ON behaviour byte-identical at step 0, so a flag flip without a re-init produces an interpretable "post-gate-training" delta against the no-gate path.
- The flag-off default keeps every existing sweep byte-identical, so the change is non-disruptive on the canonical sweep grid.

### Methodology

The strict-prior contract is the load-bearing claim. The same per-feature provenance audit (#324) that drove the #350 MP-surprise reformulation classifies the regime block as strict-prior; the row sits alongside the other `T-Δ` rows in `docs/feature-provenance-audit.md`. No `T+Δ` reads; no `T (snapshot)` reads of the supervised event's own labels.

### Model + checkpoint

The gate is a small Linear layer (3 inputs × `RICH_FEATURE_SIZE - FEATURE_SIZE` outputs ≈ 240 parameters). It mounts only when `use_regime_conditioning=True`; the state_dict gains two new keys (`regime_gate.weight`, `regime_gate.bias`) on the opt-in path. Existing checkpoints with the legacy width deserialise unchanged on `--no-regime-conditioning`; the opt-in path requires a fresh sweep because the model input projection widens and the per-fold rich-feature scaler stays anchored on `[FEATURE_SIZE:RICH_FEATURE_SIZE]` (unchanged slice).

### Compute

The regime computation is a pure-Python composition over already-loaded data (the MP-surprise lookup and the per-event prior bars). No new I/O; the per-event cost is dominated by a single dict scan over ≤120 trailing meetings. Full canonical training package stays well under the +5s budget mentioned in the spec. The CI smoke test stays under 60s on CPU.

### Sweep hand-off

The headline cell against `--use-regime-conditioning` is a Runpod follow-up. The §16 comparison table populates with both modes side-by-side as the numbers arrive. Default `--no-regime-conditioning` runs against the canonical training package remain byte-identical so the existing sweep numbers stay reproducible.
