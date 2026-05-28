# ADR 0029 — Macro-regime conditioning + gated rich-feature block

The relationship between FOMC text and market reaction is regime-dependent. "Patient" is dovish in 2014 and neutral in 2022; "data-dependent" is a hawkish hold in 2018 and a dovish pivot signal in 2024. Without conditioning, a single supervised head averages across regimes and underfits the heterogeneity. The §16 STRETCH children #305 (surprise decomposition) and #306 (retrieval-augmented features) attack the SNR problem on the target side and the retrieval side; #307 attacks it on the input-conditioning side.

The methodology angle is the mechanism: a small strict-prior macro-regime indicator block plus a multiplicative gate that lets the model modulate the rich-feature contribution by the active regime. Whether the gate produces a headline lift on the small training corpus is a separate empirical question — the ADR ships the mechanism, not the cell. The rates-head infrastructure landed under #292 / #317 is the natural mounting point: the shared encoder feeds the rates heads through the same `ForecasterBase` backbone the gate sits on, so the conditioning surface is one Linear layer wide.

## What lands

An opt-in `--use-regime-conditioning` flag on `app.train_forecaster`. Default OFF — the new fields stay `None` and `FeatureVector.as_rich_list()` does NOT append the regime block; the per-bar feature size on the legacy / opt-out path is byte-identical to pre-#307. When ON, the loader writes a 3-scalar strict-prior macro-regime indicator block onto every bar of every supervised sequence, and the model factory mounts a multiplicative gate over the legacy rich-feature slice.

Three signed scalars in `{-1, 0, +1}`:

| Feature | Strict-prior construction |
| --- | --- |
| `policy_cycle_phase_score` | Walks MP-surprise rows whose `event_date < supervised_event.event_date` and within trailing 365 calendar days; reads `ff_target_prior` (band midpoint published the day before each meeting, post-#350 strict-prior; ADR 0024). Score is `+1` (hiking) when `(latest - earliest) >= +25 bp`, `-1` (cutting) when `<= -25 bp`, `0` otherwise. Cold-start (<2 eligible meetings) collapses to `0`. |
| `vix_level_regime_score` | Reads the supervised sequence's prior 20-bar `vix_close` series, all dated strictly before `event_date` by the events-builder's no-look-ahead contract. T-1 VIX (last entry) is bucketed against tertiles of the same prior-bar series. `+1` when T-1 > upper tertile; `-1` when below the lower tertile; `0` otherwise. |
| `term_spread_sign` | Sign of `tnx_close - irx_close` (10y minus 13-week, the same series the cross-asset slice already carries) on the last prior bar (T-1). `+1` for positive slope, `-1` for inversion, `0` for missing or exact zero. |

Plus a paired `macro_regime_features_missing` flag (`1.0` when the flag is off; `0.0` when populated).

## The gate

A single `nn.Linear(3, RICH_FEATURE_SIZE - FEATURE_SIZE)` mounted on `ForecasterBase` when `use_regime_conditioning=True`. The output is squashed through `2 * sigmoid(...)` so a zero-init weight + zero-init bias produces a mask identically `1.0` across every position — the gate is a strict no-op at start of training, and the forward pass is byte-identical to the no-gate path until gradients push the gate off identity.

The mask multiplies position-wise against the legacy rich-feature slice `[FEATURE_SIZE:RICH_FEATURE_SIZE]` of the per-bar input. The regime block tail (past `RICH_FEATURE_SIZE`) is left untouched so the recurrent core can still read the regime indicator alongside the gated rich block — the gate modulates the existing rich contribution; the recurrent core decides what to do with the regime signal itself.

The gate is on the input side, not the head side. A head-side gate would amplify variance (the head reads pooled-state representations whose magnitudes are already learnt), but an input-side gate can only attenuate or pass — it never adds variance to a slot the rich-feature scaler hasn't already normalised. The variance-attenuation argument is structural, not empirical: the literature on gating mechanisms (LSTM forget gates, transformer gated linear units) consistently picks the multiplicative form for the same reason. An additive form (`x_rich += linear(regime_block)`) would inject variance the per-fold `RobustScaler` can't undo because it was fit on the unmodified slice.

## Conditional emission

The schema deliberately deviates from the #306 pattern: `RICH_FEATURE_SIZE` (the module-level constant) does NOT widen when the flag is on. Instead, the regime block appends past `RICH_FEATURE_SIZE` by `FeatureVector.as_rich_list` only when `macro_regime_features is not None`. The opt-out default keeps the per-bar feature size byte-identical to pre-#307 for every downstream caller that iterates slices inside `[0:RICH_FEATURE_SIZE]`. The new helper `rich_feature_size_with_regime(use_regime: bool)` returns the wider size on the opt-in path so model factories widen the input projection in lockstep.

This is the structural lock: the default per-bar feature size on `--no-regime-conditioning` is exactly the pre-#307 width. Pre-#307 checkpoints with the legacy width deserialise unchanged on the default path; only `--use-regime-conditioning=True` runs need a fresh per-fold rich-feature scaler fit (no widening on the default `RICH_FEATURE_SIZE` slice means the existing `RichFeatureScalerParams.medians` / `iqrs` tuples still match the legacy width).

The per-fold `RobustScaler` slice `[FEATURE_SIZE:RICH_FEATURE_SIZE]` is unchanged: the regime block (past `RICH_FEATURE_SIZE`) is not scaled. The three scalars sit in `{-1, 0, +1}`, already in a tighter range than any rich-feature scalar after IQR; running a per-fold scaler on a categorical-encoded signal would only shift the median to 0 — a no-op mathematically.

## Strict-prior contract

Every input the regime composer reads is dated strictly before `event_date`. `policy_cycle_phase_score` reads `ff_target_prior` from MP-surprise rows whose own `event_date < supervised_event.event_date`, filtered at the helper boundary by an explicit strict-prior gate. `vix_level_regime_score` reads only the supervised sequence's prior 20 bars; the events-builder's `_assert_no_lookahead` already enforces strict-before-event-date bar dates. `term_spread_sign` reads `tnx_close` and `irx_close` from the last prior bar only.

The audit row in `docs/feature-provenance-audit.md` classifies the block as strict-prior; the regression test in `tests/regression/test_feature_provenance_as_of.py` is unchanged because the default (flag-off) path produces no new fields. The dedicated regression `test_macro_regime_features.py::test_provenance_audit_strict_prior_on_regime_features` locks the flag-on contract.

## Why not the alternatives

Concatenate the regime block onto the rich slice without a gate. Simplest path: append the three scalars and let the recurrent core's Linear projection mix them on its own. Rejected: any model with a wide enough input projection already does this; the claim is the explicit conditioning mechanism, not a wider input.

Gate on the pooled hidden state instead of the per-bar input. Move the gate after `_encode` so it modulates the pooled `(B, hidden_size)` representation downstream of the recurrent core. Rejected: head-side position lets the gate amplify any direction in the pooled state (the gate output is unbounded after the Linear projection runs through the head), defeating the variance-attenuation guarantee.

Train a separate regime-prediction head and feed its softmax as a conditioning input. Rather than a strict-prior data-derived indicator, learn a supervised regime classifier off the existing rich block and use its softmax as the gate input. Rejected: the supervised regime label would have to be derived from the same forward-target window the main head reads, so the regime head would either duplicate the main supervisory signal (no lift) or train against a different label whose construction would itself need an ADR. The strict-prior data-derived path is smaller and cleaner.

Five or seven indicators instead of three (dollar-index level regime, real-rate quartile, credit-spread tier). Rejected on the small-corpus tax: every additional feature is a degree of freedom the gate has to fit on ~250 training events. Three indicators cover the three independent dimensions the literature consistently identifies as dominant regime axes — policy direction, vol level, recession proxy — without overfitting the gate.

## Framing per #334

The #334 substitution finding showed that stacking small feature families on top of the existing rich-feature input can give negative interaction lift. The regime block is itself a small feature family, so the headline lift may be negative, neutral, or positive. The mechanism — gated rich-feature block conditioned on strict-prior macro indicators — is the contribution; the cell-by-cell delta on `--use-regime-conditioning` is operator-driven via the Runpod sweep.

The code path ships because: the mechanism is defensible on its own; the block is small (3 scalars + 1 flag), so the input dimension doesn't blow up and the comparison sweep is cheap; the gate's zero-init makes OFF / ON behaviour byte-identical at step 0, so a flag flip produces an interpretable post-training delta; and the flag-off default keeps every existing sweep byte-identical, so the change is non-disruptive on the canonical grid.

## Downstream effects

The gate is a small Linear layer (3 inputs × `RICH_FEATURE_SIZE - FEATURE_SIZE` outputs ≈ 240 parameters). State_dict gains two new keys (`regime_gate.weight`, `regime_gate.bias`) on the opt-in path. Existing checkpoints with the legacy width deserialise unchanged on `--no-regime-conditioning`; the opt-in path needs a fresh sweep because the model's input projection widens. The per-fold rich-feature scaler stays anchored on `[FEATURE_SIZE:RICH_FEATURE_SIZE]` (unchanged slice).

Compute: pure-Python composition over already-loaded data (MP-surprise lookup + per-event prior bars). No new I/O; per-event cost is dominated by a dict scan over ≤120 trailing meetings. Full canonical training package stays well under the +5 s budget mentioned in the spec. CI smoke stays under 60 s on CPU.

The headline cell against `--use-regime-conditioning` is a Runpod follow-up. §16 populates with both modes side-by-side as the numbers arrive.

## References

- `backend/app/training/regime_features.py`, `backend/app/training/loaders.py`
- `backend/app/models/forecaster_base.py` (gate Linear + multiplicative apply inside `prepare_recurrent_input`)
- `backend/app/models/config.py` (`FeatureVector.macro_regime_features`; `rich_feature_size_with_regime`)
- ADR 0024 (#350 strict-prior MP-surprise — the `ff_target_prior` source), ADR 0028
- Issues #292, #307, #334
