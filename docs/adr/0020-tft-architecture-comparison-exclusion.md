# ADR 0020 — TFT excluded from the canonical architecture comparison

Status: accepted, in production (as of merge).
Date: 2026-05-27.
References:
- Issue #331.
- `backend/app/models/config.py` — `FORECASTER_ARCHITECTURES`, `CANONICAL_SWEEP_ARCHITECTURES`, `TFT_EXCLUSION_REASON`.
- `backend/app/models/factory.py` — `DeprecationWarning` raised when `architecture="tft"` is requested.
- `backend/app/models/tft.py` — `TFTEncoder` module, retained for back-compat with existing checkpoints.
- `scripts/run_regime_architecture_sweep.py` — `_DEFAULT_ARCHITECTURES` (TFT absent).
- `Makefile` — `forecaster-sweep`, `forecaster-sweep-exhaustive`, `forecaster-sweep-shuffled-control` (TFT absent from default architecture lists).
- `fed-pulse.wiki/06_Deep_Learning_Roadmap.md §6.7` — footnote citing this ADR.
- `tests/unit/test_tft_excluded_from_canonical_sweep.py` — guard against TFT silently re-entering canonical sweep defaults.

## Context

The in-repo TFT implementation (`backend/app/models/tft.py`) ports the encoder-side pieces of the published Temporal Fusion Transformer: per-timestep Variable-Selection Network (VSN), GRN-gated residual blocks, and a multi-head self-attention block over the selected representation. The forecaster pipeline mean-pools that encoder output and routes the pooled vector through the generic classifier head the other seven architectures share.

TFT is designed to emit predictions via its native quantile head paired with the VSN. The published recipe routes per-timestep variable importances + static-covariate enrichment through that head end-to-end; the head is the part the architecture's inductive bias is organised around. Mean-pooling the encoder output and feeding a generic classifier head strips half of TFT's design — the comparison row that lands at 0.3803 macro-F1 in §6.6 measures the encoder under a head it was not designed for, not the architecture as published.

Two paths are available:

- **Option A — implement TFT faithfully.** Port the native quantile output + the VSN-aware head, retune random-search HP at the new head's surface, rerun §6.7 with a faithful TFT row. Multi-day GPU + ML-engineering work; STRETCH-tier per the §6 roadmap.
- **Option B — drop TFT from the canonical comparison.** Document the exclusion in this ADR + a §6.7 footnote, remove TFT from the canonical sweep defaults, preserve the 0.3803 result in the wiki only as historical record. 1-day SHOULD-tier path per the §6 roadmap.

The §6 roadmap allows either; advisor scope does not require both. Shipping a hostile-evaluation row of TFT as a published comparison number is the option the brief explicitly rules out.

## Decision

Option B. TFT is excluded from the canonical architecture comparison.

The module `backend/app/models/tft.py` and the `"tft"` identifier in `FORECASTER_ARCHITECTURES` are retained so existing checkpoints that recorded `architecture="tft"` continue to load and the encoder unit tests at `tests/unit/test_tft.py` keep running. A new constant `CANONICAL_SWEEP_ARCHITECTURES` excludes `"tft"` and is the tuple new sweep code iterates. `build_forecaster` raises a `DeprecationWarning` (`TFT_EXCLUSION_REASON`) when asked to build a TFT instance so a future sweep that mis-includes the identifier surfaces the exclusion in the trainer logs rather than silently regressing the comparison.

The default architecture lists in `scripts/run_regime_architecture_sweep.py` and in the Makefile sweep targets (`forecaster-sweep`, `forecaster-sweep-exhaustive`, `forecaster-sweep-shuffled-control`) drop `"tft"`. The trainer CLI's `--architecture` / `--architectures` flags still accept `"tft"` so an operator can opt back in for a one-off investigation; the canonical published comparison does not include it.

`fed-pulse.wiki/06_Deep_Learning_Roadmap.md §6.7` carries a footnote citing this ADR. The 0.3803 figure from §6.6 stays in the wiki as historical record of the deprecated-target / generic-head measurement; no headline cell cites TFT going forward.

## Consequences

- Canonical sweep numbers no longer cite TFT. The four architectures cited in the §6.7 post-correction table (`gru`, `tcn`, `transformer`, `lstm_attn`) remain the published comparison set.
- Existing TFT checkpoints continue to load — the back-compat shim on the architecture identifier is preserved on purpose. The `TFTEncoder` module is not deleted.
- A future faithful-TFT reimplementation (Option A) is filed as a STRETCH follow-up; the gate for re-adding TFT to the canonical comparison is a faithful quantile-head implementation, not a re-run of the existing generic-head wiring.
- The unit test at `tests/unit/test_tft_excluded_from_canonical_sweep.py` pins `CANONICAL_SWEEP_ARCHITECTURES` and the regime-arch-sweep runner's default list. Re-adding TFT to the canonical surface requires updating this ADR + the test together.
- The `DeprecationWarning` raised at `build_forecaster` time means an operator who explicitly opts back in via `--architectures tft` still gets the build but receives a clear log message — silent regressions of the canonical comparison are blocked.
