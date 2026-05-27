# `text_path_ab_canonical.json` — why `per_bar` and `broadcast_static` match byte-for-byte

The `per_bar` and `broadcast_static` arms in this artefact have **identical** per-trial metric values across all 25 cells (5 seeds × 5 folds) and **identical** summary stats. This is the genuine output of the sweep, not a copy-paste bug. The two arms collapse to the same forward computation under the current loader wiring.

## Why the collapse is exact (not "close")

The `per_bar` adapter call is stateless (Linear → LayerNorm → GELU, no dropout, no per-position state). When the per-bar input tensor `(B, T, in_dim)` carries the *same* pooled vector at every bar `t ∈ [0, T)`, the adapter emits the *same* projected vector at every bar — which is exactly the `(B, out_dim)` slot that `broadcast_static` produces and then `unsqueeze(1).expand(-1, T, -1)`s across the sequence axis. The two `prepare_recurrent_input` branches produce bit-equivalent `(B, T, lstm_input_size)` tensors before the recurrent core ever runs.

The math is pinned by `test_per_bar_parity_with_broadcast_static_when_constant_across_bars` in `tests/unit/test_text_path_arms.py` — that test loads the same `state_dict` into both arms, runs a forward with a tile-replicated per-bar tensor, and asserts `allclose(atol=1e-6)`.

## Why this canonical sweep tile-replicates

Per ADR 0017 §"Arm A — per-bar text features":

> Default smoke path: tile-replicate the prior-N FOMC pool across the lookback so the contract holds without a new corpus ingestion. A future per-day loader populates `text_per_bar` directly and slots into the same tensor.

The canonical training package (`tp_v3_macro_aug_2026_05_25_fwd_strict_sentiment_market_core_v1.1_epv1_v1.0`) does not carry a populated `FeatureVector.text_per_bar` field. The `build_per_bar_text_tensor` helper in `backend/app/training/loaders.py` consequently falls back to its tile-replicate branch (the anchor's `text_embedding_pooled` repeated across every bar). The per-bar wiring is exercised end-to-end — model constructor, forward path, loader payload shape — but the input variance the recurrent core consumes is zero across the sequence axis, so the per-bar projection collapses to the broadcast-static projection.

## What this means for §6.15

The A/B null between `per_bar` and `broadcast_static` is the expected outcome given the canonical training package's text-per-bar payload. It is **not** evidence that the `per_bar` code path is broken; it is evidence that the input signal feeding it is identical to the broadcast-static signal. The honest framing in §6.15 is: "Arm A is plumbed end-to-end and produces byte-identical output to `broadcast_static` when fed a tile-replicated per-bar payload (the canonical training package's current state); a real per-bar A/B requires a daily-frequency text corpus the package does not yet ingest."

## Regenerating

This artefact is GPU-bound and was produced by `scripts/run_text_path_ab.py` on Runpod. If a future training package carries a populated `text_per_bar` field, regenerate with:

```
make text-path-ab TRAINING_PACKAGE_ID=<id>
```

The same parity test (`test_per_bar_parity_with_broadcast_static_when_constant_across_bars`) will continue to hold for tile-replicated inputs; the `per_bar` row of the artefact will start diverging from `broadcast_static` once the loader emits non-constant per-bar payloads.

## Cross-references

- ADR 0017 — `docs/adr/0017-text-path-architecture-decision.md`
- Issue #327 (original A/B), #389 (per_bar constructor regression test), #390 (this audit).
- `tests/unit/test_text_path_arms.py::test_per_bar_parity_with_broadcast_static_when_constant_across_bars` — invariant test.
- `tests/regression/test_text_path_ab_artefact_parity.py` — artefact-level guard (added with this change).
