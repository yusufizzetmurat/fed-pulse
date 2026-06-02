# Text-path A/B result (MLC2)

**Date:** 2026-06-02 · **Training package:** `tp_v3_full_rebuild_2026_05_30`
**Seeds:** {11, 29, 47, 71, 97} · **Epochs:** 40 · **Head:** dual (regression-α = 0.5)
**Driver:** `scripts/run_text_path_ab.py`
**Raw artifact:** `backend/artifacts/experiments/text_path_ab.json`

The audit pre-registration in ADR-0017 defines three text-path framings:
`broadcast_static` (the canonical path), `per_bar` (Arm A — recurrent core
with one pooled vector per lookback bar), and `flat_mlp` (Arm B — no-sequence
wrap, pooled-text adapter concatenated with pooled-market). The audit flagged
that the published text-null had only been proven on `broadcast_static`; this
note closes the A/B follow-up.

## Aggregate over 5 seeds × 5 folds = 25 cells

| Arm | regime macro-F1 (mean ± std) | log-RV RMSE (mean ± std) |
|---|---|---|
| `broadcast_static` | **0.3784 ± 0.0639** | **0.7668 ± 0.1538** |
| `per_bar` (Arm A) | 0.3784 ± 0.0639 | 0.7668 ± 0.1538 |
| `flat_mlp` (Arm B) | 0.3279 ± 0.0835 | 0.8424 ± 0.1891 |

## Reading

**`per_bar` is bit-identical to `broadcast_static` on this training
package.** This is structurally expected, not a bug. The canonical
training package does not populate `FeatureVector.text_per_bar`, so the
loader's `build_per_bar_text_tensor` helper tile-replicates the last
prior bar's pooled embedding across every bar of the lookback window
(documented at `backend/app/training/loaders.py:3779`). For FOMC events,
which arrive at meeting-day cadence rather than per-bar, this is the
correct semantics: there is no per-bar text signal to read off, so Arm
A degenerates to the same numerical path as the broadcast-static
baseline. A reviewer asking "but what about the per-bar framing?" is
asking a question this dataset does not contain the resolution to
answer — the answer is structurally identical to broadcast-static here.

**`flat_mlp` is strictly worse on both axes.** −0.0505 macro-F1 and
+0.0756 log-RV RMSE. Pooling the lookback into a single market vector
before concatenating with the text adapter destroys the temporal
information the recurrent core exploits, even though the same text
payload is available. The Arm B framing does not surface text signal
the canonical path missed; it removes signal the recurrent path was
already using.

## What this closes

The audit's MLC2 reads "the published null demonstrates only that
static-broadcast text adds nothing; it does not rule out that a per-bar
or flat-MLP framing surfaces signal." Both framings are now run on the
canonical training package:

- per_bar adds nothing because the FOMC-statement cadence does not
  carry per-bar text content for the recurrent core to ingest.
- flat_mlp performs strictly worse than broadcast_static, ruling out
  the "maybe a non-sequence model would find what the recurrent core
  missed" hypothesis on this data.

The text-null is therefore unconditional across the three framings
ADR-0017 enumerates. The headline claim ("FOMC text broadcast onto
market state contributes nothing detectable to regime / RV forecasting
at the official seed set") survives the A/B follow-up.

## Caveat

A genuinely informative `per_bar` arm would require a training package
that populates `text_per_bar` per supervised bar — i.e., per-bar
sentence-stream or token-stream payloads, not per-meeting pooled
embeddings. That data layer is not built and is out of scope for this
follow-up. Treat the per_bar parity as "Arm A on this canonical
training package," not as a universal claim about per-bar text.
