# Sealed-holdout pre-registration protocol (#501)

This document is the pre-registered contract for the post-cutoff sealed
holdout at `data/external/sealed_holdout/fomc_2025.jsonl`. Once committed
with witness sign-off it must not be edited; subsequent breaks of the
seal (the one allowed final-submission read) link back to the commit
SHA that introduced this file as the integrity anchor.

The companion technical documentation lives in
`data/external/sealed_holdout/README.md`. This file is the procedural
contract, not the technical one.

## Holdout definition

- **File**: `data/external/sealed_holdout/fomc_2025.jsonl`.
- **Row schema**: one JSON object per line. Fields: `event_date`
  (ISO yyyy-mm-dd, all values strictly after 2024-12-31), `event_type`
  (`statement` | `minutes`), `text` (scraped from federalreserve.gov,
  no derivation), `url` (source URL), `scraped_at_utc` (scrape
  timestamp).
- **Cutoff contract**: every row's `event_date` is strictly after the
  canonical training package's cutoff (2024-12-31). No row appears in
  any walk-forward train / val / test partition, no row appears in any
  sweep / HP grid, no row appears in any embedding / DAPT / encoder
  used during the training programme.

### Stub state at pre-registration time

The file currently ships stub rows carrying a leading
`# pragma: stub` marker in the `text` field. The integrity sha pinned
below is captured at the moment the stubs are replaced with real
scraped text. Until that swap the audit script (see "Integrity
verification" below) intentionally reports a sha mismatch — that is
the desired state, not a bug, because reporting a sealed-eval headline
against stubbed text would be a methodology defect.

## Pre-declared evaluation metrics

These are the metrics that will be reported against the sealed
holdout, declared before any model has seen the file:

- **Classification head**:
  - `regime_f1_macro` (macro-averaged F1 over the 3 vol-regime bins).
  - Per-class precision / recall / F1 (3 rows).
  - 3x3 confusion matrix.
  - Macro ROC-AUC and macro PR-AUC.
- **Regression head**:
  - RMSE on `log(forward_realized_vol_10d)`.
  - MAE on `log(forward_realized_vol_10d)`.
  - R^2 on `log(forward_realized_vol_10d)`.
- **Conformal coverage** at the pre-declared significance level
  α = 0.20 (i.e. nominal 80% coverage band).

No other metric will be added post-hoc. Adding one would invalidate
the pre-registration.

## Number of model runs against the holdout

**Exactly one (1).** The single allowed read is the final-submission
inference pass, executed against the checkpoint frozen by the final
sweep batch on the dev / val partitions. The full sweep grid is
selected and frozen on the dev splits; the holdout is read only after
no methodology decision remains open.

Any second read invalidates the pre-registration and the sealed-eval
headline must be removed from the final report.

## Seal-break protocol

1. The operator verifies the integrity sha against
   `data/external/sealed_holdout/fomc_2025.jsonl` (see "Integrity
   verification" below).
2. The operator records the code revision (git SHA of `HEAD`) the
   inference run will execute under.
3. The operator runs the single inference pass and writes the metrics
   from the pre-declared list above to
   `backend/artifacts/experiments/sealed_eval_holdout_<timestamp>.json`.
4. The operator increments `usage_count` and sets
   `last_accessed_utc` in `data/external/sealed_holdout/AUDIT_TOKEN`.
   The post-run `usage_count` must equal exactly `1`.
5. The witness named in the sign-off section below countersigns the
   result by committing the metrics file with their authenticated
   identity.

## Integrity verification

The script that computes and verifies the holdout file's sha256:

```
shasum -a 256 data/external/sealed_holdout/fomc_2025.jsonl
```

The pre-registered sha will be committed to this file in a follow-up
commit at the moment the stub replacement lands. Until then the
sha-pin block below is empty.

### Pinned sha256 (post stub replacement)

```
<INTENTIONALLY BLANK — see "Stub state at pre-registration time">
```

To pin: replace the placeholder above with the output of `shasum -a 256
data/external/sealed_holdout/fomc_2025.jsonl` as run on the commit
SHA that lands the real scraped rows. The pinning commit must
reference issue #501.

## Code revision that defines this holdout

- Pre-registration commit (this document) — see commit SHA in PR #N
  (replace `#N` with the merging PR's number once the squash SHA is
  known).
- Sealed-holdout loader: `backend/app/data/sealed_holdout_loader.py`.
- Audit-token state machine: `data/external/sealed_holdout/AUDIT_TOKEN`.

## Witness sign-off

To be filled in by the supervisor on commit:

- **Witness identity**: ________________________________________
- **Witness sign-off date** (UTC): ____________________________
- **Confirmation**: I have read the pre-registered metrics list,
  the one-run cap, and the seal-break protocol above. I countersign
  the pre-registration as valid for the final submission of the
  fed-pulse thesis project.

(The witness commits this completed block as a follow-up to the
PR that introduces the file. The follow-up commit must reference
issue #501 and must be authored by the witness's authenticated
identity for the audit trail.)
