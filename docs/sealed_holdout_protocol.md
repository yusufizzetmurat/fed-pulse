# Sealed-holdout pre-registration protocol (#501)

The pre-registered contract for the post-cutoff sealed holdout at
`data/external/sealed_holdout/fomc_2025.jsonl`. Once committed, this
file must not be edited. The one allowed final-submission read of the
seal links back to the commit SHA that introduced this file as the
integrity anchor.

Companion technical documentation lives in
`data/external/sealed_holdout/README.md`. That file covers the
technical surface; this one is the procedural contract.

## Holdout definition

- **File**: `data/external/sealed_holdout/fomc_2025.jsonl`.
- **Row schema**: one JSON object per line. Fields: `event_date`
  (ISO yyyy-mm-dd, all values strictly after 2024-12-31), `event_type`
  (`statement` | `minutes`), `text` (scraped from federalreserve.gov,
  no derivation), `url` (source URL), `scraped_at_utc` (scrape
  timestamp).
- **Cutoff contract**: every row's `event_date` falls strictly after
  the canonical training package's cutoff (2024-12-31). No row appears
  in any walk-forward train / val / test partition, no row appears in
  any sweep / HP grid, no row appears in any embedding / DAPT /
  encoder used during the training programme.

### Holdout contents at pre-registration time

Eleven FOMC statements, scraped 2026-05-29 from federalreserve.gov
press-release pages and pinned in this same commit:

| Date | Action | URL |
|---|---|---|
| 2025-01-29 | Hold 4¼–4½% | `pressreleases/monetary20250129a.htm` |
| 2025-03-19 | Hold 4¼–4½% (QT taper) | `pressreleases/monetary20250319a.htm` |
| 2025-05-07 | Hold 4¼–4½% | `pressreleases/monetary20250507a.htm` |
| 2025-06-18 | Hold 4¼–4½% | `pressreleases/monetary20250618a.htm` |
| 2025-07-30 | Hold 4¼–4½% (Bowman + Waller dissent for cut) | `pressreleases/monetary20250730a.htm` |
| 2025-09-17 | Cut to 4–4¼% (Miran dissent for deeper cut) | `pressreleases/monetary20250917a.htm` |
| 2025-10-29 | Cut to 3¾–4% (QT ends Dec 1) | `pressreleases/monetary20251029a.htm` |
| 2025-12-10 | Cut to 3½–3¾% (reserve-management buys begin) | `pressreleases/monetary20251210a.htm` |
| 2026-01-28 | Hold 3½–3¾% (new voter slate) | `pressreleases/monetary20260128a.htm` |
| 2026-03-18 | Hold 3½–3¾% | `pressreleases/monetary20260318a.htm` |
| 2026-04-29 | Hold 3½–3¾% (3-way dissent) | `pressreleases/monetary20260429a.htm` |

Each text was extracted from the `div#article p` selector of the
press-release page and joined with `\n\n`. The `Implementation Note
issued <Month D, YYYY>` trailing line on each row was cross-checked
against the row's `event_date` at pin time. Minutes (released ~3
weeks after each meeting) are excluded from this pre-registration;
adding them post-hoc would invalidate the contract.

## Pre-declared evaluation metrics

The metrics reported against the sealed holdout, declared before any
model has seen the file:

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

No other metric will be added post-hoc. Adding one invalidates the
pre-registration.

## Number of model runs against the holdout

**Exactly one (1).** The single allowed read is the final-submission
inference pass, executed against the checkpoint frozen by the final
sweep batch on the dev / val partitions. The full sweep grid is
selected and frozen on the dev splits; the holdout is read only after
no methodology decision remains open.

Any second read invalidates the pre-registration, and the sealed-eval
headline must then be removed from the final report.

## Seal-break protocol

1. Verify the integrity sha against
   `data/external/sealed_holdout/fomc_2025.jsonl` (see "Integrity
   verification" below).
2. Record the code revision (git SHA of `HEAD`) the inference run will
   execute under.
3. Run the single inference pass and write the metrics from the pre-
   declared list above to
   `backend/artifacts/experiments/sealed_eval_holdout_<timestamp>.json`.
4. Increment `usage_count` and set `last_accessed_utc` in
   `data/external/sealed_holdout/AUDIT_TOKEN`. The post-run
   `usage_count` must equal exactly `1`.

## Integrity verification

The script that computes and verifies the holdout file's sha256:

```
shasum -a 256 data/external/sealed_holdout/fomc_2025.jsonl
```

### Pinned sha256

```
2d38de11ce020b119574012a735ee1ffafd5842ed111ee5141572176cdd281c0
```

Captured at the commit that introduces the real scraped statements
into `fomc_2025.jsonl`. Any future commit that mutates that file
without producing the same sha after re-hash breaks the
pre-registration. CI should hard-fail on a sha drift; until that
check lands, manual verification before the one allowed seal-break
run is the contract.

## Code revision that defines this holdout

- Pre-registration commit (this document) — see the commit SHA on the
  PR that introduced this file.
- Sealed-holdout loader: `backend/app/data/sealed_holdout_loader.py`.
- Audit-token state machine: `data/external/sealed_holdout/AUDIT_TOKEN`.
