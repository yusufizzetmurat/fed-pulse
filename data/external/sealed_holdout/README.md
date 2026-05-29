# Sealed Post-Cutoff Holdout (R-14)

The canonical training package's `event_date` cutoff is 2024-12-31. Every
FOMC event filed after that date is genuinely held out: no row in any
walk-forward train / val / test partition, no row in any sweep / HP
grid, no row visible to any embedding / DAPT / encoder used during the
training programme.

This directory holds that reserve slice. The one-shot break-the-seal
protocol lets the final report cite a sealed-eval headline alongside
the in-protocol headline; the audit token below is the integrity
contract that gates it.

## Contents

- `fomc_2025.jsonl` — one JSON per line; fields `event_date`,
  `event_type` (`statement`|`minutes`), `text`, `url`,
  `scraped_at_utc`.
- `AUDIT_TOKEN` — JSON object with `seal_status`, `usage_count`,
  `last_accessed_utc`. Initial state: `sealed`, `0`, `null`. Flips on
  the first successful read.

## Pinned contents

`fomc_2025.jsonl` carries the eleven FOMC statements from 2025-01-29
through 2026-04-29 scraped from federalreserve.gov press-release
pages. The integrity sha256 is pinned in
`docs/sealed_holdout_protocol.md`; the table of statement dates and
the cross-check methodology live there. Re-hashing the file after any
edit must yield the pinned sha or the pre-registration is broken.

The loader still scans for legacy `# pragma: stub` markers and aborts
loudly if any are found, but no such markers exist on the pinned file.

## Read it through the loader

```python
from app.data.sealed_holdout_loader import load_sealed_holdout, audit_status

# safe at any time, no state change
audit_status()

# one-shot. After this call AUDIT_TOKEN flips to broken_by:<audit_caller>;
# subsequent calls raise SealedHoldoutAlreadyConsumedError.
rows = load_sealed_holdout(audit_caller="final-report-eval-2026-05-27")
```

See `docs/benchmark-policy.md §Sealed Holdout` for the full protocol.
