# 0003 — Market-data snapshot vs live yfinance

**Status:** accepted
**Date:** 2026-05-12

## Context

`benchmark-policy.md` enforces version immutability: a published `run_id` is never reused and protocol changes require a version bump. The market-data column of the feature vector is part of that protocol. If the underlying close prices change between training and a re-evaluation (yfinance retro-adjusts splits and dividends, Yahoo can revise historical bars), the immutability claim is a fiction.

FRED is licence-clean and authoritative for macro series. yfinance is the only free option for full-history equity index closes — FRED's `SP500` and `DJIA` series are capped at a 10-year rolling window by S&P/DJ licensing and cannot cover a 2010-start benchmark.

## Decision

- `scripts/snapshot_market_data.py` pulls every benchmarked ticker once and writes a parquet under `data/raw/market/<symbol>.parquet`. A SHA-256 entry plus the source (`fred` or `yfinance`) lands in `data/raw/market/SOURCES.lock`.
- `services/market_data.py` dispatches on `FED_PULSE_MARKET_SOURCE`: `snapshot` reads the committed parquet, `live` (default) keeps yfinance for the dashboard.
- Routing per ticker: FRED for VIX, treasury yields, USD broad index, gold AM fix, WTI, fed funds, NASDAQ composite. yfinance for `^GSPC` and `^DJI`.

## Consequences

- Benchmark reproducibility binds to the committed parquet + SHA, not to a live API.
- Live `/analyze` calls still hit yfinance — the dashboard's freshness requirement is separate from the benchmark's reproducibility requirement.
- Regenerating a snapshot from a different point in time will produce a different SHA; either commit the new parquet (and document the cutover) or pin the existing one. Never silently update.
