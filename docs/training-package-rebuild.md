# Training-package rebuild path

End-to-end recipe for producing a fresh `events.parquet` with every
feature family populated so the audit (`make audit-training-package`)
reports zero degraded sweep flags.

The pinned TP referenced by the canonical sweeps was built before
several optional feature families landed (#236 garch residual, #291
rates panel, #443 statement delta, #444 vote tally, #482 per-asset
targets, #483 multi-horizon vol). The optional builders in this repo
can populate every column the trainer supports, but they have to be
invoked explicitly. `pipeline_data_prep` stops at the registry parquet
and does not call `event_dataset_builder`.

Operator-facing reference. Commands assume `FRED_API_KEY` is set in
`.env` and the docker compose `backend` service is buildable.

## Step 0 — Pre-flight

```
test -f .env && grep -q FRED_API_KEY .env  # required for FRED-backed sidecars
docker compose build backend                # one-time, picks up dependency lock
```

## Step 1 — Pull source corpora (no GPU needed)

Each adapter ships an idempotent CLI. Re-running is a no-op when the
cache file already exists with non-empty rows; pass `FORCE=1` to refresh.

```
make pull-op-fed                           # Keith et al. 2025 OpFed CSV (1044 sentences)
make pull-beige-book                       # ~400 Beige Book national summaries since 1970
make pull-press-conferences                # FOMC press-conference PDFs since 2011
make pull-speeches                         # chair + governor speeches, 2006–present
make pull-testimonies                      # congressional testimonies, 2006–present
make pull-regional-research                # Liberty Street Economics archive
```

`LIMIT=N` caps each walk for smoke tests; `YEARS="2024 2025"` narrows
speeches/testimonies to specific years.

## Step 2 — Build FRED-backed sidecars

```
make build-rates-panel RATES_PANEL_START=2008-01-01 RATES_PANEL_END=today
make build-mp-surprises
make build-macro-state
```

These produce `data/external/fred/rates_panel.parquet`,
`data/external/fred/mp_surprises.parquet`, and
`data/external/fred/macro_state.parquet`.

## Step 3 — Run the standard data-prep pipeline

```
make data-prep DATASET_VERSION=v3 FEATURE_VERSION=v1 OWNER="$(whoami)" \
    TRAINING_PACKAGE_ID=tp_v3_<your-version-tag>
```

Writes `registry_normalized.parquet` and the split parquets under
`data/processed/<TRAINING_PACKAGE_ID>/`. Does not produce
`events.parquet`; that comes in the next step.

## Step 4 — Build events.parquet with every optional feature family

```
make build-events-parquet \
    TRAINING_PACKAGE_ID=tp_v3_<your-version-tag> \
    RATES_PANEL_PATH=data/external/fred/rates_panel.parquet \
    PER_ASSET_CACHE_DIR=data/external/yfinance
```

`RATES_PANEL_PATH` enables the #291 rates-complex feature columns
(`pre_meeting_yield_*`, `yield_*_change_5d`). `PER_ASSET_CACHE_DIR`
enables the #482 per-asset target columns
(`forward_realized_vol_10d_{gspc,ndx,dji,dxy,vix,eurusd,usdjpy,gbpusd}`).
Pass empty values to either to skip the corresponding block.

The statement-delta, vote-tally, and multi-horizon-vol blocks are
computed unconditionally when the source rows carry the necessary text
or have the prior-statement window available; no flag is required.

## Step 5 — Add GARCH(1,1) residual columns

```
make garch-baseline TRAINING_PACKAGE_ID=tp_v3_<your-version-tag>
```

Adds `forward_realized_vol_10d_garch_baseline` and
`forward_realized_vol_10d_garch_residual` to `events.parquet` in
place. Required by the `--vol-target-mode garch_residual` sweep arm.

## Step 6 — Audit the rebuild

```
make audit-training-package TRAINING_PACKAGE_ID=tp_v3_<your-version-tag>
```

The audit prints:

- a REQUIRED check on the supervised target (`forward_realized_vol_10d`),
- per-family population counts for every optional block,
- a TRAINER FLAG IMPACT section naming the specific sweep flags
  that will silently no-op given the populated columns,
- the `event_kind` distribution (corpus diversity),
- a sidecar inventory (press-conf Q&A, SEP projections).

A clean rebuild leaves the TRAINER FLAG IMPACT section reading
"All optional families have at least one populated column." If a
family is still empty, the missing builder is named in the report.

## Step 7 — Push to Hugging Face + pin

The Hub-write step is operator-only (HF write token required); the
canonical destination is
`hf://datasets/yusufizzetmurat/fed-pulse-training-package`. After the
push, capture the commit SHA and pin it in
`backend/app/models/registry.yaml` so reproducibility metadata stays
truthful.

## Step 8 — Re-run the sweeps

The headline canonical-arm sweep is unchanged:

```
make canonical-comparison TRAINING_PACKAGE_ID=tp_v3_<your-version-tag>
```

The previously-degraded arms now run against real data:

- `--vol-target-mode garch_residual` (uses the GARCH residual column
  from Step 5)
- `--rates-heads <subset>` (consumes the rates-panel target columns)
- `--symbol-embedding-dim N` with `N > 0` (uses the per-asset target
  columns)

The Option-A multi-axis block (#447) activates automatically when
`axis_time_label` / `axis_certain_label` are populated, which the
gtfintechlab ingest in Step 3 handles.
