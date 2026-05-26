# Reproducing fed-pulse end-to-end

This page walks through reproducing the training pipeline from a fresh checkout on a machine that has only Docker + an HF token. It pairs with `make reproduce-all` and `scripts/reproduce_all.py`.

## Prerequisites

- Docker + Compose v2
- Python 3.11 (the host needs Python only if you want to call the script outside the container; the Makefile target runs it inside the backend container)
- A Hugging Face PAT with `read` scope on `yusufizzetmurat`

## Steps

```sh
git clone https://github.com/yusufizzetmurat/fed-pulse.git
cd fed-pulse

# Either:
cp .env.example .env
# and fill in HF_TOKEN, or:
export HF_TOKEN=hf_xxx

# Pull artefacts from HF Hub + run the smoke training pass.
make reproduce-all
```

The target runs `python scripts/reproduce_all.py` inside the backend container with `FED_PULSE_REPRODUCE_SMOKE=1` set. The script:

1. Resolves `hf://datasets/yusufizzetmurat/fed-pulse-training-package` via `huggingface_hub.snapshot_download` and copies the result to `data/processed/canonical/`.
2. Calls `python -m app.train_forecaster --training-package-id canonical --seed 11 --epochs 1` for a single-epoch smoke training pass.
3. Reports the wall time + the exit code.

Expected wall time on the 8 GB / 4 vCPU droplet: ~15 minutes (~10 min cold artefact pull, ~5 min one-epoch training). A warm cache cuts the artefact pull to seconds.

## What this validates

- The HF Hub training-package mirror is reachable and well-formed.
- The training package's fold manifest, splits, and `events.parquet` deserialise cleanly under the current backend code.
- The forecaster forward pass + loss + optimiser step works end-to-end on the canonical schema.

What this does **not** validate:

- Full-epoch training quality (use `make train-batch` for that — see `CLAUDE.md`).
- The bake-off comparisons (those need the per-encoder embedding caches, which lazy-fetch separately via `app.data.embedding_cache.ensure_local`).
- The frontend dashboard (that runs against the deployed container; see `docs/deploy.md`).
