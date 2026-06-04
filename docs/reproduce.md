# Reproducing fed-pulse end-to-end

Reproducing the training pipeline from a fresh checkout on a machine with only Docker and an HF token. Pairs with `make reproduce-all` and `scripts/reproduce_all.py`.

## Prerequisites

- Docker + Compose v2
- Python 3.11 (host needs Python only to call the script outside the container; the Makefile target runs it inside the backend container)
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
3. Reports the wall time and the exit code.

Expected wall time on the 8 GB / 4 vCPU droplet: ~15 minutes (~10 min cold artefact pull, ~5 min one-epoch training). A warm cache cuts the artefact pull to seconds.

## What this validates

- The HF Hub training-package mirror is reachable and well-formed.
- The training package's fold manifest, splits, and `events.parquet` deserialise cleanly under the current backend code.
- The forecaster forward pass, loss, and optimiser step work end-to-end on the canonical schema.

What this does not validate:

- Full-epoch training quality (use `make train-batch` for that; see the Makefile and `docs/`).
- The bake-off comparisons (those need the per-encoder embedding caches, which lazy-fetch separately via `app.data.embedding_cache.ensure_local`).
- The frontend dashboard (runs against the deployed container at https://fedpulse.yusufizzetmurat.com/; see `docs/deploy.md`).

## Troubleshooting

- The canonical sweep runners (`scripts/run_dual_head_comparison.py`, `scripts/run_per_family_ablation.py`, `scripts/run_reproducibility_smoke.py`) auto-fall-back to eager mode on environments where `torch.compile` can't import a working triton; `TORCHDYNAMO_DISABLE=1` is the manual override.
