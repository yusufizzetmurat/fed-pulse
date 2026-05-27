# Demo + report figures

Companion to `docs/reproduce.md`. Walks through driving the dashboard end-to-end, capturing the panel screenshots the report cites, and regenerating the per-head + cross-bank figure set. Both scripts are deterministic — the same backend state plus the same viewport gives diffable PNGs run-to-run.

## Prerequisites

- Docker + Compose v2 (for the local backend stack), or a reachable deployed origin.
- Python 3.11 with `playwright` installed (`pip install playwright && playwright install chromium`).
- Pillow on the host running `build_thesis_figures.py` — already a backend dep, so the script also works inside the backend container if you would rather not install on the host.

## End-to-end demo (screenshots)

```sh
# 1. Backend + frontend on the default `make dev` ports.
make dev
# Wait for `http://localhost:3000` to respond and the backend `/health` to
# go green (≈ 30 s once the image is built).

# 2. Drive the dashboard, capture screenshots to ../fed-pulse.wiki/assets/demo/.
python scripts/demo_end_to_end.py
```

The script captures five panels — statement decomposition, market reaction, historical analogs, trajectory, settings checkpoints — and writes one PNG per panel with a UTC timestamp suffix so a re-run does not overwrite the previous capture. Add `--headed` to watch the run; `--keep-open` pauses after the last capture so the live UI is inspectable before tear-down.

Against the deployed droplet:

```sh
python scripts/demo_end_to_end.py \
    --frontend-url https://fedpulse.yusufizzetmurat.com \
    --backend-url https://fedpulse.yusufizzetmurat.com/api
```

The frontend resolves its API base URL at build time from `NEXT_PUBLIC_API_URL` (see `frontend/lib/analyze/api.ts`), so the `--backend-url` flag is informational against a prebuilt bundle. Pass `--in-repo` to write captures under `docs/demo-screenshots/` instead of the wiki repo.

## Report figures

```sh
python -m scripts.figures.build_thesis_figures
```

Default output is `../fed-pulse.wiki/assets/figures/`. Pass `--in-repo` to write to `docs/figures/` instead. The script renders four figures:

| Figure | Source |
| --- | --- |
| `architecture.png` | wiki §3 (system architecture) |
| `dual_head_comparison.png` | `backend/artifacts/experiments/dual_head_comparison_canonical.json` |
| `text_path_ab.png` | `backend/artifacts/experiments/text_path_ab_canonical.json` |
| `cross_bank_ladder.png` | wiki §6.14 (transcribed; no JSON artefact on disk) |

Each PNG ships a paired `.caption.txt` carrying the reproducibility header (commit SHA at render time, canonical training-package id, source artefact path). The report can drop the caption verbatim.

To regenerate one figure rather than the full set:

```sh
python -m scripts.figures.build_thesis_figures --only dual-head
```

## Putting it together

On a fresh checkout the full reproduce-from-scratch flow is:

```sh
# 1. Pull the canonical training package + run a 1-epoch smoke training pass.
make reproduce-all

# 2. Bring the dashboard up.
make dev

# 3. Capture the demo screenshots.
python scripts/demo_end_to_end.py

# 4. Rebuild the report figures.
python -m scripts.figures.build_thesis_figures
```

The demo screenshots land under `../fed-pulse.wiki/assets/demo/` and the figures under `../fed-pulse.wiki/assets/figures/`. Commit both to the wiki repo separately from the main repo.

## Troubleshooting

- The screenshot script raises `TimeoutError: locator "Multi-axis interpretation" not found` if the backend has no multi-axis checkpoint mounted. The panel renders an empty-state card with that string anyway, so the timeout almost always means the backend has not finished cold-starting — re-run after `/health` reports `"status": "ready"`.
- `make dev` builds the frontend bundle with `NEXT_PUBLIC_API_URL=http://localhost:8000`. Against the droplet the prod build bakes the public origin in instead; do not point a local frontend at a remote backend by changing the env var at runtime — rebuild the image.
- Pillow's default fonts vary by host. The script falls back to PIL's bitmap default if no truetype font is found, which renders correctly but at lower legibility. Install the DejaVu or system Arial fonts to match the published figures.
