# Deployment guide

This page is the operator runbook for the fed-pulse droplet. It covers the one-time setup, the deploy contract, and the rollback path. The deploy workflow under `.github/workflows/deploy.yml` is the automated half; this guide is everything outside that workflow.

## Topology

- Hostname: `fedpulse.yusufizzetmurat.com` (CNAME on the operator's domain pointed at the droplet's IP)
- Reverse proxy: Caddy 2 in its own container, automatic Let's Encrypt for the hostname
- Backend: FastAPI (uvicorn, 2 workers) on internal `:8000`
- Frontend: Next.js standalone server on internal `:3000`
- All three services live in `compose.prod.yml` and share the `fed-pulse:prod` image (built from `Dockerfile.prod`)
- HF Hub stores every model artefact; the droplet pulls the hot path eagerly at boot and lazy-fetches the alternative encoders
- The HF cache lives on a named volume (`hf-cache`) so the 2-4 GB encoder bundles survive image rebuilds

## Provisioning

Provision an [8 GB / 4 vCPU Basic Regular droplet](https://www.digitalocean.com/pricing/droplets) on DigitalOcean (around $48/month). The boot budget targets that hardware; smaller droplets will OOM during the eager-pull step.

### 1. DNS first

Create an A record `fedpulse.yusufizzetmurat.com -> <droplet ip>` and wait for propagation before doing anything else. Caddy's first-boot Let's Encrypt challenge needs the hostname to resolve to the droplet, and the cert request will rate-limit if it fails repeatedly. Verify with:

```sh
dig +short fedpulse.yusufizzetmurat.com
```

Wait until that returns the droplet IP (usually under 60 seconds after creating the A record).

### 2. One-time setup on the droplet

As `root` on the fresh droplet:

```sh
# Docker + Compose
apt-get update && apt-get install -y docker.io docker-compose-plugin git

# Clone into the canonical location the deploy workflow expects
mkdir -p /opt && cd /opt
git clone https://github.com/yusufizzetmurat/fed-pulse.git
cd fed-pulse
git checkout main

# Env file for the backend container (contains HF_TOKEN + FRED_API_KEY)
mkdir -p /etc/fed-pulse /etc/fed-pulse/data

# The runtime container runs as a non-root `fedpulse` user (UID 10001
# per Dockerfile.prod). Without an explicit chown the bind-mounted
# /etc/fed-pulse/data sits at root:root on the host and the container
# cannot write to /data/db on first boot — uvicorn aborts with
# `PermissionError: [Errno 13]` and the healthcheck never goes green.
chown -R 10001:0 /etc/fed-pulse/data
chmod -R u+rwX,g+rwX /etc/fed-pulse/data

cat <<'EOF' > /etc/fed-pulse/.env
HF_TOKEN=hf_xxx
FRED_API_KEY=xxx
ACME_EMAIL=admin@yusufizzetmurat.com
EOF
chmod 600 /etc/fed-pulse/.env

# Authorise the GH Actions SSH key (paste the public side of the
# DROPLET_SSH_KEY secret; private side stays in GH).
cat your_id_ed25519.pub >> /root/.ssh/authorized_keys

# First build + start. Subsequent deploys go through the GH Actions
# workflow; this seeds the image and the persistent caches.
docker compose -f compose.prod.yml up -d --build

# Sanity check
curl -sf https://fedpulse.yusufizzetmurat.com/health
```

## GitHub Actions secrets

The deploy workflow needs three secrets on the repository (Settings -> Secrets and variables -> Actions):

| Secret | Contents | Where it is used |
|---|---|---|
| `DROPLET_SSH_KEY` | Private ed25519 key (the public side lives in `/root/.ssh/authorized_keys` on the droplet) | `deploy.yml` writes it to `~/.ssh/id_ed25519` before sshing in |
| `HF_TOKEN` | Hugging Face PAT with `read` scope on `yusufizzetmurat` | Threaded into the droplet env file; consumed by the entrypoint to pull hot-path artefacts |
| `HF_TOKEN_WRITE` | Hugging Face PAT with `write` scope on `yusufizzetmurat` | Used by the manual `hf-push` workflow only; never lives on the operator's laptop |
| `FRED_API_KEY` | FRED API key | Consumed by `app.services.fred_client` for the rates panel refresh |

## Deploy contract

Pushing to `main` triggers `.github/workflows/deploy.yml`. The workflow does:

1. Polls the GitHub Actions API for the `ci.yml` run targeting the same commit sha and refuses to proceed unless it concluded `success`. Times out at 10 minutes.
2. Writes `DROPLET_SSH_KEY` to a temp file with `chmod 600`.
3. Sshes to the droplet as `root` (override via the `ssh_user` workflow_dispatch input).
4. Records the current droplet HEAD as the prior-known-good sha (BEFORE fetching `main` from origin), fetches `main`, hard-resets to it, and runs `docker compose -f compose.prod.yml up -d --build`.
5. Polls `https://fedpulse.yusufizzetmurat.com/health` until success or a 60-second timeout.
6. On health-probe success: writes the now-current sha to `/etc/fed-pulse/last-deploy.sha` (the next rollback target).
7. On health-probe failure: hard-resets to the recorded prior sha and rebuilds, then re-runs the smoke probe. Exit 1 on the original failure, exit 2 if the rollback also failed (the droplet is then in an inconsistent state and needs manual recovery).

`dev` continues to be the integration branch; the deploy workflow does not fire on `dev` pushes. Releases happen by running the `promote.yml` workflow_dispatch — it fast-forwards `main` to a specified `dev` sha (default: `dev` HEAD), which in turn triggers `deploy.yml`. The promote workflow uses `git push --force-with-lease` so a stale view of `main` (e.g. a hotfix that landed in parallel) refuses the push instead of silently rewinding.

## Pushing artefacts to HF Hub

The one-time and follow-up artefact pushes run server-side via the `hf-push` workflow (`.github/workflows/hf-push.yml`). Trigger it manually from the GH Actions tab:

- `dry-run=true` prints the upload plan without contacting HF Hub. Inspect the per-file listing for accidental secrets / git metadata before flipping to `false`.
- `repos` is an optional comma-separated list of kinds (`encoder,forecaster,retrieval,trajectory,rates_heads,training_package,embedding_caches`). Leaving it empty pushes everything via `--all`.
- The script reads `HF_TOKEN_WRITE` from the GH secret store, so the operator's laptop never holds the write-scoped token.

After a successful run, copy the printed commit shas into `backend/app/models/registry.yaml` under each `artefacts.<key>.revision` field, commit + push to `dev`, then run `promote.yml` to roll the pinned shas onto `main`.

## Rollback (manual)

If the automated rollback in the workflow itself fails (e.g. an SSH outage), ssh to the droplet and run:

```sh
cd /opt/fed-pulse
PRIOR=$(cat /etc/fed-pulse/last-deploy.sha)
git reset --hard "${PRIOR}"
docker compose -f compose.prod.yml up -d --build
curl -sf https://fedpulse.yusufizzetmurat.com/health
```

## Manual smoke checklist

After a deploy lands, the operator should:

- `curl -sf https://fedpulse.yusufizzetmurat.com/health` returns 200
- `curl -sf https://fedpulse.yusufizzetmurat.com/api/symbols` returns the symbol list
- The browser smoke: open `https://fedpulse.yusufizzetmurat.com/`, post a sample FOMC sentence, confirm `/analyze` renders cards

## Known constraints

- First request after a cold boot of a non-eager encoder pulls a ~600 MB parquet from HF Hub; allow 15-30 seconds. Subsequent requests hit the local cache.
- Eager-pull at boot dominates the cold-start time. Budget under 90 seconds on the 8 GB droplet for the canonical hot path. The `hf-cache` named volume survives image rebuilds, so a deploy that does not bump artefact pins typically skips the eager pull entirely.
- Caddy's Let's Encrypt cache lives in the `caddy_data` named volume; do not `docker compose down -v` in production or the next boot will retry the ACME challenge from scratch.
- Container processes drop to UID 10001 (`fedpulse`) via `s6-setuidgid` before exec'ing uvicorn / node, so files written into `/etc/fed-pulse/data` land under that UID on the host. Do not `chown root` over them — the next container start will refuse to write.
