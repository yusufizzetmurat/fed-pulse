# Deployment guide

This page is the operator runbook for the fed-pulse droplet. It covers the one-time setup, the deploy contract, and the rollback path. The deploy workflow under `.github/workflows/deploy.yml` is the automated half; this guide is everything outside that workflow.

## Topology

- Hostname: `fedpulse.yusufizzetmurat.com` (CNAME on the operator's domain pointed at the droplet's IP)
- Reverse proxy: Caddy 2 in its own container, automatic Let's Encrypt for the hostname
- Backend: FastAPI (uvicorn, 2 workers) on internal `:8000`
- Frontend: Next.js standalone server on internal `:3000`
- All three services live in `compose.prod.yml` and share the `fed-pulse:prod` image (built from `Dockerfile.prod`)
- HF Hub stores every model artefact; the droplet pulls the hot path eagerly at boot and lazy-fetches the alternative encoders

## Provisioning

Provision an [8 GB / 4 vCPU Basic Regular droplet](https://www.digitalocean.com/pricing/droplets) on DigitalOcean (around $48/month). The boot budget targets that hardware; smaller droplets will OOM during the eager-pull step.

One-time setup commands as `root` on the fresh droplet:

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

DNS prerequisite: the operator must point an A record `fedpulse.yusufizzetmurat.com -> <droplet ip>` before the first boot — Caddy needs the hostname to resolve to the droplet for the Let's Encrypt challenge to succeed.

## GitHub Actions secrets

The deploy workflow needs three secrets on the repository (Settings -> Secrets and variables -> Actions):

| Secret | Contents | Where it is used |
|---|---|---|
| `DROPLET_SSH_KEY` | Private ed25519 key (the public side lives in `/root/.ssh/authorized_keys` on the droplet) | `deploy.yml` writes it to `~/.ssh/id_ed25519` before sshing in |
| `HF_TOKEN` | Hugging Face PAT with `read` scope on `yusufizzetmurat` (or `write` if also pushing artefacts via the workflow) | Threaded into the droplet env file; consumed by the entrypoint to pull hot-path artefacts |
| `FRED_API_KEY` | FRED API key | Consumed by `app.services.fred_client` for the rates panel refresh |

## Deploy contract

Pushing to `main` triggers `.github/workflows/deploy.yml`. The workflow does:

1. Writes `DROPLET_SSH_KEY` to a temp file with `chmod 600`.
2. Sshes to the droplet as `root` (override via the `ssh_user` workflow_dispatch input).
3. Records the prior HEAD into `/etc/fed-pulse/last-deploy.sha`, fetches `main`, hard-resets to it, and runs `docker compose -f compose.prod.yml up -d --build`.
4. Polls `https://fedpulse.yusufizzetmurat.com/health` until success or a 60-second timeout.
5. On health-probe failure: hard-resets to the recorded prior sha and rebuilds. The job exits non-zero.

`dev` continues to be the integration branch; the deploy workflow does not fire on `dev` pushes. Releases happen by running the `promote.yml` workflow_dispatch — it fast-forwards `main` to a specified `dev` sha (default: `dev` HEAD), which in turn triggers `deploy.yml`.

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
- Eager-pull at boot dominates the cold-start time. Budget under 90 seconds on the 8 GB droplet for the canonical hot path.
- Caddy's Let's Encrypt cache lives in the `caddy_data` named volume; do not `docker compose down -v` in production or the next boot will retry the ACME challenge from scratch.
