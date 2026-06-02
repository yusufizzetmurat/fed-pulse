FROM python:3.11-slim@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0 AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"


FROM base AS builder-runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip

COPY pyproject.toml ./
COPY requirements.lock ./

# Align with Dockerfile.prod: install third-party deps from the
# hash-pinned lock file so dev images cannot silently drift off prod's
# resolved versions. The bare ``.`` install that follows registers the
# local package without touching the resolved tree (every transitive
# dep is already satisfied by the lock install above).
RUN /opt/venv/bin/pip install --no-cache-dir --require-hashes -r requirements.lock \
    && /opt/venv/bin/pip install --no-cache-dir --no-deps "."


FROM builder-runtime AS builder-dev

# Dev extras (pytest, mypy, etc.) live only in pyproject.toml; no
# matching ``requirements-dev.lock`` ships in-tree yet, so install
# them from the project metadata. Base deps were already pinned in
# ``builder-runtime``.
RUN /opt/venv/bin/pip install --no-cache-dir ".[dev]"


FROM base AS runtime

COPY --from=builder-runtime /opt/venv /opt/venv
COPY . .

EXPOSE 8000

# The eager-pull shim hydrates ``MODELS_DIR`` from HF (eager: true
# artefacts) when files are missing on disk. It never raises out, so a
# fresh container without HF_TOKEN still boots and the cold-start
# bootstrap in main.py picks up the gap on first /analyze.
CMD ["sh", "-c", "python -m app.boot.eager_pull && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]


FROM base AS dev

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder-dev /opt/venv /opt/venv
COPY . .

EXPOSE 8000

CMD ["sh", "-c", "python -m app.boot.eager_pull && exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"]
