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

RUN /opt/venv/bin/pip install --no-cache-dir "."


FROM builder-runtime AS builder-dev

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
