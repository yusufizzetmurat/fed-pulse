FROM python:3.11-slim@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

# This image is the dev / research target consumed by `make dev-cpu` and
# `make dev-gpu`; the CMD at the bottom runs uvicorn with --reload. There
# is no separate production stage today, so the [dev] extra (pytest,
# mypy, ruff, hypothesis, py-spy) is intentionally baked in — running
# tests, type-checks, and flamegraph captures all happen against the
# same image. When a production deployment lands, split this into a
# multi-stage build with a runtime layer that installs only `"."`.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[dev]"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
