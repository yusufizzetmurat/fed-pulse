FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[dev]"

# py-spy is the on-demand sampling profiler used by docs/perf/*.svg captures;
# it needs the same image so attaching to the uvicorn worker does not require
# a host-side install. The CPython binary in this image lets py-spy resolve
# native frames without the ptrace caveats of an alpine base.
RUN pip install --no-cache-dir py-spy>=0.3.14

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
