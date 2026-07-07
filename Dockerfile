# ── Stage 1: base image ────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── Stage 2: dependencies ──────────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ── Stage 3: application ───────────────────────────────────────────────────
FROM deps AS app

# Copy source package
COPY src/ src/
COPY configs/ configs/
COPY pyproject.toml .

# Install gdo package in editable mode
RUN pip install -e . --no-deps

# Health check: the package imports cleanly
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import gdo; print('OK')" || exit 1

# Training CLI. Override the config at run time, e.g.:
#   docker run gdo --config configs/adam_mnist.yaml
ENTRYPOINT ["python", "-m", "gdo"]
CMD ["--help"]
