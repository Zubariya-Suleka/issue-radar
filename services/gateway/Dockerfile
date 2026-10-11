# Gateway image (D-018). Build context is the repo root (it needs the workspace lockfile):
#   docker compose build gateway
#
# Two stages: the builder has uv and build caches; the final image only gets the finished
# virtualenv. Model weights are NOT baked in (they are mounted from the HF cache on G:), so
# the image stays ~2 GB (mostly CPU torch) and a model change doesn't need a rebuild.

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# The venv holds third-party dependencies only, never our own package. It then changes only
# when the lockfile does, so the ~1.8 GB layer that copies it below stays cached across code
# edits. (Installing the package into the venv made every code edit re-copy all of it.)
COPY pyproject.toml uv.lock ./
COPY services/gateway/pyproject.toml services/gateway/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace \
        --package issue-radar-gateway --extra local --extra safety


FROM python:3.12-slim
# Don't run as root: if the process is compromised, it can't modify the image's system files.
RUN useradd --create-home --uid 1000 gateway
COPY --from=builder /app/.venv /app/.venv
# Our code: a few KB, its own layer, the only one rebuilt on a code edit.
COPY services/gateway/src /app/src
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1
WORKDIR /app
USER gateway
EXPOSE 8000

# Docker marks the container unhealthy if /health stops answering. start-period covers
# model loading plus warm-up: ~75 s from cold on the G: hard disk (D-018). slim has no curl,
# so use Python.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"]

CMD ["uvicorn", "gateway.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
