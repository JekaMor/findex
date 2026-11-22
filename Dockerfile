# syntax=docker/dockerfile:1
#
#   docker build -t findex .
#   docker run --rm -p 8000:8000 findex                     # the baked-in index
#   docker run --rm -p 8000:8000 -e INDEX_PATH=/data/my.bin -v "$PWD:/data" findex
#
# Stage 1 installs the locked dependencies and builds the demo index; stage 2
# keeps only the virtualenv, the corpus and the index, and runs as a non-root user.

ARG PYTHON=3.14

FROM python:${PYTHON}-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

# Dependencies first, from uv.lock only: this layer stays cached until the lock changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Then the project itself, installed as a regular package (not editable).
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# The demo corpus (40 public-domain books, see the corpus-1 release) and its index.
ARG CORPUS_URL=https://github.com/JekaMor/findex/releases/download/corpus-1/gutenberg-40.tar.gz
ARG CORPUS_SHA256=28416048767420b82b7930d8fb3f3272cdd17adc8b5bb291613a0342e4853106
ADD --checksum=sha256:${CORPUS_SHA256} ${CORPUS_URL} /tmp/corpus.tar.gz
RUN mkdir data \
 && tar -xzf /tmp/corpus.tar.gz -C data \
 && /app/.venv/bin/findex index data --out index.bin --executor serial


FROM python:${PYTHON}-slim
RUN useradd --create-home --uid 10001 findex
WORKDIR /app
COPY --from=builder --chown=findex:findex /app/.venv /app/.venv
COPY --from=builder --chown=findex:findex /app/data /app/data
COPY --from=builder --chown=findex:findex /app/index.bin /app/index.bin

# Configuration is environment only; a platform overrides PORT, and anything else.
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    INDEX_PATH=/app/index.bin \
    HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=INFO
USER findex
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"]
CMD ["findex", "serve"]
