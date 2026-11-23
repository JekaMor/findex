# syntax=docker/dockerfile:1
#
#   docker build -t findex .
#   docker run --rm -p 8000:8000 findex                     # the baked-in index
#   docker run --rm -p 8000:8000 -e INDEX_PATH=/data/my.bin -v "$PWD:/data" findex
#
# Stage 1 installs the locked dependencies (with the `semantic` extra), builds the
# demo index, fetches the embedding model and the passage embeddings; stage 2 keeps
# only what runs, as a non-root user, with no network access needed at run time.

ARG PYTHON=3.14

FROM python:${PYTHON}-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    FASTEMBED_CACHE_PATH=/app/models
WORKDIR /app

# Dependencies first, from uv.lock only: this layer stays cached until the lock changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra semantic --no-install-project

# Then the project itself, installed as a regular package (not editable).
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra semantic --no-editable

# The demo corpus (40 public-domain books, see the corpus-1 release) and its index.
ARG CORPUS_URL=https://github.com/JekaMor/findex/releases/download/corpus-1/gutenberg-40.tar.gz
ARG CORPUS_SHA256=28416048767420b82b7930d8fb3f3272cdd17adc8b5bb291613a0342e4853106
ADD --checksum=sha256:${CORPUS_SHA256} ${CORPUS_URL} /tmp/corpus.tar.gz
RUN mkdir data \
 && tar -xzf /tmp/corpus.tar.gz -C data \
 && /app/.venv/bin/findex index data --out index.bin --executor serial

# The model for query embeddings (all-MiniLM-L6-v2, ONNX), downloaded once here.
RUN /app/.venv/bin/python -c "from findex.semantic import FastEmbedder; FastEmbedder()"

# Passage embeddings, computed once with `findex embed` (34,429 passages take ~30
# minutes on a laptop CPU, too long for a build). They refer to documents by file
# name, so this index's own doc_ids do not matter.
ARG EMBEDDINGS_URL=https://github.com/JekaMor/findex/releases/download/embeddings-1/index-emb.tar.gz
ARG EMBEDDINGS_SHA256=a9c03a5910b64297cdafd6344de9f0f82368c6939a566124f2c5ab3f2c6f2eb5
ADD --checksum=sha256:${EMBEDDINGS_SHA256} ${EMBEDDINGS_URL} /tmp/emb.tar.gz
RUN tar -xzf /tmp/emb.tar.gz -C /app


FROM python:${PYTHON}-slim
RUN useradd --create-home --uid 10001 findex
WORKDIR /app
COPY --from=builder --chown=findex:findex /app/.venv /app/.venv
COPY --from=builder --chown=findex:findex /app/data /app/data
COPY --from=builder --chown=findex:findex /app/index.bin /app/index.emb.* /app/
COPY --from=builder --chown=findex:findex /app/models /app/models

# Configuration is environment only; a platform overrides PORT, and anything else.
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    INDEX_PATH=/app/index.bin \
    HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=INFO \
    FASTEMBED_CACHE_PATH=/app/models \
    HF_HUB_OFFLINE=1
USER findex
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"]
CMD ["findex", "serve"]
