"""The findex web service: a JSON API and a small HTML search page.

    findex serve --index web.bin         # or: uvicorn findex.web.app:app

API: ``GET /search``, ``GET /docs/{doc_id}``, ``GET /stats``, ``GET /health``
(OpenAPI page at ``/docs``). Pages: ``GET /`` and ``GET /doc/{doc_id}``.
"""

import logging
import time
import unicodedata
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from findex.corpus import document_text
from findex.model import Index
from findex.query import QueryError, parse
from findex.search import ScorerName
from findex.semantic import Mode, Passages, Semantic, default_prefix
from findex.snippets import Part, first_hit, highlight_parts
from findex.store import open_index
from findex.web import models
from findex.web.service import Page, document_slice, search_page, snippet_text
from findex.web.settings import Settings, get_settings

log = logging.getLogger("findex.web")
HERE = Path(__file__).parent
templates = Jinja2Templates(directory=HERE / "templates")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Open the index once for the life of the process (Lab 3's ``open_index``)."""
    settings = get_settings()  # a bad INDEX_PATH fails here, at start-up
    started = time.perf_counter()
    with open_index(settings.index_path) as index:
        app.state.index = index
        app.state.started = time.monotonic()
        log.info(
            "index %s loaded in %.0f ms: %r",
            settings.index_path,
            (time.perf_counter() - started) * 1000,
            index,
        )
        app.state.semantic = _load_semantic(settings, index)
        yield
        app.state.index = None
        app.state.semantic = None
    log.info("index closed")


def _load_semantic(settings: Settings, index: Index) -> Semantic | None:
    """Passages and the model, when the embeddings exist and the optional
    ``semantic`` extra is installed; otherwise the semantic modes are off."""
    prefix = settings.embeddings_path or default_prefix(settings.index_path)
    if not Path(f"{prefix}.npy").is_file():
        log.info("no embeddings at %s: keyword search only", prefix)
        return None
    started = time.perf_counter()
    try:
        passages = Passages.load(prefix, index)
        from findex.semantic import FastEmbedder

        semantic = Semantic(passages, FastEmbedder(passages.model))
    except (ImportError, ValueError) as exc:
        log.warning("semantic search off: %s", exc)
        return None
    log.info(
        "%d passages and %s loaded in %.0f ms",
        passages.doc_ids.size,
        passages.model,
        (time.perf_counter() - started) * 1000,
    )
    return semantic


def get_index(request: Request) -> Index:
    index: Index | None = getattr(request.app.state, "index", None)
    if index is None:
        raise HTTPException(503, "index not loaded")
    return index


def get_semantic(request: Request) -> Semantic | None:
    semantic: Semantic | None = getattr(request.app.state, "semantic", None)
    return semantic


IndexDep = Annotated[Index, Depends(get_index)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
SemanticDep = Annotated[Semantic | None, Depends(get_semantic)]


def create_app() -> FastAPI:
    app = FastAPI(
        title="findex",
        summary="Full-text search over a small corpus: BM25, Boolean queries, phrases.",
        version=version("findex"),
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    app.middleware("http")(_request_id_and_errors)
    _api_routes(app)
    _page_routes(app)
    return app


# --- every request: an id, an access log line, no tracebacks to the client -------


async def _request_id_and_errors(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        # The full traceback goes to the log with the id; the client gets the
        # id only, so a bug report can be matched to the log line.
        log.exception(
            "unhandled error [%s] %s %s", request_id, request.method, request.url.path
        )
        response = JSONResponse(
            {"detail": "internal server error", "request_id": request_id},
            status_code=500,
        )
    response.headers["x-request-id"] = request_id
    log.info(
        "[%s] %s %s -> %d in %.1f ms",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started) * 1000,
    )
    return response


# --- JSON API --------------------------------------------------------------------

Q = Annotated[
    str, Query(min_length=1, max_length=200, description="findex query syntax")
]
K = Annotated[int, Query(ge=1, le=50, description="Results per page.")]
PageNo = Annotated[int, Query(ge=1, le=100, description="1-based page number.")]


def _found(
    index: Index,
    q: str,
    scorer: ScorerName,
    k: int,
    page: int,
    mode: Mode,
    semantic: Semantic | None,
) -> Page:
    if mode != "keyword" and semantic is None:
        raise HTTPException(503, f"{mode} search is not available on this server")
    try:
        return search_page(index, q, scorer, k, page, mode, semantic)
    except QueryError as exc:
        raise HTTPException(422, f"bad query: {exc}") from exc


def _api_routes(app: FastAPI) -> None:
    @app.get("/search", response_model=models.SearchResponse, tags=["api"])
    def search_api(  # def, not async def: scoring is CPU work (see README, Lab 7)
        index: IndexDep,
        semantic: SemanticDep,
        q: Q,
        k: K = 10,
        mode: Mode = "keyword",
        scorer: ScorerName = "bm25",
        page: PageNo = 1,
    ) -> models.SearchResponse:
        """Ranked results for ``q``, ``k`` per page.

        ``mode``: ``keyword`` (BM25 over the Boolean query), ``semantic``
        (embeddings, best passage per document) or ``hybrid`` (both, fused
        with Reciprocal Rank Fusion)."""
        found = _found(index, q, scorer, k, page, mode, semantic)
        return models.SearchResponse(
            query=found.query,
            parsed=found.parsed,
            mode=found.mode,
            scorer=found.scorer,
            page=found.page,
            k=found.k,
            total=found.total,
            pages=found.pages,
            took_ms=round(found.took_ms, 2),
            results=[
                models.Hit(
                    rank=hit.rank,
                    doc_id=hit.result.doc_id,
                    title=hit.result.title,
                    score=round(hit.result.score, 4),
                    source=hit.source,
                    url=hit.result.url,
                    snippet=(plain := snippet_text(hit.parts))[0],
                    highlights=plain[1],
                )
                for hit in found.hits
            ],
        )

    @app.get(
        "/docs/{doc_id}",
        response_model=models.Document,
        responses={404: {"model": models.Error}},
        tags=["api"],
    )
    def document_api(
        index: IndexDep,
        doc_id: int,
        offset: Annotated[int, Query(ge=0)] = 0,
        length: Annotated[int, Query(ge=1, le=200_000)] = 20_000,
    ) -> models.Document:
        """A document's metadata and a slice of its text."""
        meta = index.doc_meta.get(doc_id)
        if meta is None:
            raise HTTPException(404, f"no document {doc_id}")
        part = document_slice(meta.path, meta.url, offset, length)
        return models.Document(
            doc_id=doc_id,
            title=meta.title,
            source=meta.url or Path(meta.path).name,
            url=meta.url,
            tokens=index.doc_length(doc_id),
            chars=part.chars,
            offset=part.offset,
            text=part.text,
            next_offset=part.next_offset,
        )

    @app.get("/stats", response_model=models.Stats, tags=["api"])
    def stats_api(
        request: Request, index: IndexDep, settings: SettingsDep
    ) -> models.Stats:
        """Size of the loaded index."""
        return models.Stats(
            version=version("findex"),
            documents=index.num_docs,
            terms=len(index),
            tokens=sum(index.doc_lengths.values()),
            avg_doc_length=round(index.avg_doc_length, 1),
            positions=index.positions,
            index_path=str(settings.index_path),
            index_bytes=settings.index_path.stat().st_size,
            uptime_s=round(time.monotonic() - request.app.state.started, 1),
        )

    @app.get(
        "/health",
        response_model=models.Health,
        responses={503: {"model": models.Error}},
        tags=["api"],
    )
    def health(index: IndexDep) -> models.Health:
        """200 once the index is loaded, 503 before: what the platform probes."""
        return models.Health(status="ok", documents=index.num_docs)


# --- HTML pages --------------------------------------------------------------------


def _page_routes(app: FastAPI) -> None:
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(
        request: Request,
        index: IndexDep,
        settings: SettingsDep,
        semantic: SemanticDep,
        q: Annotated[str, Query(max_length=200)] = "",
        mode: Mode = "keyword",
        scorer: ScorerName = "bm25",
        page: Annotated[int, Query(ge=1, le=100)] = 1,
    ) -> HTMLResponse:
        found: Page | None = None
        error: str | None = None
        if mode != "keyword" and semantic is None:
            mode = "keyword"
        if q.strip():
            try:
                found = search_page(index, q, scorer, 10, page, mode, semantic)
            except QueryError as exc:
                error = str(exc)
        return templates.TemplateResponse(
            request,
            "search.html",
            {
                "q": q,
                "mode": mode,
                "modes": ("keyword", "semantic", "hybrid")
                if semantic
                else ("keyword",),
                "scorer": scorer,
                "found": found,
                "error": error,
                "site": settings.site_name,
                "documents": index.num_docs,
            },
            status_code=422 if error else 200,
        )

    @app.get("/doc/{doc_id}", response_class=HTMLResponse, include_in_schema=False)
    def document_page(
        request: Request,
        index: IndexDep,
        settings: SettingsDep,
        doc_id: int,
        q: Annotated[str, Query(max_length=200)] = "",
        offset: Annotated[int | None, Query(ge=0)] = None,
    ) -> HTMLResponse:
        meta = index.doc_meta.get(doc_id)
        if meta is None:
            raise HTTPException(404, f"no document {doc_id}")
        length = 30_000
        terms = _terms(q)
        if offset is None:  # open the document where the query first matches
            offset = _first_match(meta.path, meta.url, terms)
        part = document_slice(meta.path, meta.url, offset, length)
        parts: tuple[Part, ...] = ()
        if part.text is not None:
            parts = _highlight(part.text, terms)
        return templates.TemplateResponse(
            request,
            "document.html",
            {
                "meta": meta,
                "doc_id": doc_id,
                "source": meta.url or Path(meta.path).name,
                "tokens": index.doc_length(doc_id),
                "part": part,
                "parts": parts,
                "q": q,
                "length": length,
                "site": settings.site_name,
            },
        )


def _terms(q: str) -> frozenset[str]:
    try:
        return frozenset(parse(q).terms()) if q.strip() else frozenset()
    except QueryError:
        return frozenset()


def _first_match(path: str, url: str | None, terms: frozenset[str]) -> int:
    """An offset a little before the first match, so the page opens there."""
    text = document_text(path, url) if terms else None
    start = first_hit(unicodedata.normalize("NFC", text), terms) if text else None
    return 0 if start is None else max(0, start - 300)


def _highlight(text: str, terms: frozenset[str]) -> tuple[Part, ...]:
    return tuple(highlight_parts(text, terms))


app = create_app()
