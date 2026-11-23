"""The ``findex`` command: ``findex index``, ``findex search``, ``findex stats``.

Results go to stdout; logs, progress bars and errors go to stderr, so
``findex search index.bin "q" --json > hits.jsonl`` captures only the data.
Logging is configured here, once, and nowhere in the library.
"""

import contextlib
import io
import json
import logging
import os
import queue
import sys
import time
import tracemalloc
from collections.abc import Generator, Iterable, Iterator
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import asdict
from importlib.metadata import version
from itertools import islice
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal, NoReturn, TypedDict

import typer
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from findex.boolean import EngineName
from findex.corpus import (
    Document,
    corpus_paths,
    is_jsonl,
    iter_documents,
)
from findex.index import build_index
from findex.model import Index
from findex.parallel import ExecutorName, build_parallel, gil_enabled
from findex.query import Node, QueryError, parse
from findex.search import SCORERS, ScorerName, search
from findex.semantic import (
    Found,
    Mode,
    Passages,
    Semantic,
    default_prefix,
    find,
    found_snippet,
    query_terms,
)
from findex.snippets import Part
from findex.stats import Stats, eager_stats, index_stats, lazy_stats
from findex.store import Format, IndexFormatError, open_index, save
from findex.sysinfo import peak_rss

if TYPE_CHECKING:
    from findex.crawler import CrawlStats
    from findex.semantic import Embedder

# The crawler (asyncio, httpx) is imported inside `findex crawl` only: profiling
# showed it was a third of the start-up time of every other command.

log = logging.getLogger("findex.cli")

app = typer.Typer(
    name="findex",
    help="A small full-text search engine: build an index, then search it.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

# Both look up sys.stdout / sys.stderr when they write, not when created.
out = Console()
err = Console(stderr=True)

LEVELS: Final = (logging.WARNING, logging.INFO, logging.DEBUG)
# The benchmark (README, Lab 5): with the GIL, processes are fastest; on a
# free-threaded build threads are, with no pickling and no process start-up.
DEFAULT_EXECUTOR: Final[ExecutorName] = "processes" if gil_enabled() else "threads"
_handler: logging.Handler | None = None


class ResultRecord(TypedDict):
    """One line of ``findex search --json``."""

    query: str
    rank: int
    doc_id: int
    score: float
    title: str
    path: str
    url: str | None
    snippet: str | None


# --- plumbing ------------------------------------------------------------------


def configure_logging(verbosity: int) -> None:
    """-v: INFO, -vv: DEBUG, otherwise warnings only. Always to stderr."""
    global _handler
    logger = logging.getLogger("findex")
    if _handler is not None:  # called again (tests run the app many times)
        logger.removeHandler(_handler)
    _handler = RichHandler(console=err, show_time=False, show_path=False)
    _handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    level = LEVELS[min(verbosity, len(LEVELS) - 1)]
    # The handler's own level matters too: findex.crawler logs at INFO into
    # crawl.log even without -v, and those records also reach this handler.
    _handler.setLevel(level)
    logger.addHandler(_handler)
    logger.setLevel(level)


def _never_crash_on_encoding() -> None:
    # A Windows console or a redirected stdout may use a legacy code page; a
    # character it lacks should become "?", not a UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors="replace")


def fail(message: str, code: int = 1) -> NoReturn:
    """An expected problem: one line on stderr, a nonzero exit, no traceback."""
    err.print(f"[bold red]error:[/] {escape(message)}", soft_wrap=True)
    raise typer.Exit(code)


def _version(show: bool) -> None:
    if show:
        out.print(f"findex {version('findex')}")
        raise typer.Exit


@app.callback()
def main(
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="-v: progress info, -vv: timings and cache.",
        ),
    ] = 0,
    show_version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version, is_eager=True, help="Show the version."
        ),
    ] = False,
) -> None:
    """A small full-text search engine: build an index, then search it."""
    _never_crash_on_encoding()
    configure_logging(verbose)


# --- findex index ----------------------------------------------------------------


@app.command("index")
def index_command(
    corpus: Annotated[
        Path,
        typer.Argument(
            help="Directory with .txt files (searched recursively), "
            "or a .jsonl file written by findex crawl."
        ),
    ],
    out_path: Annotated[
        Path, typer.Option("--out", "-o", help="Where to write the index.")
    ] = Path("index.bin"),
    positions: Annotated[
        bool,
        typer.Option(
            "--positions/--no-positions", help="Store token positions (phrase queries)."
        ),
    ] = True,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Only the first N documents.")
    ] = None,
    executor: Annotated[
        ExecutorName,
        typer.Option(
            help="Run the chunks in a loop, a thread pool or a process pool. "
            "Default: processes, or threads on a free-threaded (no-GIL) Python."
        ),
    ] = DEFAULT_EXECUTOR,
    workers: Annotated[
        int | None,
        typer.Option(min=1, help="Threads or processes. Default: os.cpu_count()."),
    ] = None,
    fmt: Annotated[
        Format | None,
        typer.Option("--format", help="Default: pickle for .pkl, binary otherwise."),
    ] = None,
) -> None:
    """Build an index from a corpus directory or a crawl's .jsonl file and save it."""
    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=err,
    )
    if is_jsonl(corpus):
        # One file of crawled pages: the chunk-by-path parallel build does not
        # apply, and a crawl is small next to the book corpus. Serial it is.
        with corpus.open("rb") as lines:
            total = sum(1 for _ in lines)
        total = total if limit is None else min(total, limit)
        log.info("indexing %d pages from %s", total, corpus)
        with progress:
            task = progress.add_task("indexing", total=total)
            started = time.perf_counter()
            docs = islice(iter_documents(corpus), limit)
            index = build_index(_advancing(docs, progress, task), "array", positions)
            build_line = f"{time.perf_counter() - started:.2f} s (serial, JSON lines)"
        merge_line = None
    elif corpus.is_dir():
        paths = corpus_paths(corpus)[:limit]
        log.info("indexing %d documents from %s", len(paths), corpus)
        with progress:
            task = progress.add_task("indexing", total=len(paths))
            report = build_parallel(
                paths,
                executor,
                workers,
                positions,
                on_chunk=lambda chunk: progress.advance(task, len(chunk.paths)),
            )
        index = report.index
        build_line = (
            f"{report.wall:.2f} s ({report.executor}, {report.workers} workers)"
        )
        merge_line = f"{report.merge:.2f} s"
    else:
        fail(f"no corpus directory or .jsonl file at {corpus}")

    started = time.perf_counter()
    try:
        used = save(index, out_path, fmt)
    except OSError as exc:
        fail(f"cannot write {out_path}: {exc.strerror or exc}")
    saved = time.perf_counter() - started
    log.info("saved %s (%s) in %.2f s", out_path, used, saved)

    table = Table(title=f"Index written to {out_path}", show_header=False)
    table.add_column(style="bold")
    table.add_column(justify="right")
    table.add_row("documents", f"{index.num_docs:,}")
    table.add_row("terms", f"{len(index):,}")
    table.add_row("postings", f"{index.posting_count():,}")
    table.add_row("positions", "yes" if index.positions else "no")
    table.add_row("format", used)
    table.add_row("file size", f"{out_path.stat().st_size / 2**20:.1f} MiB")
    table.add_row("build", build_line)
    if merge_line is not None:
        table.add_row("of which merge", merge_line)
    table.add_row("save", f"{saved:.2f} s")
    out.print(table)


def _advancing(
    docs: Iterable[Document], progress: Progress, task: TaskID
) -> Iterator[Document]:
    for doc in docs:
        yield doc
        progress.advance(task)


# --- findex embed ------------------------------------------------------------------


@app.command("embed")
def embed_command(
    index_path: Annotated[Path, typer.Argument(metavar="INDEX", help="An index file.")],
    out_prefix: Annotated[
        Path | None,
        typer.Option(
            "--out", help="Default: the index path with .emb instead of its suffix."
        ),
    ] = None,
    words: Annotated[int, typer.Option(min=20, help="Words per passage.")] = 200,
    model: Annotated[
        str, typer.Option(help="A fastembed model name.")
    ] = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    """Cut documents into passages and embed them, for --mode semantic / hybrid."""
    from findex.semantic import default_prefix, embed_index

    prefix = out_prefix or default_prefix(index_path)
    embedder = _embedder(model)
    if not index_path.is_file():
        fail(f"index file not found: {index_path}")
    started = time.perf_counter()
    with (
        open_index(index_path) as index,
        Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=err,
        ) as progress,
    ):
        task = progress.add_task("embedding", total=None)
        count = embed_index(
            index,
            embedder,
            prefix,
            words,
            on_start=lambda total: progress.update(task, total=total),
            on_batch=lambda n: progress.advance(task, n),
        )
    took = time.perf_counter() - started
    out.print(
        f"{count:,} passages embedded with {embedder.name} in {took:.0f} s -> {prefix}.npy"
    )


def _embedder(model: str) -> Embedder:
    try:
        from findex.semantic import FastEmbedder

        return FastEmbedder(model)
    except ImportError:
        fail("semantic search needs the optional extra: pip install 'findex[semantic]'")


# --- findex search ---------------------------------------------------------------


def _parse_all(queries: list[str]) -> list[Node]:
    nodes: list[Node] = []
    for query in queries:
        try:
            nodes.append(parse(query))
        except QueryError as exc:
            fail(f"bad query {query!r}: {exc}", code=2)
    return nodes


def _as_text(parts: list[Part] | None) -> Text:
    if parts is None:
        return Text("—", style="dim")
    return Text.assemble(
        *((piece, "bold yellow") if hit else piece for piece, hit in parts)
    )


def _results_table(
    index: Index,
    query: str,
    caption: str,
    found: list[Found],
    terms: frozenset[str],
    snippets: bool,
    semantic: Semantic | None,
) -> Table:
    table = Table(
        title=f"[bold]{escape(query)}[/]", caption=escape(caption), expand=True
    )
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("score", justify="right", no_wrap=True)
    table.add_column("document", ratio=1)
    if snippets:
        table.add_column("snippet", ratio=3)
    for rank, hit in enumerate(found, 1):
        result = hit.result
        where = result.url or Path(result.path).name
        title = Text.assemble(result.title, "\n", (where, "dim"))
        row: list[str | Text] = [str(rank), f"{result.score:.3f}", title]
        if snippets:
            row.append(_as_text(_snippet(index, hit, terms, semantic)))
        table.add_row(*row)
    return table


def _snippet(
    index: Index, hit: Found, terms: frozenset[str], semantic: Semantic | None
) -> list[Part] | None:
    parts = found_snippet(index, hit, terms, semantic)
    if parts == [("(text not available)", False)]:
        result = hit.result
        log.warning("no snippet for %s: file not found", result.url or result.path)
        return None
    return parts


def _json_records(
    index: Index,
    query: str,
    found: list[Found],
    terms: frozenset[str],
    snippets: bool,
    semantic: Semantic | None,
) -> Iterator[ResultRecord]:
    for rank, hit in enumerate(found, 1):
        result = hit.result
        parts = _snippet(index, hit, terms, semantic) if snippets else None
        yield {
            "query": query,
            "rank": rank,
            "doc_id": result.doc_id,
            "score": result.score,
            "title": result.title,
            "path": result.path,
            "url": result.url,
            "snippet": None if parts is None else "".join(piece for piece, _ in parts),
        }


@app.command("search")
def search_command(
    index_path: Annotated[Path, typer.Argument(metavar="INDEX", help="An index file.")],
    queries: Annotated[
        list[str],
        typer.Argument(
            metavar="QUERY...",
            help="e.g. 'sea AND (pirate OR treasure) NOT whale \"dead man\"'",
        ),
    ],
    k: Annotated[int, typer.Option("--k", "-k", min=1, help="How many results.")] = 10,
    mode: Annotated[
        Mode,
        typer.Option(
            help="keyword: BM25 over the query; semantic: embeddings; hybrid: both, RRF."
        ),
    ] = "keyword",
    scorer: Annotated[ScorerName, typer.Option(help="Ranking function.")] = "bm25",
    engine: Annotated[
        EngineName, typer.Option(help="How Boolean operators are computed.")
    ] = "merge",
    embeddings: Annotated[
        Path | None,
        typer.Option(help="Embeddings prefix (default: the index path with .emb)."),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="JSON lines on stdout instead of a table.")
    ] = False,
    snippets: Annotated[
        bool, typer.Option("--snippets/--no-snippets", help="Show text around matches.")
    ] = True,
) -> None:
    """Search an index; several queries can be given at once."""
    if mode == "keyword":
        _parse_all(queries)  # bad syntax fails before the index is opened
    if not index_path.is_file():
        fail(f"index file not found: {index_path}")
    try:
        with open_index(index_path) as index:
            log.info("opened %r", index)
            semantic = None
            if mode != "keyword":
                semantic = _load_semantic(
                    index, embeddings or default_prefix(index_path)
                )
            for query in queries:
                _run_query(
                    index, query, mode, scorer, k, engine, as_json, snippets, semantic
                )
    except IndexFormatError as exc:
        fail(str(exc))


def _load_semantic(index: Index, prefix: Path) -> Semantic:
    if not Path(f"{prefix}.npy").is_file():
        fail(f"no embeddings at {prefix}.npy: run `findex embed` first")
    try:
        passages = Passages.load(prefix, index)
    except ValueError as exc:
        fail(f"embeddings at {prefix} do not fit this index: {exc}")
    return Semantic(passages, _embedder(passages.model))


def _run_query(
    index: Index,
    query: str,
    mode: Mode,
    scorer: ScorerName,
    k: int,
    engine: EngineName,
    as_json: bool,
    snippets: bool,
    semantic: Semantic | None,
) -> None:
    try:
        if mode == "keyword":
            node = parse(query)
            results = search(index, node, SCORERS[scorer], k, engine)
            found = [Found(r, None) for r in results]
            caption = str(node)
        else:
            found, _ = find(index, query, mode, k, SCORERS[scorer], semantic)
            caption = f"{mode} search"
    except QueryError as exc:  # e.g. a phrase on an index without positions
        fail(f"cannot run {query!r}: {exc}", code=2)
    terms = query_terms(query)
    if as_json:
        for record in _json_records(index, query, found, terms, snippets, semantic):
            sys.stdout.write(json.dumps(record) + "\n")
        sys.stdout.flush()
    elif found:
        out.print(
            _results_table(index, query, caption, found, terms, snippets, semantic)
        )
    else:
        out.print(f"[bold]{escape(query)}[/]: nothing found")


# --- findex crawl ----------------------------------------------------------------


@contextlib.contextmanager
def _crawl_log(path: Path) -> Generator[None]:
    """Send findex.crawler (one line per URL, retries) and asyncio's own warnings
    to ``path``. The event loop only puts records on a queue; a listener thread
    does the file writes, so logging never blocks the loop."""
    records: queue.SimpleQueue[logging.LogRecord] = queue.SimpleQueue()
    file = logging.FileHandler(path, mode="w", encoding="utf-8")
    file.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    listener = QueueListener(records, file)
    to_queue = QueueHandler(records)
    loggers = [logging.getLogger("findex.crawler"), logging.getLogger("asyncio")]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.addHandler(to_queue)
    loggers[0].setLevel(logging.INFO)
    listener.start()
    try:
        yield
    finally:
        for lg, level in zip(loggers, levels, strict=True):
            lg.removeHandler(to_queue)
            lg.setLevel(level)
        listener.stop()
        file.close()


def _live_stats(stats: CrawlStats, max_pages: int, started: float) -> Table:
    elapsed = max(time.perf_counter() - started, 1e-9)
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column(justify="right")
    table.add_row("pages", f"{stats.pages} / {max_pages}")
    table.add_row("pages/s", f"{stats.pages / elapsed:.1f}")
    table.add_row("in flight", str(stats.in_flight))
    table.add_row("errors", str(stats.errors))
    table.add_row("skipped", str(stats.skipped))
    table.add_row("queue", str(stats.queued))
    table.add_row("elapsed", f"{elapsed:.1f} s")
    return table


def _sleep_briefly(_: int) -> None:
    time.sleep(0.2)


@contextlib.contextmanager
def _parse_pool(kind: Literal["threads", "processes"]) -> Generator[Executor | None]:
    """None means the event loop's own thread pool."""
    if kind == "threads":
        yield None
        return
    workers = min(2, os.cpu_count() or 1)
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        # Start every worker now, before the event loop runs: the pool would
        # otherwise spawn them on first use, from inside the loop, and starting
        # a process blocks for a good part of a second on Windows.
        list(pool.map(_sleep_briefly, range(workers)))
        yield pool


async def _crawl_to_file(
    seed: str, out_path: Path, stats: CrawlStats, **options: Any
) -> None:
    from findex.crawler import crawl

    with out_path.open("w", encoding="utf-8") as sink:
        async with contextlib.aclosing(crawl([seed], stats=stats, **options)) as pages:
            async for page in pages:
                # Written as each page arrives; nothing piles up in memory.
                sink.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")
                sink.flush()


@app.command("crawl")
def crawl_command(
    seed: Annotated[str, typer.Argument(help="Where to start, an http(s) URL.")],
    max_pages: Annotated[
        int, typer.Option(min=1, help="Stop after this many HTML pages.")
    ] = 500,
    concurrency: Annotated[
        int, typer.Option(min=1, help="Requests in flight at most, all hosts together.")
    ] = 10,
    per_host: Annotated[
        int, typer.Option(min=1, help="Requests in flight at most to one host.")
    ] = 2,
    delay: Annotated[
        float,
        typer.Option(min=0.0, help="Seconds between request starts to one host."),
    ] = 0.25,
    out_path: Annotated[
        Path, typer.Option("--out", "-o", help="JSON lines, one page per line.")
    ] = Path("data/crawl.jsonl"),
    log_path: Annotated[Path, typer.Option("--log", help="Per-URL status log.")] = Path(
        "crawl.log"
    ),
    whole_host: Annotated[
        bool,
        typer.Option(
            help="Follow links anywhere on the seed's host, not only under it."
        ),
    ] = False,
    timeout: Annotated[
        float, typer.Option(min=0.1, help="Seconds per request attempt.")
    ] = 10.0,
    debug: Annotated[
        bool,
        typer.Option(
            help="asyncio debug mode: log anything blocking the loop > 100 ms."
        ),
    ] = False,
    parse_in: Annotated[
        Literal["threads", "processes"],
        typer.Option(help="Where pages are parsed: never on the event loop."),
    ] = "processes",
) -> None:
    """Crawl a site politely (robots.txt, per-host limits) into a .jsonl corpus."""
    import asyncio

    from findex.crawler import USER_AGENT, CrawlStats, canonical

    if canonical(seed) is None:
        fail(f"not an http(s) URL: {seed}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = CrawlStats()
    started = time.perf_counter()
    options: dict[str, Any] = {
        "max_pages": max_pages,
        "concurrency": concurrency,
        "per_host": per_host,
        "delay": delay,
        "stay_under_seed": not whole_host,
        "timeout": timeout,
    }
    log.info("crawling %s as %r", seed, USER_AGENT)
    try:
        with (
            _crawl_log(log_path),
            Live(
                get_renderable=lambda: _live_stats(stats, max_pages, started),
                console=err,
                refresh_per_second=4,
            ),
            _parse_pool(parse_in) as pool,
        ):
            options["parse_pool"] = pool
            asyncio.run(_crawl_to_file(seed, out_path, stats, **options), debug=debug)
    except KeyboardInterrupt:
        fail(f"interrupted after {stats.pages} pages (kept in {out_path})", code=130)
    wall = time.perf_counter() - started

    table = Table(title=f"Crawl of {escape(seed)}", show_header=False)
    table.add_column(style="bold")
    table.add_column(justify="right")
    table.add_row("pages", f"{stats.pages:,}")
    table.add_row("requests", f"{stats.requests:,}")
    table.add_row("errors", f"{stats.errors:,}")
    table.add_row("skipped", f"{stats.skipped:,}")
    statuses = ", ".join(f"{k}: {v}" for k, v in sorted(stats.statuses.items()))
    table.add_row("statuses", statuses or "-")
    table.add_row("wall", f"{wall:.2f} s")
    table.add_row("pages/s", f"{stats.pages / wall:.2f}")
    table.add_row("peak RSS", f"{peak_rss() / 2**20:.0f} MiB")
    table.add_row("written to", escape(str(out_path)))
    table.add_row("log", escape(str(log_path)))
    out.print(table)


# --- findex serve -----------------------------------------------------------------


def uvicorn_log_config(level: str) -> dict[str, Any]:
    """One line per record on stdout, where container platforms collect logs."""
    line = {"format": "%(asctime)s %(levelname)-7s %(name)s: %(message)s"}
    stdout = {
        "class": "logging.StreamHandler",
        "formatter": "line",
        "stream": "ext://sys.stdout",
    }
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"line": line},
        "handlers": {"stdout": stdout},
        "loggers": {
            "uvicorn": {"handlers": ["stdout"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": [], "propagate": False},
            "findex": {"handlers": ["stdout"], "level": level, "propagate": False},
        },
    }


@app.command("serve")
def serve_command(
    index_path: Annotated[
        Path | None,
        typer.Option("--index", envvar="INDEX_PATH", help="Index file to serve."),
    ] = None,
    host: Annotated[
        str, typer.Option(envvar="HOST", help="Address to bind.")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535, envvar="PORT")] = 8000,
    workers: Annotated[
        int,
        typer.Option(min=1, envvar="WEB_CONCURRENCY", help="Server processes."),
    ] = 1,
) -> None:
    """Serve the web page and the JSON API (uvicorn + FastAPI)."""
    import uvicorn
    from pydantic import ValidationError

    from findex.web.settings import Settings

    if index_path is not None:
        # Through the environment, so every uvicorn worker process sees it too.
        os.environ["INDEX_PATH"] = str(index_path)
    try:
        settings = Settings()
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"]).upper()
        fail(f"bad configuration: {where}: {first['msg']}")
    log.info(
        "serving %s on http://%s:%d with %d worker(s)",
        settings.index_path,
        host,
        port,
        workers,
    )
    uvicorn.run(
        "findex.web.app:app",
        host=host,
        port=port,
        workers=workers,
        log_config=uvicorn_log_config(settings.log_level),
        access_log=False,  # the app logs each request itself, with its id
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


# --- findex stats ----------------------------------------------------------------


@app.command("stats")
def stats_command(
    path: Annotated[
        Path, typer.Argument(help="An index file, or a corpus directory to stream.")
    ],
    top: Annotated[int, typer.Option(min=1, help="How many top terms.")] = 50,
    eager: Annotated[
        bool, typer.Option(help="Corpus only: the list-based version from Lab 1.")
    ] = False,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Corpus only: the first N documents.")
    ] = None,
) -> None:
    """Documents, tokens, vocabulary and the most frequent terms."""
    tracemalloc.start()
    started = time.perf_counter()
    if path.is_dir() or is_jsonl(path):
        source = f"corpus {path} ({'eager' if eager else 'lazy'})"
        stats = (eager_stats if eager else lazy_stats)(path, limit, top)
    elif path.is_file():
        source = f"index {path}"
        try:
            with open_index(path) as index:
                stats = index_stats(index, top)
        except IndexFormatError as exc:
            fail(str(exc))
    else:
        fail(f"no such index file or corpus directory: {path}")
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    out.print(_stats_table(source, stats, elapsed, peak))


def _stats_table(source: str, stats: Stats, elapsed: float, peak: int) -> Table:
    summary = Table(title=escape(source), show_header=False)
    summary.add_column(style="bold")
    summary.add_column(justify="right")
    summary.add_row("documents", f"{stats.documents:,}")
    summary.add_row("tokens", f"{stats.tokens:,}")
    summary.add_row("vocabulary", f"{stats.vocabulary:,}")
    summary.add_row("elapsed", f"{elapsed:.2f} s")
    summary.add_row("peak memory", f"{peak / 2**20:.1f} MiB")
    top = Table(title=f"top {len(stats.top)} terms")
    top.add_column("#", justify="right", style="dim")
    top.add_column("term")
    top.add_column("count", justify="right")
    for rank, (term, count) in enumerate(stats.top, 1):
        top.add_row(str(rank), escape(term), f"{count:,}")
    layout = Table.grid(padding=(0, 4))
    layout.add_row(summary, top)
    return layout
