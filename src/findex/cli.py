"""The ``findex`` command: ``findex index``, ``findex search``, ``findex stats``.

Results go to stdout; logs, progress bars and errors go to stderr, so
``findex search index.bin "q" --json > hits.jsonl`` captures only the data.
Logging is configured here, once, and nowhere in the library.
"""

import io
import json
import logging
import sys
import time
import tracemalloc
from collections.abc import Iterator
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Final, NoReturn, TypedDict

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from findex.boolean import EngineName
from findex.corpus import corpus_paths
from findex.model import Index
from findex.parallel import ExecutorName, build_parallel, gil_enabled
from findex.query import Node, QueryError, parse
from findex.search import SCORERS, ScorerName, SearchResult, search
from findex.snippets import Part, snippet_parts
from findex.stats import Stats, eager_stats, index_stats, lazy_stats
from findex.store import Format, IndexFormatError, open_index, save

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
    logger.addHandler(_handler)
    logger.setLevel(LEVELS[min(verbosity, len(LEVELS) - 1)])


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
        Path, typer.Argument(help="Directory with .txt files (searched recursively).")
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
    """Build an index from a corpus directory and save it."""
    if not corpus.is_dir():
        fail(f"corpus directory not found: {corpus}")
    paths = corpus_paths(corpus)[:limit]
    log.info("indexing %d documents from %s", len(paths), corpus)

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=err,
    ) as progress:
        task = progress.add_task("indexing", total=len(paths))
        report = build_parallel(
            paths,
            executor,
            workers,
            positions,
            on_chunk=lambda chunk: progress.advance(task, len(chunk.paths)),
        )
    index = report.index

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
    table.add_row(
        "build", f"{report.wall:.2f} s ({report.executor}, {report.workers} workers)"
    )
    table.add_row("of which merge", f"{report.merge:.2f} s")
    table.add_row("save", f"{saved:.2f} s")
    out.print(table)


# --- findex search ---------------------------------------------------------------


def _parse_all(queries: list[str]) -> list[Node]:
    nodes: list[Node] = []
    for query in queries:
        try:
            nodes.append(parse(query))
        except QueryError as exc:
            fail(f"bad query {query!r}: {exc}", code=2)
    return nodes


def _snippet_parts(result: SearchResult, terms: set[str]) -> list[Part] | None:
    try:
        text = Path(result.path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        log.warning("no snippet for %s: file not found", result.path)
        return None
    return snippet_parts(text, terms)


def _as_text(parts: list[Part] | None) -> Text:
    if parts is None:
        return Text("—", style="dim")
    return Text.assemble(
        *((piece, "bold yellow") if hit else piece for piece, hit in parts)
    )


def _results_table(
    query: str, node: Node, results: list[SearchResult], snippets: bool
) -> Table:
    table = Table(
        title=f"[bold]{escape(query)}[/]", caption=escape(str(node)), expand=True
    )
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("score", justify="right", no_wrap=True)
    table.add_column("document", ratio=1)
    if snippets:
        table.add_column("snippet", ratio=3)
    terms = set(node.terms())
    for rank, result in enumerate(results, 1):
        title = Text.assemble(result.title, "\n", (Path(result.path).name, "dim"))
        row: list[str | Text] = [str(rank), f"{result.score:.3f}", title]
        if snippets:
            row.append(_as_text(_snippet_parts(result, terms)))
        table.add_row(*row)
    return table


def _json_records(
    query: str, node: Node, results: list[SearchResult], snippets: bool
) -> Iterator[ResultRecord]:
    terms = set(node.terms())
    for rank, result in enumerate(results, 1):
        parts = _snippet_parts(result, terms) if snippets else None
        yield {
            "query": query,
            "rank": rank,
            "doc_id": result.doc_id,
            "score": result.score,
            "title": result.title,
            "path": result.path,
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
    scorer: Annotated[ScorerName, typer.Option(help="Ranking function.")] = "bm25",
    engine: Annotated[
        EngineName, typer.Option(help="How Boolean operators are computed.")
    ] = "merge",
    as_json: Annotated[
        bool, typer.Option("--json", help="JSON lines on stdout instead of a table.")
    ] = False,
    snippets: Annotated[
        bool, typer.Option("--snippets/--no-snippets", help="Show text around matches.")
    ] = True,
) -> None:
    """Search an index; several queries can be given at once."""
    nodes = _parse_all(queries)  # bad syntax fails before the index is opened
    if not index_path.is_file():
        fail(f"index file not found: {index_path}")
    try:
        with open_index(index_path) as index:
            log.info("opened %r", index)
            for query, node in zip(queries, nodes, strict=True):
                _run_query(index, query, node, scorer, k, engine, as_json, snippets)
    except IndexFormatError as exc:
        fail(str(exc))


def _run_query(
    index: Index,
    query: str,
    node: Node,
    scorer: ScorerName,
    k: int,
    engine: EngineName,
    as_json: bool,
    snippets: bool,
) -> None:
    try:
        results = search(index, node, SCORERS[scorer], k, engine)
    except QueryError as exc:  # e.g. a phrase on an index without positions
        fail(f"cannot run {query!r}: {exc}", code=2)
    if as_json:
        for record in _json_records(query, node, results, snippets):
            sys.stdout.write(json.dumps(record) + "\n")
        sys.stdout.flush()
    elif results:
        out.print(_results_table(query, node, results, snippets))
    else:
        out.print(f"[bold]{escape(query)}[/]: nothing found")


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
    if path.is_dir():
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
