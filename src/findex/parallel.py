"""Parallel index construction as map-reduce: ``build_partial`` per chunk, then ``merge``.

Map: each chunk is a list of *paths* plus the doc_id of its first file, so a
worker reads its own files (nothing big crosses a process boundary on the way
in) and its doc_ids never collide with another chunk's. A partial index comes
back as compact uint32 arrays, cheap to pickle.

Reduce: chunks cover consecutive doc_id ranges, so merging is concatenation in
chunk order; every postings list stays sorted. The merge runs in the parent,
one chunk after another: by Amdahl's law it caps the speedup.

The same two functions run under a plain loop, a ``ThreadPoolExecutor`` or a
``ProcessPoolExecutor``; nothing else differs between the three.
"""

import logging
import math
import multiprocessing
import os
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import (
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from findex.corpus import read_documents
from findex.index import build_index
from findex.model import ArrayPostings, DocMeta, Index, to_arrays
from findex.timing import timed

log = logging.getLogger(__name__)

type ExecutorName = Literal["serial", "threads", "processes"]
EXECUTOR_NAMES: Final[tuple[ExecutorName, ...]] = ("serial", "threads", "processes")
TASKS_PER_WORKER: Final = (
    4  # more, smaller chunks even out books of very different size
)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A run of consecutive files; ``paths[i]`` gets doc_id ``first_doc_id + i``."""

    first_doc_id: int
    paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class PartialIndex:
    """The index of one chunk, plus who built it and how much CPU that took."""

    first_doc_id: int
    postings: dict[str, ArrayPostings]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]
    positions: bool
    pid: int
    process_cpu: float  # time.process_time() of the worker process after this chunk


def chunks(paths: Sequence[Path], count: int) -> list[Chunk]:
    """Split ``paths`` into at most ``count`` chunks of consecutive files."""
    if not paths:
        return []
    size = math.ceil(len(paths) / max(1, count))
    return [
        Chunk(start, tuple(paths[start : start + size]))
        for start in range(0, len(paths), size)
    ]


def build_partial(chunk: Chunk, positions: bool = True) -> PartialIndex:
    """Index one chunk. Module-level, so a process pool can pickle a reference to it."""
    index = build_index(
        read_documents(chunk.paths, chunk.first_doc_id), "array", positions
    )
    return PartialIndex(
        first_doc_id=chunk.first_doc_id,
        postings={term: to_arrays(plist) for term, plist in index.items()},
        doc_lengths=index.doc_lengths,
        doc_meta=index.doc_meta,
        positions=positions,
        pid=os.getpid(),
        process_cpu=time.process_time(),
    )


@timed
def merge(partials: Iterable[PartialIndex]) -> Index:
    """One index from partial ones, whatever order they finished in.

    Terms keep the order in which a single serial pass would first meet them,
    so the result saves to exactly the same bytes as ``build_index``.
    """
    ordered = sorted(partials, key=lambda p: p.first_doc_id)
    postings: dict[str, ArrayPostings] = {}
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    for part in ordered:
        for term, plist in part.postings.items():
            merged = postings.get(term)
            if merged is None:
                postings[term] = plist  # first chunk with this term: take it as is
            else:
                merged.doc_ids.extend(plist.doc_ids)
                merged.tfs.extend(plist.tfs)
                merged.positions.extend(plist.positions)
        doc_lengths.update(part.doc_lengths)
        doc_meta.update(part.doc_meta)
    positions = ordered[0].positions if ordered else False
    return Index(dict(postings), doc_lengths, doc_meta, "array", positions)


@dataclass(frozen=True, slots=True)
class BuildReport:
    index: Index
    executor: ExecutorName
    workers: int
    tasks: int
    wall: float  # seconds, the whole build including the merge
    merge: float  # seconds spent merging in the parent
    worker_cpu: float  # CPU seconds of worker processes (processes executor only)


def _make_pool(executor: ExecutorName, workers: int) -> Executor:
    if executor == "threads":
        return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="findex")
    # "spawn" everywhere (the default on Windows and macOS): every worker is a
    # fresh interpreter that imports findex, never a fork of a parent that may
    # hold locks or threads. That is why launches must sit under __main__.
    return ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    )


type OnChunk = Callable[[Chunk], None]


def _run_pool(
    executor: ExecutorName,
    workers: int,
    work: list[Chunk],
    positions: bool,
    on_chunk: OnChunk,
) -> list[PartialIndex]:
    pool = _make_pool(executor, workers)
    futures: dict[Future[PartialIndex], Chunk] = {}
    try:
        futures = {pool.submit(build_partial, c, positions): c for c in work}
        partials: list[PartialIndex] = []
        for future in as_completed(futures):
            # result() re-raises a worker's exception here, with the worker's
            # own traceback attached (as __cause__ for processes), so a failed
            # chunk fails the whole build instead of hanging it.
            partials.append(future.result())
            on_chunk(futures[future])
        return partials
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _ignore(_: Chunk) -> None:
    pass


def build_parallel(
    paths: Sequence[Path],
    executor: ExecutorName = "processes",
    workers: int | None = None,
    positions: bool = True,
    on_chunk: OnChunk = _ignore,
) -> BuildReport:
    """Index ``paths`` with ``executor``; ``workers`` defaults to ``os.cpu_count()``.

    ``on_chunk`` is called in the calling thread as each chunk finishes (for a
    progress bar). "serial" runs the same chunks one after another in a loop.
    """
    workers = 1 if executor == "serial" else max(1, workers or os.cpu_count() or 1)
    work = chunks(paths, workers * TASKS_PER_WORKER)
    log.info("%d files in %d chunks, %d %s", len(paths), len(work), workers, executor)
    started = time.perf_counter()
    if executor == "serial":
        partials: list[PartialIndex] = []
        for chunk in work:
            partials.append(build_partial(chunk, positions))
            on_chunk(chunk)
    else:
        partials = _run_pool(executor, workers, work, positions, on_chunk)

    merge_started = time.perf_counter()
    index = merge(partials)
    merge_seconds = time.perf_counter() - merge_started
    wall = time.perf_counter() - started

    worker_cpu = 0.0
    if executor == "processes":
        # Each worker reports its process CPU after every chunk; the last report
        # per process is that process's total so far.
        last: dict[int, float] = {}
        for part in partials:
            last[part.pid] = max(last.get(part.pid, 0.0), part.process_cpu)
        worker_cpu = sum(last.values())
    log.info("built in %.2f s, merge %.2f s", wall, merge_seconds)
    return BuildReport(
        index, executor, workers, len(work), wall, merge_seconds, worker_cpu
    )


def gil_enabled() -> bool:
    """False on a free-threaded build running without the GIL (3.13t, 3.14t)."""
    check = getattr(sys, "_is_gil_enabled", None)  # absent before 3.13
    return True if check is None else bool(check())
