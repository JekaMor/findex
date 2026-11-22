"""Lost updates on a shared Counter: with the GIL, and without it.

    python scripts/race_demo.py data                 # normal build
    .venv-ft/Scripts/python scripts/race_demo.py data    # free-threaded build

Eight threads count the tokens of the same books into one Counter. Every
variant must end with exactly the serial total; the unlocked one may not.
"""

import argparse
import dis
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize

THREADS = 8


def shared_no_lock(chunks: list[list[str]]) -> Counter[str]:
    counts: Counter[str] = Counter()

    def work(tokens: list[str]) -> None:
        for token in tokens:
            counts[token] += 1  # read, add, write back: three steps, no lock

    _run(work, chunks)
    return counts


def shared_with_lock(chunks: list[list[str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    lock = threading.Lock()

    def work(tokens: list[str]) -> None:
        for token in tokens:
            with lock:
                counts[token] += 1

    _run(work, chunks)
    return counts


def per_thread_then_merge(chunks: list[list[str]]) -> Counter[str]:
    """What findex does: no shared state while working, one merge at the end."""
    partial: list[Counter[str]] = [Counter() for _ in chunks]

    def work_on(i: int) -> Callable[[list[str]], None]:
        def work(tokens: list[str]) -> None:
            mine = partial[i]
            for token in tokens:
                mine[token] += 1

        return work

    threads = [
        threading.Thread(target=work_on(i), args=(c,)) for i, c in enumerate(chunks)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total: Counter[str] = Counter()
    for c in partial:
        total.update(c)
    return total


def _run(work: Callable[[list[str]], None], chunks: list[list[str]]) -> None:
    threads = [threading.Thread(target=work, args=(c,)) for c in chunks]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--books", type=int, default=6)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    tokens = [
        t
        for d in islice(iter_documents(args.corpus), args.books)
        for t in tokenize(d.text)
    ]
    expected = len(tokens)
    chunks = [tokens[i::THREADS] for i in range(THREADS)]
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(f"Python {sys.version.split()[0]}, sys._is_gil_enabled() = {gil}")
    print(
        f"{THREADS} threads, {expected:,} tokens from {args.books} books, {args.repeat} runs\n"
    )
    print("| Variant | Lost updates (min … max) | Time (s, median) |")
    print("|---|---|---|")
    for variant in (shared_no_lock, shared_with_lock, per_thread_then_merge):
        lost: list[int] = []
        times: list[float] = []
        for _ in range(args.repeat):
            started = time.perf_counter()
            counts = variant(chunks)
            times.append(time.perf_counter() - started)
            lost.append(expected - counts.total())
        times.sort()
        print(
            f"| `{variant.__name__}` | {min(lost):,} … {max(lost):,} | "
            f"{times[len(times) // 2]:.2f} |"
        )

    print("\n`counts[token] += 1` compiles to:\n")
    dis.dis(compile("counts[token] += 1", "<demo>", "exec"))


if __name__ == "__main__":
    main()
