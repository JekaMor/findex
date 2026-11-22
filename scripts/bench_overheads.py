"""Where the process pool's time goes: ``python scripts/bench_overheads.py data``.

Measures, outside any pool, the parts that do not shrink with more workers:
starting spawn workers, pickling partial indexes back to the parent, the
slowest single chunk, and the merge.
"""

import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from findex.corpus import corpus_paths
from findex.parallel import TASKS_PER_WORKER, build_partial, chunks, merge

HOLD = 0.5  # each task keeps its worker busy, so the pool must start them all


def ready(_: int) -> int:
    import findex.parallel  # noqa: F401  # what a real worker imports

    time.sleep(HOLD)
    return os.getpid()


def main(corpus: Path) -> None:
    paths = corpus_paths(corpus)
    for workers in (2, 4, 8):
        started = time.perf_counter()
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
            pids = set(pool.map(ready, range(workers)))
        overhead = time.perf_counter() - started - HOLD
        print(
            f"spawn {workers} workers, import findex, shut down: "
            f"{overhead:.2f} s ({len(pids)} processes)"
        )

    for workers in (1, 4, 8):
        work = chunks(paths, workers * TASKS_PER_WORKER)
        partials, times = [], []
        for chunk in work:
            started = time.perf_counter()
            partials.append(build_partial(chunk))
            times.append(time.perf_counter() - started)
        started = time.perf_counter()
        blobs = [pickle.dumps(p, pickle.HIGHEST_PROTOCOL) for p in partials]
        dumped = time.perf_counter() - started
        started = time.perf_counter()
        back = [pickle.loads(b) for b in blobs]
        loaded = time.perf_counter() - started
        started = time.perf_counter()
        merge(back)
        merged = time.perf_counter() - started
        print(
            f"\n{workers} workers -> {len(work)} chunks: work {sum(times):.2f} s, "
            f"slowest chunk {max(times):.2f} s ({max(times) / sum(times):.0%} of the work)"
        )
        print(
            f"  results pickled: {sum(map(len, blobs)) / 2**20:.1f} MiB, "
            f"dumps {dumped:.2f} s + loads {loaded:.2f} s; merge {merged:.2f} s"
        )
    print(f"\nPython {sys.version.split()[0]}, os.cpu_count() = {os.cpu_count()}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data"))
