"""Numbers for the Lab 2 README: ``python scripts/measure_index.py data``.

Prints Markdown tables: postings layouts, the cost of positions, pickle vs the
binary format, and the merge vs set benchmark.
"""

import gc
import subprocess
import sys
import tempfile
import time
import timeit
import tracemalloc
from pathlib import Path

from findex.corpus import iter_documents
from findex.index import build_index
from findex.model import LAYOUTS, Index
from findex.query import parse
from findex.store import load, save

MIB = 2**20


def measured_build(
    root: Path, layout: str, positions: bool
) -> tuple[Index, float, int, int]:
    """(index, seconds, peak bytes, bytes still held once built)."""
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    index = build_index(iter_documents(root), layout, positions)
    elapsed = time.perf_counter() - started
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return index, elapsed, peak, retained


_LOAD_ONCE = (
    "import sys, time; from pathlib import Path; from findex.store import load; "
    "t = time.perf_counter(); load(Path(sys.argv[1])); print(time.perf_counter() - t)"
)


def load_seconds(path: Path) -> float:
    """Best of 3 loads, each in a fresh process: no other objects for gc to scan."""
    runs = [
        float(
            subprocess.run(
                [sys.executable, "-c", _LOAD_ONCE, str(path)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        for _ in range(3)
    ]
    return min(runs)


def save_and_load(index: Index, path: Path) -> tuple[int, float, float]:
    """(file bytes, save seconds, load seconds)."""
    started = time.perf_counter()
    save(index, path)
    saved = time.perf_counter() - started
    return path.stat().st_size, saved, load_seconds(path)


def main(root: Path) -> None:
    tmp = Path(tempfile.mkdtemp())
    print("## Postings layouts (pickle file)\n")
    print("| Layout | Build | Peak memory | Retained index | File | Load |")
    print("|---|---|---|---|---|---|")
    formats = []
    for layout in LAYOUTS:
        index, built, peak, retained = measured_build(root, layout, False)
        size, _, loaded = save_and_load(index, tmp / f"{layout}.pkl")
        print(
            f"| {layout} | {built:.1f} s | {peak / MIB:.1f} MiB | {retained / MIB:.1f} MiB "
            f"| {size / MIB:.1f} MiB | {loaded:.2f} s |"
        )
        if layout == "slots":
            pkl_saved = save_and_load(index, tmp / "index.pkl")
            formats = [
                ("pickle", pkl_saved),
                ("binary", save_and_load(index, tmp / "index.bin")),
            ]
        del index

    print("\n## Positions\n")
    print("| Layout | Retained index | Pickle file | Binary file |")
    print("|---|---|---|---|")
    for layout in ("slots", "array"):
        index, _, _, retained = measured_build(root, layout, True)
        pkl, _, _ = save_and_load(index, tmp / f"{layout}-pos.pkl")
        binary, _, _ = save_and_load(index, tmp / f"{layout}-pos.bin")
        print(
            f"| {layout} + positions | {retained / MIB:.1f} MiB | {pkl / MIB:.1f} MiB | {binary / MIB:.1f} MiB |"
        )
        del index

    print("\n## Formats (slots index, no positions)\n")
    print("| Format | File | Save | Load | Loads as |")
    print("|---|---|---|---|---|")
    for kind, (size, saved, loaded) in formats:
        loads_as = "slots" if kind == "pickle" else "array"
        print(
            f"| {kind} | {size / MIB:.1f} MiB | {saved:.2f} s | {loaded:.2f} s | {loads_as} |"
        )

    index = load(tmp / "index.bin")
    by_freq = sorted(
        index.postings,
        key=lambda t: (len(index.postings[t]), sum(index.postings[t].tfs), t),
    )
    rare, common = by_freq[:2], by_freq[-2:][::-1]
    print("\n## merge vs set (binary index, array layout)\n")
    print(
        f"Most common: {', '.join(f'`{t}` (df={len(index.postings[t])})' for t in common)}; "
        f"rarest: {', '.join(f'`{t}` (df={len(index.postings[t])})' for t in rare)}.\n"
    )
    print("| Query | Hits | merge | set |")
    print("|---|---|---|---|")
    for a, b in (common, rare):
        for query in (f"{a} {b}", f"{a} OR {b}", f"{a} NOT {b}"):
            hits = len(parse(query).evaluate(index))
            times = []
            for engine in ("merge", "set"):
                timer = timeit.Timer(
                    lambda q=query, e=engine: parse(q).evaluate(index, e)
                )
                number, _ = timer.autorange()
                best = min(timer.repeat(5, number)) / number
                times.append(f"{best * 1e6:.1f} µs")
            print(f"| `{query}` | {hits} | {times[0]} | {times[1]} |")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data"))
