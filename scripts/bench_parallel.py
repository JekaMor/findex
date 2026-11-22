"""Benchmark parallel indexing: serial vs threads vs processes, GIL vs free-threaded.

    # one interpreter's rows (each measurement runs in a fresh process):
    python scripts/bench_parallel.py run data --label gil --out bench/gil.json
    .venv-ft/Scripts/python scripts/bench_parallel.py run data --label free-threaded \
        --out bench/ft.json --executors serial threads
    # table + plot from the JSON files (needs matplotlib: uv run --group bench):
    uv run --group bench python scripts/bench_parallel.py report bench/gil.json bench/ft.json

``run`` re-launches this script with ``one`` for every measurement, using the same
interpreter it was started with.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

from findex.corpus import corpus_paths
from findex.parallel import build_parallel, gil_enabled
from findex.sysinfo import peak_rss

WORKERS = [1, 2, 4, 8, os.cpu_count() or 1]


def one(corpus: Path, executor: str, workers: int) -> dict[str, object]:
    """A single build, measured from inside the process that runs it."""
    paths = corpus_paths(corpus)
    cpu_started = time.process_time()
    report = build_parallel(paths, executor, workers, positions=True)  # type: ignore[arg-type]
    parent_cpu = time.process_time() - cpu_started
    return {
        "executor": executor,
        "workers": workers,
        "wall": report.wall,
        "merge": report.merge,
        # Threads run inside the parent, so its CPU time includes them; worker
        # processes report their own (start-up and imports included).
        "cpu": parent_cpu + report.worker_cpu,
        "peak_rss": peak_rss(),
        "gil": gil_enabled(),  # checked after the build: imports can re-enable it
        "documents": report.index.num_docs,
    }


def configs(executors: list[str]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for executor in executors:
        counts = [1] if executor == "serial" else sorted(set(WORKERS))
        out += [(executor, n) for n in counts]
    return out


def run(corpus: Path, label: str, out: Path, executors: list[str], repeat: int) -> None:
    def measure(executor: str, workers: int) -> dict[str, object]:
        result = subprocess.run(
            [sys.executable, __file__, "one", str(corpus), executor, str(workers)],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    measure("serial", 1)  # warm-up: file cache, .pyc files
    rows = []
    for executor, workers in configs(executors):
        samples = [measure(executor, workers) for _ in range(repeat)]
        row = {
            "label": label,
            "executor": executor,
            "workers": workers,
            "runs": repeat,
            "gil": samples[0]["gil"],
            **{
                key: statistics.median(float(s[key]) for s in samples)  # type: ignore[arg-type]
                for key in ("wall", "cpu", "peak_rss", "merge")
            },
            "wall_all": [s["wall"] for s in samples],
        }
        print(
            f"{label:>14} {executor:>9} x{workers}: wall {row['wall']:.2f}s "
            f"cpu {row['cpu']:.2f}s rss {row['peak_rss'] / 2**20:.0f}MiB "
            f"merge {row['merge']:.2f}s",
            flush=True,
        )
        rows.append(row)
    meta = {
        "python": sys.version.split()[0],
        "free_threaded_build": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "machine": platform.processor(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1), encoding="utf-8")


def report(files: list[Path], plot: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    rows = [row for d in data for row in d["rows"]]
    baseline = next(
        r["wall"] for r in rows if r["executor"] == "serial" and r["gil"]
    )  # serial, normal build: speedup 1.0x
    print(
        "| Python | Executor | Workers | Wall (s) | CPU (s) | Peak RSS (MiB) | Merge (s) | Speedup |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        python = "3.14t, GIL off" if not r["gil"] else "3.14, GIL on"
        print(
            f"| {python} | {r['executor']} | {r['workers']} | {r['wall']:.2f} | "
            f"{r['cpu']:.2f} | {r['peak_rss'] / 2**20:.0f} | {r['merge']:.2f} | "
            f"{baseline / r['wall']:.2f}× |"
        )

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    top = max(WORKERS)
    ax.plot([1, top], [1, top], "--", color="grey", label="ideal (linear)")
    series = {}
    for r in rows:
        if r["executor"] == "serial":
            continue
        name = f"{r['executor']}, {'GIL off (3.14t)' if not r['gil'] else 'GIL on'}"
        series.setdefault(name, []).append((r["workers"], baseline / r["wall"]))
    for name, points in series.items():
        points.sort()
        ax.plot(*zip(*points, strict=True), marker="o", label=name)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(set(WORKERS)), [str(w) for w in sorted(set(WORKERS))])
    ax.set_xlabel("workers")
    ax.set_ylabel("speedup over serial (wall time)")
    ax.set_ylim(0, top + 0.5)
    ax.set_title("findex index: speedup vs workers (40 books, 30 MB)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot)
    print(f"\nplot: {plot}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_one = sub.add_parser("one")
    p_one.add_argument("corpus", type=Path)
    p_one.add_argument("executor")
    p_one.add_argument("workers", type=int)
    p_run = sub.add_parser("run")
    p_run.add_argument("corpus", type=Path)
    p_run.add_argument("--label", required=True)
    p_run.add_argument("--out", type=Path, required=True)
    p_run.add_argument("--repeat", type=int, default=3)
    p_run.add_argument(
        "--executors", nargs="+", default=["serial", "threads", "processes"]
    )
    p_report = sub.add_parser("report")
    p_report.add_argument("files", nargs="+", type=Path)
    p_report.add_argument("--plot", type=Path, default=Path("bench/speedup.png"))
    args = parser.parse_args()

    if args.command == "one":
        print(json.dumps(one(args.corpus, args.executor, args.workers)))
    elif args.command == "run":
        run(args.corpus, args.label, args.out, args.executors, args.repeat)
    else:
        report(args.files, args.plot)


# Required with "spawn": worker processes import this module again.
if __name__ == "__main__":
    main()
