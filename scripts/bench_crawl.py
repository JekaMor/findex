"""Crawl benchmark: the same seed and page budget at different concurrency.

    python scripts/bench_crawl.py https://docs.python.org/3/library/ --out bench/crawl.json

Each configuration runs in a fresh process (so peak RSS is that crawl's own) and
leaves its per-URL log in bench/crawl-<name>.log.
"""

import argparse
import asyncio
import contextlib
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from findex.crawler import CrawlStats, crawl
from findex.sysinfo import peak_rss

# (name, concurrency, per_host, delay between starts to one host)
CONFIGS = [
    ("c1", 1, 1, 0.05),
    ("c5", 5, 5, 0.05),
    ("c20", 20, 20, 0.05),
    ("c20-polite", 20, 2, 0.25),  # the defaults: what politeness costs
]


async def _crawl(
    seed: str, max_pages: int, concurrency: int, per_host: int, delay: float
) -> CrawlStats:
    stats = CrawlStats()
    with tempfile.TemporaryFile("w+", encoding="utf-8") as sink:
        async with contextlib.aclosing(
            crawl(
                [seed],
                max_pages=max_pages,
                concurrency=concurrency,
                per_host=per_host,
                delay=delay,
                stats=stats,
            )
        ) as pages:
            async for page in pages:
                sink.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")
    return stats


def one(
    seed: str, max_pages: int, concurrency: int, per_host: int, delay: float, log: Path
) -> dict[str, object]:
    import logging

    handler = logging.FileHandler(log, mode="w", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    logger = logging.getLogger("findex.crawler")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    started = time.perf_counter()
    stats = asyncio.run(_crawl(seed, max_pages, concurrency, per_host, delay))
    wall = time.perf_counter() - started
    return {
        "pages": stats.pages,
        "requests": stats.requests,
        "errors": stats.errors,
        "skipped": stats.skipped,
        "statuses": stats.statuses,
        "wall": wall,
        "pages_per_s": stats.pages / wall,
        "peak_rss": peak_rss(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed")
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("bench/crawl.json"))
    parser.add_argument(
        "--one", nargs=4, metavar=("CONCURRENCY", "PER_HOST", "DELAY", "LOG")
    )
    args = parser.parse_args()

    if args.one:
        concurrency, per_host, delay, log = args.one
        print(
            json.dumps(
                one(
                    args.seed,
                    args.max_pages,
                    int(concurrency),
                    int(per_host),
                    float(delay),
                    Path(log),
                )
            )
        )
        return

    rows = []
    for name, concurrency, per_host, delay in CONFIGS:
        log = args.out.parent / f"crawl-{name}.log"
        result = subprocess.run(
            [
                sys.executable,
                __file__,
                args.seed,
                "--max-pages",
                str(args.max_pages),
                "--one",
                str(concurrency),
                str(per_host),
                str(delay),
                str(log),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        row = {
            "name": name,
            "concurrency": concurrency,
            "per_host": per_host,
            "delay": delay,
            **json.loads(result.stdout),
        }
        rows.append(row)
        print(
            f"{name:>11}: {row['pages']} pages in {row['wall']:.1f} s = {row['pages_per_s']:.1f}/s, "
            f"requests {row['requests']}, errors {row['errors']}, rss {row['peak_rss'] / 2**20:.0f} MiB",
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"seed": args.seed, "max_pages": args.max_pages, "rows": rows}, indent=1
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
