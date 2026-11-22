"""Lab 3's per-posting scorer vs the NumPy one, on real indexes.

    python scripts/make_passages.py data data/passages.jsonl     # once
    findex index data/passages.jsonl -o passages.bin             # once
    python scripts/bench_scoring.py index.bin passages.bin

For each index: a very common term, a rare one and a three-term query. Times are
the median of many runs of the ranking alone (match is cached, so both rankers
pay the same for it), BM25, k = 10.
"""

import statistics
import sys
import timeit
from pathlib import Path

from findex.query import parse
from findex.scoring import BM25
from findex.search import rank_numpy, rank_python
from findex.store import open_index


def timed_ms(fn: object, repeat: int = 7) -> float:
    timer = timeit.Timer(fn)  # type: ignore[arg-type]  # a zero-argument callable
    number, _ = timer.autorange()
    return statistics.median(timer.repeat(repeat, number)) / number * 1000


def main(paths: list[Path]) -> None:
    scorer = BM25()
    print(
        "| Index (documents) | Query | Docs matched | Lab 3 scorer, ms | NumPy, ms | Speedup |"
    )
    print("|---|---|---|---|---|---|")
    for path in paths:
        with open_index(path) as index:
            by_df = sorted(index, key=index.df)
            rare = next(t for t in by_df if t.isalpha() and index.df(t) >= 1)
            queries = {
                "common term": "the",
                "rare term": rare,
                "3 terms": "sea OR ship OR captain",
            }
            for label, q in queries.items():
                node = parse(q)
                matched = len(index.match(node))
                assert rank_numpy(index, node, scorer, 10) == rank_python(
                    index, node, scorer, 10
                )
                py = timed_ms(lambda node=node: rank_python(index, node, scorer, 10))
                np_ = timed_ms(lambda node=node: rank_numpy(index, node, scorer, 10))
                print(
                    f"| {path.name} ({index.num_docs:,}) | {label}: `{q}` | {matched:,} | "
                    f"{py:.3f} | {np_:.3f} | {py / np_:.1f}× |"
                )


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:]])
