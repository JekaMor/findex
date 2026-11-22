"""Ranked search over a saved index.

``python -m findex.search index.bin 'python AND (async OR await) NOT java "event loop"'``

Several queries can be given at once; a repeated one is answered from the
cache (``-v`` shows it in the log). The query language is described in
``findex.query``.
"""

import argparse
import heapq
import logging
import sys
import time
import tracemalloc
from collections.abc import Sequence
from dataclasses import dataclass, field
from operator import itemgetter
from pathlib import Path

from findex.boolean import ENGINES
from findex.model import Index
from findex.query import Node, QueryError, parse
from findex.scoring import BM25, Scorer, TfIdf
from findex.snippets import snippet
from findex.store import open_index
from findex.timing import timed

SCORERS: dict[str, Scorer] = {"bm25": BM25(), "tfidf": TfIdf()}
DEFAULT_SCORER: Scorer = SCORERS["bm25"]


@dataclass(frozen=True, slots=True, order=True)
class SearchResult:
    """One ranked hit. Ordered by score, then doc_id, so ``sorted(results)`` works
    (title and path don't take part in comparisons)."""

    score: float
    doc_id: int
    title: str = field(compare=False)
    path: str = field(compare=False)

    def __str__(self) -> str:
        return f"{self.score:7.3f}  {self.title}  ({Path(self.path).name})"


def as_node(query: str | Node) -> Node:
    return query if isinstance(query, Node) else parse(query)


@timed
def search(
    index: Index,
    query: str | Node,
    scorer: Scorer = DEFAULT_SCORER,
    k: int = 10,
    engine: str = "merge",
) -> list[SearchResult]:
    """The ``k`` best documents matching ``query``, best first.

    The Boolean query decides which documents qualify; each query term then adds
    ``scorer(term, posting, index)`` for every qualifying document it occurs in.
    """
    node = as_node(query)
    scores = dict.fromkeys(index.match(node, engine), 0.0)
    for term in dict.fromkeys(node.terms()):  # each term once, in query order
        for posting in index.get(term, ()):
            if posting.doc_id in scores:
                scores[posting.doc_id] += scorer(term, posting, index)
    # O(n log k): a heap of the k best so far, not a sort of all n matches.
    best = heapq.nlargest(k, scores.items(), key=itemgetter(1))
    return [
        SearchResult(
            score, doc_id, index.doc_meta[doc_id].title, index.doc_meta[doc_id].path
        )
        for doc_id, score in best
    ]


def _text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="findex.search",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("index", type=Path)
    parser.add_argument("queries", nargs="+", metavar="query")
    parser.add_argument("--scorer", choices=sorted(SCORERS), default="bm25")
    parser.add_argument("-k", type=int, default=10, help="how many results")
    parser.add_argument("--engine", choices=sorted(ENGINES), default="merge")
    parser.add_argument("--no-snippets", action="store_true")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="log timings and cache hits"
    )
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO, format="  [log] %(message)s", stream=sys.stdout
        )
        logging.getLogger("findex").setLevel(logging.INFO)
    mark = ("\033[1;33m", "\033[0m") if sys.stdout.isatty() else ("**", "**")
    scorer = SCORERS[args.scorer]

    tracemalloc.start()
    started = time.perf_counter()
    with open_index(args.index) as index:
        print(f"index:        {index!r}  ({time.perf_counter() - started:.3f} s)")
        print(f"scorer:       {scorer!r}")
        for query in args.queries:
            print(f"\nquery:        {query}")
            try:
                node = parse(query)
                started = time.perf_counter()
                results = search(index, node, scorer, args.k, args.engine)
            except QueryError as exc:
                print(f"error:        {exc}")
                continue
            searched = time.perf_counter() - started
            print(f"parsed:       {node}")
            terms = set(node.terms())
            for rank, result in enumerate(results, 1):
                print(f"{rank:3}. {result}")
                text = None if args.no_snippets else _text(result.path)
                if text is not None:
                    print(f"       {snippet(text, terms, mark=mark)}")
            if not results:
                print("     nothing found")
            # Flush now, so pending output is not written inside the next timed call.
            print(f"search:       {searched * 1000:.2f} ms", flush=True)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\npeak memory:  {peak / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
