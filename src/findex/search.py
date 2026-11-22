"""Boolean search over a saved index.

``python -m findex.search index.bin "query" [--engine merge|set]``

Query language, evaluated left to right with no precedence:

- ``a b``     both (implicit AND); ``a AND b`` is the same
- ``a OR b``  either
- ``a NOT b`` a but not b; a leading ``NOT b`` means every document without b

Operators must be upper case; ``or`` is an ordinary word. Each word goes through
the same tokenizer as the corpus, so ``Alice's`` finds ``alice's``. A word that
splits into several tokens (``well-known``) needs all of them.
"""

import argparse
import operator
import time
import tracemalloc
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from findex.model import Index
from findex.store import load
from findex.tokenize import tokenize

Op = Literal["AND", "OR", "NOT"]
OPERATORS: frozenset[str] = frozenset({"AND", "OR", "NOT"})


# --- two-pointer merges over lists sorted by doc_id: O(len(a) + len(b)) ------


def intersect(a: Sequence[int], b: Sequence[int]) -> list[int]:
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        x, y = a[i], b[j]
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            i += 1
        else:
            j += 1
    return out


def union(a: Sequence[int], b: Sequence[int]) -> list[int]:
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        x, y = a[i], b[j]
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            out.append(x)
            i += 1
        else:
            out.append(y)
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def difference(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Items of ``a`` that are not in ``b``."""
    out: list[int] = []
    i = j = 0
    while i < len(a):
        if j == len(b) or a[i] < b[j]:
            out.append(a[i])
            i += 1
        elif a[i] == b[j]:
            i += 1
            j += 1
        else:
            j += 1
    return out


# --- engines -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Engine:
    """How to turn postings into an operand and combine operands."""

    fetch: Callable[[Sequence[int]], Any]
    and_: Callable[[Any, Any], Any]
    or_: Callable[[Any, Any], Any]
    not_: Callable[[Any, Any], Any]
    result: Callable[[Any], list[int]]


ENGINES: dict[str, Engine] = {
    # Works on the sorted lists as they are; every result stays sorted.
    "merge": Engine(lambda ids: ids, intersect, union, difference, list),
    # Hashes each postings list into a set first, sorts once at the end.
    "set": Engine(set, operator.and_, operator.or_, operator.sub, sorted),
}


def parse(query: str) -> list[tuple[Op, list[str]]]:
    """``"a OR b NOT c"`` -> ``[("AND", ["a"]), ("OR", ["b"]), ("NOT", ["c"])]``."""
    clauses: list[tuple[Op, list[str]]] = []
    op: Op = "AND"
    for word in query.split():
        if word in OPERATORS:
            op = word  # type: ignore[assignment]
            continue
        terms = list(tokenize(word))
        if terms:
            clauses.append((op, terms))
        op = "AND"
    return clauses


def search(index: Index, query: str, engine: str = "merge") -> list[int]:
    """Sorted doc_ids matching ``query``."""
    eng = ENGINES[engine]
    result = None
    for op, terms in parse(query):
        operand = eng.fetch(index.doc_ids(terms[0]))
        for term in terms[1:]:
            operand = eng.and_(operand, eng.fetch(index.doc_ids(term)))
        if result is None:
            result = (
                eng.not_(eng.fetch(index.all_doc_ids()), operand)
                if op == "NOT"
                else operand
            )
        elif op == "OR":
            result = eng.or_(result, operand)
        elif op == "NOT":
            result = eng.not_(result, operand)
        else:
            result = eng.and_(result, operand)
    return [] if result is None else eng.result(result)


def describe(index: Index, doc_ids: Iterable[int]) -> list[str]:
    return [
        f"{index.doc_meta[i].title}  ({Path(index.doc_meta[i].path).name})"
        for i in doc_ids
    ]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="findex.search",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("index", type=Path)
    parser.add_argument("query")
    parser.add_argument("--engine", choices=sorted(ENGINES), default="merge")
    parser.add_argument("--show", type=int, default=10, help="how many titles to print")
    args = parser.parse_args(argv)

    tracemalloc.start()
    started = time.perf_counter()
    index = load(args.index)
    loaded = time.perf_counter() - started
    started = time.perf_counter()
    hits = search(index, args.query, args.engine)
    searched = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"query:        {args.query}")
    print(f"engine:       {args.engine}")
    print(f"hits:         {len(hits)}")
    for line in describe(index, hits[: args.show]):
        print(f"  {line}")
    if len(hits) > args.show:
        print(f"  ... and {len(hits) - args.show} more")
    print(f"load:         {loaded:.3f} s")
    print(f"search:       {searched * 1000:.3f} ms")
    print(f"peak memory:  {peak / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
