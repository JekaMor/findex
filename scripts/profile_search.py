"""A steady-state search workload to profile: the index opened once, then many
three-term queries, the way the web service runs them.

    python scripts/profile_search.py index.bin                 # BM25 + snippets
    python scripts/profile_search.py index.bin --no-snippets   # ranking only
    py-spy record -o docs/flame-search.svg -- python scripts/profile_search.py index.bin
"""

import argparse
import itertools
import time
from pathlib import Path

from findex.corpus import document_text
from findex.query import parse
from findex.search import search
from findex.snippets import snippet_parts
from findex.store import open_index

WORDS = (
    "sea ship captain whale night house love war king queen death life father mother "
    "letter money god heart fire water blood gold horse church friend door"
).split()


def queries(n: int) -> list[str]:
    """n different three-term queries (OR, so every one has many matches)."""
    triples = itertools.islice(itertools.combinations(WORDS, 3), n)
    return [" OR ".join(t) for t in triples]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("-n", type=int, default=200)
    parser.add_argument("--no-snippets", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    with open_index(args.index) as index:
        for q in queries(args.n):
            node = parse(q)
            results = search(index, node, k=10)
            if not args.no_snippets:
                terms = set(node.terms())
                for r in results:
                    text = document_text(r.path, r.url)
                    if text is not None:
                        snippet_parts(text, terms)
    took = time.perf_counter() - started
    print(f"{args.n} queries in {took:.2f} s ({took / args.n * 1000:.1f} ms each)")


if __name__ == "__main__":
    main()
