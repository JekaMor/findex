"""The three ranking sanity checks on the real corpus.

``python scripts/ranking_checks.py index.bin`` prints Markdown for the README.
"""

import sys
from pathlib import Path

from findex.model import Index, Posting
from findex.scoring import BM25, TfIdf
from findex.search import search
from findex.store import open_index

SCORERS = {"BM25": BM25(), "TF-IDF": TfIdf()}


def name(index: Index, doc_id: int) -> str:
    return Path(index.doc_meta[doc_id].path).name


def tf(index: Index, term: str, doc_id: int) -> int:
    return next((p.tf for p in index.get(term, ()) if p.doc_id == doc_id), 0)


def doc_named(index: Index, file_name: str) -> int:
    return next(d for d in index.doc_meta if name(index, d) == file_name)


def rare_beats_common(index: Index, common: str, rare: str) -> None:
    print(f"### 1. `{common} OR {rare}`\n")
    print(f"df({common}) = {index.df(common)}, df({rare}) = {index.df(rare)}.\n")
    for label, scorer in SCORERS.items():
        print(f"**{label}**, top 5:\n")
        print(f"| # | Book | words | tf({common}) | tf({rare}) | score |")
        print("|---|---|---|---|---|---|")
        for rank, r in enumerate(search(index, f"{common} OR {rare}", scorer, k=5), 1):
            print(
                f"| {rank} | {r.title[:40]} | {index.doc_length(r.doc_id):,} "
                f"| {tf(index, common, r.doc_id)} | {tf(index, rare, r.doc_id)} "
                f"| {r.score:.2f} |"
            )
        print()


def saturation(index: Index, term: str, file_name: str) -> None:
    doc_id = doc_named(index, file_name)
    print(f"### 2. `{term}` in {index.doc_meta[doc_id].title}\n")
    print(
        f"Real tf = {tf(index, term, doc_id)}; below the same document with tf "
        "set by hand.\n"
    )
    print("| tf | BM25 | +last | TF-IDF | +last |")
    print("|---|---|---|---|---|")
    for n in (1, 2, 3, 5, 10, 19, 20, 100):
        row = [str(n)]
        for scorer in SCORERS.values():
            now = scorer(term, Posting(doc_id, n), index)
            before = scorer(term, Posting(doc_id, n - 1), index) if n > 1 else 0.0
            row += [f"{now:.3f}", f"{now - before:+.3f}"]
        print("| " + " | ".join(row) + " |")
    print()


def short_beats_long(index: Index, term: str) -> None:
    print(f"### 3. `{term}`\n")
    print("| Book | words | tf | BM25 | TF-IDF |")
    print("|---|---|---|---|---|")
    bm25 = {r.doc_id: r.score for r in search(index, term, SCORERS["BM25"])}
    tfidf = {r.doc_id: r.score for r in search(index, term, SCORERS["TF-IDF"])}
    for doc_id, score in bm25.items():
        print(
            f"| {index.doc_meta[doc_id].title[:40]} | {index.doc_length(doc_id):,} "
            f"| {tf(index, term, doc_id)} | {score:.2f} | {tfidf[doc_id]:.2f} |"
        )
    print()


def main(path: Path) -> None:
    with open_index(path) as index:
        print(f"avg document length: {index.avg_doc_length:,.0f} words\n")
        rare_beats_common(index, "love", "yearling")
        saturation(index, "whale", "pg2701.txt")
        short_beats_long(index, "absentees")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "index.bin"))
