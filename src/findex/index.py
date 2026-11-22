"""Build the inverted index in one streaming pass and save it.

``python -m findex.index data/ --out index.bin``
"""

import argparse
import re
import time
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from itertools import islice
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.model import (
    LAYOUTS,
    ArrayPostings,
    DocMeta,
    Index,
    Layout,
    PlainPosting,
    Posting,
)
from findex.store import FORMATS, save
from findex.tokenize import tokenize

_TITLE = re.compile(r"^Title:[ \t]*(.+?)\s*$", re.MULTILINE)


def title_of(doc: Document) -> str:
    """The Gutenberg ``Title:`` header line, or the file name without extension."""
    match = _TITLE.search(doc.text, 0, 5000)
    return match.group(1) if match else doc.path.stem


def _terms(
    text: str, positions: bool
) -> tuple[int, Iterable[tuple[str, int, tuple[int, ...]]]]:
    """(document length, (term, tf, positions) for each distinct term)."""
    if not positions:
        counts = Counter(tokenize(text))
        return counts.total(), ((term, tf, ()) for term, tf in counts.items())
    where: defaultdict[str, list[int]] = defaultdict(list)
    length = 0
    for length, term in enumerate(tokenize(text), 1):
        where[term].append(length - 1)
    return length, ((term, len(pos), tuple(pos)) for term, pos in where.items())


def build_index(
    docs: Iterable[Document], layout: Layout = "slots", positions: bool = False
) -> Index:
    """Consume ``docs`` once. Only the index grows; the corpus keeps streaming.

    Documents must arrive in increasing ``doc_id`` order (``iter_documents``
    guarantees it), so appending keeps every postings list sorted for free.
    """
    acc: defaultdict[str, list] | defaultdict[str, ArrayPostings]
    add: Callable[[str, int, int, tuple[int, ...]], None]
    if layout == "array":
        arrays: defaultdict[str, ArrayPostings] = defaultdict(ArrayPostings)
        acc = arrays

        def add(term, doc_id, tf, pos):
            arrays[term].append(doc_id, tf, pos)
    else:
        record = Posting if layout == "slots" else PlainPosting
        lists: defaultdict[str, list] = defaultdict(list)
        acc = lists

        def add(term, doc_id, tf, pos):
            lists[term].append(record(doc_id, tf, pos))

    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    last = -1
    for doc in docs:
        if doc.doc_id <= last:
            raise ValueError(
                f"doc_id {doc.doc_id} after {last}: postings would be unsorted"
            )
        last = doc.doc_id
        # The current document's counter is flushed into the postings, then dropped.
        length, terms = _terms(doc.text, positions)
        for term, tf, pos in terms:
            add(term, doc.doc_id, tf, pos)
        doc_lengths[doc.doc_id] = length
        doc_meta[doc.doc_id] = DocMeta(str(doc.path), title_of(doc))

    # A plain dict: looking up an unknown term must not insert it.
    return Index(dict(acc), doc_lengths, doc_meta, layout, positions)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="findex.index", description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--format", choices=FORMATS, help="default: pickle for .pkl, else binary"
    )
    parser.add_argument(
        "--layout", choices=LAYOUTS, default="slots", help="postings in memory"
    )
    parser.add_argument(
        "--positions", action="store_true", help="also store token positions"
    )
    parser.add_argument("--limit", type=int, help="only the first N documents")
    args = parser.parse_args(argv)

    tracemalloc.start()
    started = time.perf_counter()
    index = build_index(
        islice(iter_documents(args.root), args.limit), args.layout, args.positions
    )
    built = time.perf_counter() - started
    started = time.perf_counter()
    fmt = save(index, args.out, args.format)
    saved = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"documents:    {len(index.doc_meta)}")
    print(f"terms:        {len(index.postings)}")
    print(f"postings:     {index.posting_count()}")
    print(f"layout:       {index.layout}{' + positions' if index.positions else ''}")
    print(f"build:        {built:.2f} s")
    print(f"save:         {saved:.2f} s ({fmt})")
    print(f"file:         {args.out} ({args.out.stat().st_size / 2**20:.1f} MiB)")
    print(f"peak memory:  {peak / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
