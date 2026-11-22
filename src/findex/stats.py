"""Corpus statistics (``findex stats``): documents, tokens, vocabulary, top terms.

From a corpus directory they are counted in one streaming pass (Lab 1); from an
index file they are read off the postings.
"""

import heapq
from collections import Counter
from collections.abc import Iterator
from itertools import islice
from operator import itemgetter
from pathlib import Path
from typing import NamedTuple

from findex.corpus import iter_documents
from findex.model import ArrayPostings, Index
from findex.tokenize import tokenize


class Stats(NamedTuple):
    documents: int
    tokens: int
    vocabulary: int
    top: tuple[tuple[str, int], ...]


def _summarise(documents: int, counts: Counter[str], top: int) -> Stats:
    return Stats(documents, counts.total(), len(counts), tuple(counts.most_common(top)))


def lazy_stats(root: Path, limit: int | None = None, top: int = 50) -> Stats:
    """Stream documents -> tokens -> Counter; only the Counter grows."""
    counts: Counter[str] = Counter()
    documents = 0
    for doc in islice(iter_documents(root), limit):
        documents += 1
        counts.update(tokenize(doc.text))
    return _summarise(documents, counts, top)


def eager_stats(root: Path, limit: int | None = None, top: int = 50) -> Stats:
    """Deliberately greedy: every text, then every token, held in lists."""
    docs = list(islice(iter_documents(root), limit))
    token_lists = [list(tokenize(doc.text)) for doc in docs]
    counts: Counter[str] = Counter()
    for tokens in token_lists:
        counts.update(tokens)
    return _summarise(len(docs), counts, top)


def _term_counts(index: Index) -> Iterator[tuple[str, int]]:
    """(term, occurrences in the whole corpus), one term at a time."""
    for term, plist in index.items():
        if isinstance(plist, ArrayPostings):  # sum the tf array, build no Postings
            yield term, sum(plist.tfs)
        else:
            yield term, sum(p.tf for p in plist)


def index_stats(index: Index, top: int = 50) -> Stats:
    """The same numbers as ``lazy_stats``, read from a built index."""
    best = heapq.nlargest(top, _term_counts(index), key=itemgetter(1))
    tokens = sum(index.doc_lengths.values())
    return Stats(index.num_docs, tokens, len(index), tuple(best))
