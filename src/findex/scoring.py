"""Scorers: what one posting of a query term adds to its document's score.

A scorer is anything that can be called as ``scorer(term, posting, index)`` and
returns a float. ``TfIdf`` and ``BM25`` are classes with ``__call__``; a plain
function with that signature works just as well. Nobody inherits from ``Scorer``:
it only describes the shape (a ``typing.Protocol``).
"""

import math
from dataclasses import dataclass
from typing import Protocol

from findex.model import Index, PostingLike


class Scorer(Protocol):
    def __call__(self, term: str, posting: PostingLike, index: Index) -> float: ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    """``tf · log(N / df)``: grows linearly with tf, ignores document length."""

    def __call__(self, term: str, posting: PostingLike, index: Index) -> float:
        return posting.tf * math.log(index.num_docs / index.df(term))


@dataclass(frozen=True, slots=True)
class BM25:
    """Okapi BM25.

    ``k1`` sets how fast tf saturates: each extra occurrence adds less, and the
    score never exceeds ``idf · (k1 + 1)``. ``b`` sets how much a document longer
    than average is penalised: ``b = 0`` ignores length, ``b = 1`` divides tf by
    the full length ratio.
    """

    k1: float = 1.5
    b: float = 0.75

    def idf(self, term: str, index: Index) -> float:
        # The "+ 1" keeps it positive even for a term in every document.
        n, df = index.num_docs, index.df(term)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def __call__(self, term: str, posting: PostingLike, index: Index) -> float:
        tf = posting.tf
        length_ratio = index.doc_length(posting.doc_id) / index.avg_doc_length
        saturation = (
            tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length_ratio))
        )
        return self.idf(term, index) * saturation
