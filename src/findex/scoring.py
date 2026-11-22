"""Scorers: what a query term adds to the score of each document it occurs in.

Two shapes, both structural (``typing.Protocol``, nobody inherits):

- ``Scorer`` — ``scorer(term, posting, index) -> float``, one posting at a time
  (Lab 3). Any function with that signature is one.
- ``ArrayScorer`` — ``scorer.score_arrays(term, doc_ids, tfs, index)``: every
  posting of the term at once, as NumPy arrays (Lab 8). ``TfIdf`` and ``BM25``
  are both; ``search`` uses the array form when a scorer has it.

The array versions do the same floating-point operations in the same order as
the per-posting ones, so they give bit-for-bit the same scores.
"""

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from findex.model import Index, PostingLike

type Ints = npt.NDArray[np.int32]
type Floats = npt.NDArray[np.float64]


class Scorer(Protocol):
    def __call__(self, term: str, posting: PostingLike, index: Index) -> float: ...


@runtime_checkable
class ArrayScorer(Protocol):
    def score_arrays(
        self, term: str, doc_ids: Ints, tfs: Ints, index: Index
    ) -> Floats: ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    """``tf · log(N / df)``: grows linearly with tf, ignores document length."""

    def __call__(self, term: str, posting: PostingLike, index: Index) -> float:
        return posting.tf * math.log(index.num_docs / index.df(term))

    def score_arrays(self, term: str, doc_ids: Ints, tfs: Ints, index: Index) -> Floats:
        # The same log as above, once per term; then one multiply per posting.
        weight = math.log(index.num_docs / index.df(term))
        return (tfs * weight).astype(np.float64, copy=False)  # already float64


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

    def score_arrays(self, term: str, doc_ids: Ints, tfs: Ints, index: Index) -> Floats:
        # doc_length_array[doc_ids]: fancy indexing pulls this term's document
        # lengths in one step; the rest are ufuncs over whole arrays.
        length_ratio = index.doc_length_array[doc_ids] / index.avg_doc_length
        saturation = (
            tfs * (self.k1 + 1) / (tfs + self.k1 * (1 - self.b + self.b * length_ratio))
        )
        return self.idf(term, index) * saturation
