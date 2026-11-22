"""Ranked search: the query decides which documents match, a scorer orders them.

Two ways to rank the matches, giving the same results:

- ``rank_python`` — Lab 3: a Python loop over every posting, a dict of scores,
  ``heapq.nlargest``. Kept as the reference, and for scorers that are plain
  per-posting functions.
- ``rank_numpy`` — Lab 8: each term's postings as int32 arrays, one array
  expression per term, ``np.argpartition`` for the top k.
"""

import heapq
from dataclasses import dataclass, field
from operator import itemgetter
from pathlib import Path
from typing import Final, Literal

import numpy as np

from findex.boolean import EngineName
from findex.model import Index
from findex.query import Node, parse
from findex.scoring import BM25, ArrayScorer, Scorer, TfIdf
from findex.timing import timed

type ScorerName = Literal["bm25", "tfidf"]
SCORERS: Final[dict[ScorerName, Scorer]] = {"bm25": BM25(), "tfidf": TfIdf()}
DEFAULT_SCORER: Final[Scorer] = SCORERS["bm25"]

type Ranked = list[tuple[int, float]]  # (doc_id, score), best first


@dataclass(frozen=True, slots=True, order=True)
class SearchResult:
    """One ranked hit. Ordered by score, then doc_id, so ``sorted(results)`` works
    (title and path don't take part in comparisons)."""

    score: float
    doc_id: int
    title: str = field(compare=False)
    path: str = field(compare=False)
    url: str | None = field(default=None, compare=False)

    def __str__(self) -> str:
        return f"{self.score:7.3f}  {self.title}  ({Path(self.path).name})"


def as_node(query: str | Node) -> Node:
    return query if isinstance(query, Node) else parse(query)


def rank_python(
    index: Index, node: Node, scorer: Scorer, k: int, engine: EngineName = "merge"
) -> Ranked:
    """Lab 3: every posting of every query term, one Python call each."""
    scores = dict.fromkeys(index.match(node, engine), 0.0)
    for term in dict.fromkeys(node.terms()):  # each term once, in query order
        for posting in index.get(term, ()):
            if posting.doc_id in scores:
                scores[posting.doc_id] += scorer(term, posting, index)
    # O(n log k): a heap of the k best so far, not a sort of all n matches.
    return heapq.nlargest(k, scores.items(), key=itemgetter(1))


def rank_numpy(
    index: Index, node: Node, scorer: ArrayScorer, k: int, engine: EngineName = "merge"
) -> Ranked:
    """Lab 8: the same ranking, with no Python loop over postings.

    Ties are broken the way ``rank_python`` breaks them (the lower doc_id first),
    so the two agree exactly, at the top-k boundary too.
    """
    matched = np.asarray(index.match(node, engine), dtype=np.int64)  # sorted
    if matched.size == 0 or k <= 0:
        return []
    scores = np.zeros(index.doc_length_array.size, dtype=np.float64)
    for term in dict.fromkeys(node.terms()):
        arrays = index.postings_arrays(term)
        if arrays is not None:
            doc_ids, tfs = arrays
            # A term's doc_ids are distinct, so plain fancy-index += is right
            # (np.add.at would only be needed for repeated indexes).
            scores[doc_ids] += scorer.score_arrays(term, doc_ids, tfs, index)
    candidate = scores[matched]  # only documents the Boolean query accepted
    k = min(k, matched.size)
    if k < matched.size:
        # O(n): partition so the k largest come first, in no particular order.
        top = np.argpartition(-candidate, k - 1)[:k]
        threshold = candidate[top].min()
        above = np.flatnonzero(candidate > threshold)
        tied = np.flatnonzero(candidate == threshold)[: k - above.size]
        chosen = np.concatenate((above, tied))
    else:
        chosen = np.arange(matched.size)
    # Best score first, then the lower doc_id: what heapq.nlargest does.
    order = chosen[np.lexsort((matched[chosen], -candidate[chosen]))]
    return [(int(matched[i]), float(candidate[i])) for i in order]


@timed
def search(
    index: Index,
    query: str | Node,
    scorer: Scorer = DEFAULT_SCORER,
    k: int = 10,
    engine: EngineName = "merge",
) -> list[SearchResult]:
    """The ``k`` best documents matching ``query``, best first.

    The Boolean query decides which documents qualify; each query term then adds
    its score to every qualifying document it occurs in.
    """
    node = as_node(query)
    if isinstance(scorer, ArrayScorer):
        best = rank_numpy(index, node, scorer, k, engine)
    else:
        best = rank_python(index, node, scorer, k, engine)
    return [
        SearchResult(score, doc_id, meta.title, meta.path, meta.url)
        for doc_id, score in best
        for meta in (index.doc_meta[doc_id],)
    ]
