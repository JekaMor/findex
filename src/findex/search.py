"""Ranked search: the query decides which documents match, a scorer orders them."""

import heapq
from dataclasses import dataclass, field
from operator import itemgetter
from pathlib import Path
from typing import Final, Literal

from findex.boolean import EngineName
from findex.model import Index
from findex.query import Node, parse
from findex.scoring import BM25, Scorer, TfIdf
from findex.timing import timed

type ScorerName = Literal["bm25", "tfidf"]
SCORERS: Final[dict[ScorerName, Scorer]] = {"bm25": BM25(), "tfidf": TfIdf()}
DEFAULT_SCORER: Final[Scorer] = SCORERS["bm25"]


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
    engine: EngineName = "merge",
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
