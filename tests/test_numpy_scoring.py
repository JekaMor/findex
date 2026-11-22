"""The NumPy scorer must rank exactly like Lab 3's Python loop."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from findex.corpus import Document
from findex.index import build_index
from findex.model import LAYOUTS, Index, Layout
from findex.query import And, Node, Not, Or, Phrase, Term, parse
from findex.scoring import BM25, ArrayScorer, TfIdf
from findex.search import SCORERS, rank_numpy, rank_python, search
from findex.store import open_index, save

QUERIES = [
    "whale",
    "the",
    "sea OR ship OR captain",
    "the a",
    "кіт OR пес",
    "NOT whale",
    '"dead man"',
    "treasure OR whale NOT sea",
    "unicorn",
]


def build(*texts: str, layout: Layout = "slots", positions: bool = True) -> Index:
    docs = (Document(i, Path(f"d{i}.txt"), t) for i, t in enumerate(texts))
    return build_index(docs, layout, positions)


@pytest.mark.parametrize("scorer", ["bm25", "tfidf"])
@pytest.mark.parametrize("query", QUERIES)
@pytest.mark.parametrize("k", [1, 2, 10])
def test_same_ranking_and_scores_as_lab3(
    corpus_index: Index, scorer: str, query: str, k: int
) -> None:
    node = parse(query)
    chosen = SCORERS[scorer]  # type: ignore[index]  # parametrized with valid keys
    assert isinstance(chosen, ArrayScorer)

    assert rank_numpy(corpus_index, node, chosen, k) == rank_python(
        corpus_index, node, chosen, k
    )


@pytest.mark.parametrize("layout", LAYOUTS)
def test_every_postings_layout_gives_the_same_arrays(layout: Layout) -> None:
    index = build("a b a", "b c", "a c c", layout=layout)

    arrays = index.postings_arrays("a")

    assert arrays is not None
    doc_ids, tfs = arrays
    assert doc_ids.dtype == np.int32 and tfs.dtype == np.int32
    assert doc_ids.tolist() == [0, 2] and tfs.tolist() == [2, 1]
    assert index.postings_arrays("zzz") is None
    assert index.doc_length_array.tolist() == [3, 2, 3]


def test_mapped_index_gives_zero_copy_views(index_file: Path) -> None:
    with open_index(index_file) as index:
        arrays = index.postings_arrays("the")
        assert arrays is not None
        doc_ids, tfs = arrays
        assert not doc_ids.flags.owndata and not tfs.flags.owndata  # views, no copy
        assert (tfs > 0).all()
        ranked = rank_numpy(index, parse("the OR whale"), BM25(), 3)
        assert ranked == rank_python(index, parse("the OR whale"), BM25(), 3)
        del arrays, doc_ids, tfs
    # Closing with the views gone works; with one alive it must not crash either.
    with open_index(index_file) as index:
        kept = index.postings_arrays("the")
    assert index.closed and kept is not None


def test_ties_at_the_k_boundary_keep_the_lower_doc_id() -> None:
    # Five identical documents: every score is equal.
    index = build(*["same words here"] * 5, "other")

    for k in range(1, 6):
        expected = rank_python(index, parse("same"), BM25(), k)
        assert rank_numpy(index, parse("same"), BM25(), k) == expected
        assert [doc_id for doc_id, _ in expected] == list(range(k))


def test_a_plain_function_scorer_still_works() -> None:
    def by_tf(term: str, posting: object, index: Index) -> float:
        return float(posting.tf)  # type: ignore[attr-defined]  # PostingLike

    index = build("x", "x x x", "x x")

    assert not isinstance(by_tf, ArrayScorer)
    assert [r.doc_id for r in search(index, "x", by_tf)] == [1, 2, 0]


def test_array_scorers_match_per_posting_scores_exactly() -> None:
    index = build("a a b", "a " * 10 + "c", "b c", "a")
    doc_ids, tfs = index.postings_arrays("a")  # type: ignore[misc]  # "a" exists
    for scorer in (BM25(), BM25(k1=1.2, b=0.3), TfIdf()):
        per_posting = [scorer("a", p, index) for p in index["a"]]
        assert scorer.score_arrays("a", doc_ids, tfs, index).tolist() == per_posting


# --- property: any corpus, any query ---------------------------------------------------

WORDS = st.sampled_from(["a", "b", "c", "dd", "ee", "кіт"])
TEXTS = st.lists(WORDS, max_size=12).map(" ".join)


def trees() -> st.SearchStrategy[Node]:
    leaf = st.one_of(
        WORDS.map(Term),
        st.lists(WORDS, min_size=2, max_size=3).map(lambda w: Phrase(tuple(w))),
    )
    return st.recursive(
        leaf,
        lambda kids: st.one_of(
            st.tuples(kids, kids).map(lambda lr: And(*lr)),
            st.tuples(kids, kids).map(lambda lr: Or(*lr)),
            kids.map(Not),
        ),
        max_leaves=5,
    )


@settings(deadline=None, max_examples=150)
@given(
    st.lists(TEXTS, min_size=1, max_size=8),
    trees(),
    st.integers(1, 6),
    st.sampled_from(["bm25", "tfidf"]),
    st.booleans(),
)
def test_numpy_ranking_equals_python_ranking(
    texts: list[str], tree: Node, k: int, scorer: str, mapped: bool
) -> None:
    index = build(*texts, layout="array")
    chosen = SCORERS[scorer]  # type: ignore[index]  # sampled from valid keys
    assert isinstance(chosen, ArrayScorer)
    if not mapped:
        assert rank_numpy(index, tree, chosen, k) == rank_python(index, tree, chosen, k)
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "i.bin"
        save(index, path)
        with open_index(path) as opened:
            assert rank_numpy(opened, tree, chosen, k) == rank_python(
                opened, tree, chosen, k
            )
