"""The three ranking sanity checks, on tiny corpora where the answer is known."""

import math
from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import build_index
from findex.model import Posting
from findex.scoring import BM25, Scorer, TfIdf
from findex.search import search

FILLER = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do"


def make_index(*texts):
    return build_index(Document(i, Path(f"d{i}.txt"), t) for i, t in enumerate(texts))


def ranking(index, query, scorer):
    return [r.doc_id for r in search(index, query, scorer)]


@pytest.mark.parametrize("scorer", [TfIdf(), BM25()])
def test_rare_term_outranks_common_term(scorer):
    # "common" is in every doc but one, "rare" in a single one.
    index = make_index(
        f"common {FILLER}",
        f"rare {FILLER}",
        f"common {FILLER}",
        f"common {FILLER}",
        FILLER,
    )

    assert ranking(index, "common OR rare", scorer)[0] == 1


def test_bm25_saturates_term_frequency():
    index = make_index(FILLER, "x " + FILLER, FILLER)
    bm25, tfidf = BM25(), TfIdf()

    def gain(scorer, tf):
        """What the tf-th occurrence adds, the document length kept the same."""

        def score(n):
            return scorer("x", Posting(1, n), index)

        return score(tf) - score(tf - 1)

    assert gain(bm25, 20) < 0.05 * gain(bm25, 2)  # the 20th adds under 5% of the 2nd
    assert gain(tfidf, 20) == pytest.approx(gain(tfidf, 2))  # TF-IDF never saturates
    # and BM25 has a ceiling: idf * (k1 + 1)
    assert bm25("x", Posting(1, 10**6), index) < bm25.idf("x", index) * (bm25.k1 + 1)


def test_bm25_prefers_short_document_with_one_match():
    long_doc = "x " + " ".join([FILLER] * 50)
    index = make_index(long_doc, f"x {FILLER}", FILLER)

    assert ranking(index, "x", BM25()) == [1, 0]
    tied = search(index, "x", TfIdf())
    assert tied[0].score == tied[1].score  # TF-IDF ignores length
    assert ranking(index, "x", BM25(b=0)) == [0, 1]  # b = 0: length ignored, a tie


def test_bm25_formula_by_hand():
    index = make_index("x y", "y y y y", "z")
    n, df, avgdl = 3, 1, 7 / 3
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    expected = idf * 1 * 2.5 / (1 + 1.5 * (0.25 + 0.75 * 2 / avgdl))

    assert BM25()("x", Posting(0, 1), index) == pytest.approx(expected)
    assert TfIdf()("x", Posting(0, 1), index) == pytest.approx(math.log(3))


def test_any_callable_is_a_scorer():
    def by_tf(term: str, posting: Posting, index) -> float:
        return posting.tf

    scorer: Scorer = by_tf
    index = make_index("x", "x x x", "x x")

    assert ranking(index, "x", scorer) == [1, 2, 0]


def test_scorers_have_useful_reprs():
    assert repr(BM25()) == "BM25(k1=1.5, b=0.75)"
    assert repr(TfIdf()) == "TfIdf()"
