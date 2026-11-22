import logging
from pathlib import Path

import pytest

from findex.boolean import ENGINES
from findex.corpus import Document
from findex.index import build_index
from findex.query import Term
from findex.scoring import TfIdf
from findex.search import SearchResult, main, search
from findex.store import save

TEXTS = [
    "Title: Cats\nthe cat sat on the mat with another cat",  # 0
    "the dog ran after the event loop",  # 1
    "cat and dog",  # 2
    "Кіт і пес",  # 3
]


@pytest.fixture(scope="module")
def index():
    return build_index(
        (Document(i, Path(f"d{i}.txt"), t) for i, t in enumerate(TEXTS)), positions=True
    )


def test_results_are_ranked_best_first(index):
    results = search(index, "cat")

    # "cat and dog" (3 tokens) beats the 11-token doc with two cats: BM25's
    # length normalisation at work.
    assert [r.doc_id for r in results] == [2, 0]
    assert results[0].score > results[1].score > 0
    assert results[1].title == "Cats"
    assert all(isinstance(r, SearchResult) for r in results)


def test_results_sort_by_score(index):
    results = search(index, "cat OR dog OR кіт")

    assert sorted(results, reverse=True) == results
    assert min(results) == results[-1]


def test_k_limits_results(index):
    assert len(search(index, "cat OR dog", k=1)) == 1


def test_query_can_be_a_tree(index):
    assert search(index, Term("cat") & ~Term("dog")) == search(index, "cat NOT dog")


def test_only_negative_query_matches_with_zero_score(index):
    results = search(index, "NOT cat")

    assert [r.doc_id for r in results] == [1, 3]
    assert {r.score for r in results} == {0.0}


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_engines_rank_the_same(index, engine):
    assert search(index, "cat OR dog", engine=engine) == search(index, "cat OR dog")


def test_scorer_is_swappable(index):
    # TF-IDF ignores length, so the doc with two cats wins.
    assert [r.doc_id for r in search(index, "cat", TfIdf())] == [0, 2]


def test_search_is_timed(index, caplog):
    with caplog.at_level(logging.INFO, logger="findex.timing"):
        search(index, "cat")

    assert any(r.getMessage().startswith("search took") for r in caplog.records)


def test_main_ranks_snippets_and_caches(index, tmp_path, capsys, monkeypatch, caplog):
    for i, text in enumerate(TEXTS):
        (tmp_path / f"d{i}.txt").write_text(text, encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # doc paths are relative, like data/... in real use
    save(index, tmp_path / "index.bin")

    with caplog.at_level(logging.INFO, logger="findex"):
        main(["index.bin", '"event loop"', "cat", "cat", "a OR (", "-v", "-k", "1"])

    out = capsys.readouterr().out
    for needle in (
        "Index(terms=",
        "BM25(k1=1.5, b=0.75)",
        "parsed:       cat",
        "1.   ",
        "**cat**",
        "**event** **loop**",
        "error:        the query ends too early",
        "peak memory",
    ):
        assert needle in out
    logged = [r.getMessage() for r in caplog.records]
    assert "match cat [merge]: 2 docs, computed" in logged
    assert "match cat [merge]: 2 docs, cache hit" in logged
