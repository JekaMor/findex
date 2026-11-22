"""Properties checked by Hypothesis on generated inputs, not on examples."""

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from findex.boolean import difference, intersect, union
from findex.corpus import Document, read_documents
from findex.index import build_index
from findex.model import LAYOUTS, Index, Layout, to_arrays
from findex.parallel import build_partial, chunks, merge
from findex.query import And, Node, Not, Or, Phrase, Term, parse
from findex.store import load, save
from findex.tokenize import tokenize

# --- strategies --------------------------------------------------------------------

sorted_ids = st.lists(st.integers(0, 200), unique=True).map(sorted)
words = st.sampled_from(
    ["whale", "sea", "ship", "кіт", "пес", "don't", "straße", "café", "42", "the"]
)
texts = st.lists(st.one_of(words, st.text(max_size=8)), max_size=30).map(" ".join)
corpora = st.lists(texts, min_size=1, max_size=6)


def build(corpus: list[str], layout: Layout, positions: bool) -> Index:
    docs = (Document(i, Path(f"d{i}.txt"), t) for i, t in enumerate(corpus))
    return build_index(docs, layout, positions)


def plain(index: Index) -> object:
    postings = {
        term: (list(a.doc_ids), list(a.tfs), list(a.positions))
        for term, a in ((t, to_arrays(p)) for t, p in index.items())
    }
    return postings, index.doc_lengths, index.doc_meta, index.positions


# --- merges: the fast two-pointer walk agrees with the obvious set version -------


@given(sorted_ids, sorted_ids)
def test_merges_agree_with_sets(a: list[int], b: list[int]) -> None:
    assert intersect(a, b) == sorted(set(a) & set(b))
    assert union(a, b) == sorted(set(a) | set(b))
    assert difference(a, b) == sorted(set(a) - set(b))


# --- index invariants ---------------------------------------------------------------


@pytest.mark.slow
@given(corpora, st.sampled_from(LAYOUTS), st.booleans())
def test_postings_are_sorted_and_consistent(
    corpus: list[str], layout: Layout, positions: bool
) -> None:
    index = build(corpus, layout, positions)

    for term, plist in index.items():
        ids = [p.doc_id for p in plist]
        assert ids == sorted(set(ids)), term  # sorted, no duplicates
        assert all(p.tf > 0 for p in plist)
        if positions:
            assert all(len(p.positions) == p.tf for p in plist)
    for doc_id, length in index.doc_lengths.items():
        assert length == sum(
            p.tf for plist in index.values() for p in plist if p.doc_id == doc_id
        )


@settings(max_examples=40, deadline=None)
@given(
    corpora, st.sampled_from(LAYOUTS), st.booleans(), st.sampled_from([".bin", ".pkl"])
)
def test_save_then_load_gives_the_same_index(
    corpus: list[str], layout: Layout, positions: bool, suffix: str
) -> None:
    index = build(corpus, layout, positions)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"index{suffix}"
        save(index, path)
        assert plain(load(path)) == plain(index)


# --- tokenizer ------------------------------------------------------------------------


@given(st.text())
def test_tokenize_is_idempotent_and_never_yields_empty(text: str) -> None:
    tokens = list(tokenize(text))

    assert all(tokens)
    assert list(tokenize(" ".join(tokens))) == tokens


# --- query parser: a printed tree parses back to the same tree --------------------


def trees() -> st.SearchStrategy[Node]:
    term = words.flatmap(lambda w: st.just(Term(next(tokenize(w)))))
    phrase = st.lists(words, min_size=2, max_size=3).map(
        lambda ws: Phrase(tuple(next(tokenize(w)) for w in ws))
    )
    return st.recursive(
        st.one_of(term, phrase),
        lambda kids: st.one_of(
            st.tuples(kids, kids).map(lambda lr: And(*lr)),
            st.tuples(kids, kids).map(lambda lr: Or(*lr)),
            kids.map(Not),
        ),
        max_leaves=8,
    )


@given(trees())
def test_printed_query_parses_back_to_the_same_tree(tree: Node) -> None:
    assert parse(str(tree)) == tree


@settings(deadline=None)
@given(trees(), corpora)
def test_merge_and_set_engines_agree(tree: Node, corpus: list[str]) -> None:
    index = build(corpus, "slots", positions=True)

    assert tree.evaluate(index, "merge") == tree.evaluate(index, "set")


@settings(deadline=None, max_examples=40)
@given(corpora, st.integers(1, 7))
def test_any_chunking_merges_to_the_serial_index(corpus: list[str], count: int) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = []
        for i, text in enumerate(corpus):
            path = root / f"d{i}.txt"
            path.write_text(text, encoding="utf-8")
            paths.append(path)
        serial = build_index(read_documents(paths), "array", positions=True)

        merged = merge(build_partial(c) for c in chunks(paths, count))

        assert plain(merged) == plain(serial)
