import logging
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import build_index
from findex.model import ArrayPostings, Index, Posting
from findex.query import parse

TEXTS = ["the cat the cat the", "a dog", "cat and dog"]


def build(layout="slots", positions=False):
    docs = (Document(i, Path(f"d{i}.txt"), t) for i, t in enumerate(TEXTS))
    return build_index(docs, layout, positions)


@pytest.fixture
def index():
    return build()


def test_index_speaks_the_mapping_protocol(index):
    assert isinstance(index, Mapping)
    assert len(index) == 5
    assert "cat" in index
    assert "unicorn" not in index
    assert index["cat"] == [Posting(0, 2), Posting(2, 1)]
    with pytest.raises(KeyError):
        index["unicorn"]
    assert list(index) == ["the", "cat", "a", "dog", "and"]
    assert index.get("unicorn") is None
    assert dict(index.items())["dog"] == [Posting(1, 1), Posting(2, 1)]


def test_repr(index):
    assert repr(index) == "Index(terms=5, docs=3, layout='slots', positions=False)"


def test_document_statistics(index):
    assert index.num_docs == 3
    assert index.doc_length(0) == 5
    assert index.df("cat") == 2
    assert index.df("unicorn") == 0
    assert index.avg_doc_length == pytest.approx(10 / 3)


def test_avg_doc_length_is_computed_once(index):
    assert "avg_doc_length" not in vars(index)
    first = index.avg_doc_length
    index.doc_lengths[0] = 1000  # the cached value no longer looks at this

    assert "avg_doc_length" in vars(index)
    assert index.avg_doc_length == first


def test_array_postings_read_as_a_sequence_of_postings():
    plist = build("array", positions=True)["cat"]

    assert isinstance(plist, ArrayPostings)
    assert isinstance(plist, Sequence)
    assert list(plist) == [Posting(0, 2, (1, 3)), Posting(2, 1, (0,))]
    assert plist[1] == plist[-1] == Posting(2, 1, (0,))
    assert plist[:1] == [Posting(0, 2, (1, 3))]
    with pytest.raises(IndexError):
        plist[2]


def test_closed_index_refuses_lookups(index):
    index.close()
    index.close()  # closing twice is fine

    assert repr(index) == "Index(closed)"
    with pytest.raises(ValueError, match="closed"):
        index["cat"]
    with pytest.raises(ValueError, match="closed"):
        len(index)


def test_match_is_cached_per_index(index, caplog):
    tree = parse("cat OR dog")

    with caplog.at_level(logging.DEBUG, logger="findex"):
        first = index.match(tree)
        again = index.match(parse("cat  OR  dog"))  # an equal tree is the same key

    assert first == again == (0, 1, 2)
    messages = [r.getMessage() for r in caplog.records if r.name == "findex.model"]
    assert messages == [
        "match (cat OR dog) [merge]: 3 docs, computed",
        "match (cat OR dog) [merge]: 3 docs, cache hit",
    ]
    assert build().match(tree) == first  # another index: its own cache


def test_pickle_keeps_data_and_starts_a_fresh_cache(index):
    index.match(parse("cat"))
    copy = pickle.loads(pickle.dumps(index))

    assert isinstance(copy, Index)
    assert dict(copy.items()) == dict(index.items())
    assert copy._cached_match.cache_info().currsize == 0
