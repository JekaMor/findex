import dataclasses
from pathlib import Path

import pytest

from findex.corpus import Document, iter_documents
from findex.index import build_index, main, title_of
from findex.model import LAYOUTS, ArrayPostings, DocMeta, PlainPosting, Posting


def docs(*texts):
    return (Document(i, Path(f"d{i}.txt"), text) for i, text in enumerate(texts))


def test_posting_and_docmeta_are_frozen_slotted_and_hashable():
    for record in (Posting(1, 2), DocMeta("a.txt", "A")):
        assert not hasattr(record, "__dict__")
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.title = "x"  # type: ignore[misc]
    assert len({Posting(1, 2), Posting(1, 2), Posting(1, 3)}) == 2
    assert len({DocMeta("a.txt", "A"), DocMeta("a.txt", "A")}) == 1


def test_plain_posting_has_a_dict():
    assert hasattr(PlainPosting(1, 2), "__dict__")


def test_postings_are_sorted_with_term_frequencies():
    index = build_index(docs("the cat the cat the", "a dog", "Cat and dog"))

    assert index.postings["cat"] == [Posting(0, 2), Posting(2, 1)]
    assert index.postings["the"] == [Posting(0, 3)]
    assert index.doc_ids("dog") == [1, 2]
    assert index.doc_ids("missing") == ()
    assert "missing" not in index.postings
    assert index.doc_lengths == {0: 5, 1: 2, 2: 3}


@pytest.mark.parametrize("layout", LAYOUTS)
def test_every_layout_gives_the_same_doc_ids(layout):
    index = build_index(docs("a b a", "b c", "a c c"), layout)

    assert index.layout == layout
    assert [list(index.doc_ids(t)) for t in "abc"] == [[0, 2], [0, 1], [1, 2]]
    assert index.posting_count() == 6


def test_array_layout_stores_no_posting_objects():
    plist = build_index(docs("x y x"), "array").postings["x"]

    assert isinstance(plist, ArrayPostings)
    assert plist.doc_ids.typecode == "I"
    assert list(plist.tfs) == [2]


@pytest.mark.parametrize("layout", ["slots", "array"])
def test_positions_are_token_offsets(layout):
    index = build_index(docs("to be or not to be"), layout, positions=True)

    plist = index.postings["be"]
    if layout == "array":
        assert list(plist.positions) == [1, 5]
    else:
        assert plist == [Posting(0, 2, (1, 5))]


def test_documents_out_of_order_are_rejected():
    out_of_order = [Document(1, Path("b"), "x"), Document(0, Path("a"), "y")]

    with pytest.raises(ValueError, match="unsorted"):
        build_index(out_of_order)


def test_title_comes_from_gutenberg_header_or_file_name():
    header = "The Project Gutenberg eBook\n\nTitle: Alice's Adventures\n\nAuthor: X\n"

    assert title_of(Document(0, Path("pg11.txt"), header)) == "Alice's Adventures"
    assert title_of(Document(0, Path("notes.txt"), "no header here")) == "notes"


def test_build_consumes_the_lazy_stream(tmp_path):
    (tmp_path / "a.txt").write_text("Title: First\nalpha beta", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")

    index = build_index(iter_documents(tmp_path))

    assert len(index.doc_meta) == 2
    assert len(index.doc_ids("beta")) == 2
    assert "First" in {m.title for m in index.doc_meta.values()}


def test_main_builds_and_saves(tmp_path, capsys):
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    out = tmp_path / "index.bin"

    main([str(tmp_path), "--out", str(out)])

    assert out.exists()
    printed = capsys.readouterr().out
    for needle in (
        "documents:    1",
        "terms:        2",
        "binary",
        "build",
        "peak memory",
    ):
        assert needle in printed
