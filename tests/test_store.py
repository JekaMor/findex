from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import build_index
from findex.model import LAYOUTS, to_arrays
from findex.store import MAGIC, MappedPostings, format_for, load, open_index, save

TEXTS = ["Title: One\nthe cat the", "dog п'ять cat", "the end"]


def build(layout="slots", positions=False):
    docs = (Document(i, Path(f"d{i}.txt"), t) for i, t in enumerate(TEXTS))
    return build_index(docs, layout, positions)


def as_plain_data(index):
    """Everything an index holds, independent of the postings layout."""
    postings = {}
    for term, plist in index.postings.items():
        arrays = to_arrays(plist)
        postings[term] = (
            list(arrays.doc_ids),
            list(arrays.tfs),
            list(arrays.positions),
        )
    return postings, index.doc_lengths, index.doc_meta, index.positions


@pytest.mark.parametrize("positions", [False, True])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("name", ["index.pkl", "index.bin"])
def test_round_trip(tmp_path, name, layout, positions):
    index = build(layout, positions)
    path = tmp_path / name

    save(index, path)
    loaded = load(path)

    assert as_plain_data(loaded) == as_plain_data(index)
    expected_layout = layout if name.endswith(".pkl") else "array"
    assert loaded.layout == expected_layout


def test_format_is_chosen_by_suffix_and_detected_on_load(tmp_path):
    assert format_for(Path("x.pkl")) == "pickle"
    assert format_for(Path("x.bin")) == "binary"

    path = tmp_path / "forced.bin"
    assert save(build(), path, "pickle") == "pickle"
    assert not path.read_bytes().startswith(MAGIC)
    assert load(path).layout == "slots"


def test_binary_file_starts_with_magic(tmp_path):
    path = tmp_path / "index.bin"
    save(build(), path)

    assert path.read_bytes().startswith(MAGIC)


def test_unknown_file_is_rejected(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="not a findex index"):
        load(path)


def test_open_index_maps_binary_file_and_reads_lazily(tmp_path):
    index = build(positions=True)
    path = tmp_path / "index.bin"
    save(index, path)

    with open_index(path) as opened:
        assert isinstance(opened.postings, MappedPostings)
        assert as_plain_data(opened) == as_plain_data(index)
        assert opened.df("cat") == 2
        assert "cat" in opened
        assert "unicorn" not in opened
    assert opened.closed


def test_open_index_closes_when_the_block_raises(tmp_path):
    path = tmp_path / "index.bin"
    save(build(), path)

    with pytest.raises(RuntimeError), open_index(path) as opened:
        raise RuntimeError("boom")

    assert opened.closed
    with pytest.raises(ValueError, match="closed"):
        opened["cat"]
    path.unlink()  # on Windows this fails while the file is still mapped


def test_open_index_also_opens_pickle(tmp_path):
    path = tmp_path / "index.pkl"
    save(build(), path)

    with open_index(path) as opened:
        assert opened.layout == "slots"
        assert opened.df("the") == 2
    assert opened.closed
