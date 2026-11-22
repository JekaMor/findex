"""Save and load an Index: pickle, or a binary file of uint32 arrays.

pickle is for index files you built yourself, nothing else. ``pickle.load``
does not just read data: a pickle is a small program that tells the loader
which callables to import and call to rebuild objects (``__reduce__``). A
crafted file can name ``os.system`` with any command, and it runs the moment
the file is loaded, before you get to look at the result. So never load a
pickle downloaded or received from someone else.

The binary format holds only a JSON header and raw numbers, so loading it can
never run code::

    MAGIC (8 bytes) | header length (uint64 LE) | JSON header | postings data

The header lists the documents and, for each term, ``[term, offset, n, m]``:
at ``offset`` in the data there are ``n`` doc_ids, ``n`` tfs and ``m``
positions, each a little-endian uint32.
"""

import json
import pickle
import sys
from array import array
from pathlib import Path
from typing import Literal

from findex.model import ArrayPostings, DocMeta, Index, PostingList, to_arrays

Format = Literal["binary", "pickle"]
FORMATS: tuple[Format, ...] = ("binary", "pickle")
MAGIC = b"FINDEX\x00\x01"
_PICKLE_SUFFIXES = {".pkl", ".pickle"}

assert array("I").itemsize == 4, "the binary format assumes 4-byte unsigned ints"


def format_for(path: Path) -> Format:
    return "pickle" if path.suffix in _PICKLE_SUFFIXES else "binary"


def save(index: Index, path: Path, fmt: Format | None = None) -> Format:
    """Write ``index`` to ``path``; returns the format used."""
    fmt = fmt or format_for(path)
    if fmt == "pickle":
        with path.open("wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        _save_binary(index, path)
    return fmt


def load(path: Path) -> Index:
    """Read an index written by ``save``; the format is detected from the file."""
    with path.open("rb") as f:
        head = f.read(len(MAGIC))
    if head == MAGIC:
        return _load_binary(path)
    if not head.startswith(b"\x80"):
        raise ValueError(f"{path} is not a findex index")
    # Trusted files only: see the module docstring.
    with path.open("rb") as f:
        index = pickle.load(f)
    if not isinstance(index, Index):
        raise TypeError(f"{path} does not contain an Index")
    return index


def _counts(plist: PostingList) -> tuple[int, int]:
    """(postings, positions) in a postings list."""
    if isinstance(plist, ArrayPostings):
        return len(plist.doc_ids), len(plist.positions)
    return len(plist), sum(len(p.positions) for p in plist)


def _little_endian(a: array[int]) -> bytes:
    if sys.byteorder == "big":
        a = array("I", a)
        a.byteswap()
    return a.tobytes()


def _from_little_endian(data: memoryview) -> array[int]:
    a = array("I")
    a.frombytes(data)
    if sys.byteorder == "big":
        a.byteswap()
    return a


def _save_binary(index: Index, path: Path) -> None:
    # Offsets come from the lengths alone, so the header is written first and
    # the postings are then streamed out one term at a time.
    terms = []
    offset = 0
    for term, plist in index.postings.items():
        n, m = _counts(plist)
        terms.append([term, offset, n, m])
        offset += (2 * n + m) * 4
    header = json.dumps(
        {
            "positions": index.positions,
            "doc_lengths": list(index.doc_lengths.items()),
            "doc_meta": [[i, m.path, m.title] for i, m in index.doc_meta.items()],
            "terms": terms,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(len(header).to_bytes(8, "little"))
        f.write(header)
        for plist in index.postings.values():
            arrays = to_arrays(plist)
            f.write(_little_endian(arrays.doc_ids))
            f.write(_little_endian(arrays.tfs))
            f.write(_little_endian(arrays.positions))


def _load_binary(path: Path) -> Index:
    data = memoryview(path.read_bytes())
    start = len(MAGIC) + 8
    header_len = int.from_bytes(data[len(MAGIC) : start], "little")
    header = json.loads(bytes(data[start : start + header_len]))
    body = data[start + header_len :]

    postings: dict[str, PostingList] = {}
    for term, offset, n, m in header["terms"]:
        tfs_at = offset + 4 * n
        positions_at = tfs_at + 4 * n
        postings[term] = ArrayPostings(
            _from_little_endian(body[offset:tfs_at]),
            _from_little_endian(body[tfs_at:positions_at]),
            _from_little_endian(body[positions_at : positions_at + 4 * m]),
        )
    return Index(
        postings,
        {doc_id: length for doc_id, length in header["doc_lengths"]},
        {doc_id: DocMeta(p, title) for doc_id, p, title in header["doc_meta"]},
        layout="array",
        positions=header["positions"],
    )
