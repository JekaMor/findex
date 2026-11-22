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

``open_index`` memory-maps a binary file instead of reading it: only the header
is parsed up front, each term's postings are read when asked for, and the file
stays open until the ``with`` block ends.
"""

import json
import mmap
import pickle
import sys
from array import array
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Final, Literal, TypedDict, cast

from findex.model import ArrayPostings, DocMeta, Index, PostingList, to_arrays
from findex.timing import timed

type Buffer = bytes | memoryview | mmap.mmap

Format = Literal["binary", "pickle"]
FORMATS: Final[tuple[Format, ...]] = ("binary", "pickle")
MAGIC = b"FINDEX\x00\x01"
_PICKLE_SUFFIXES: Final = frozenset({".pkl", ".pickle"})


class Header(TypedDict):
    """The JSON header of a binary index file."""

    positions: bool
    doc_lengths: list[tuple[int, int]]  # [doc_id, length]
    # [doc_id, path, title] or, for a crawled page, [doc_id, path, title, url]
    doc_meta: list[tuple[int, str, str] | tuple[int, str, str, str]]
    terms: list[tuple[str, int, int, int]]  # [term, offset, postings, positions]


assert array("I").itemsize == 4, "the binary format assumes 4-byte unsigned ints"


class IndexFormatError(ValueError):
    """The file exists but is not an index written by ``save``."""


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


@timed
def load(path: Path, *, lazy: bool = False) -> Index:
    """Read an index written by ``save``; the format is detected from the file.

    ``lazy=True`` memory-maps a binary file and reads postings on demand; the
    index must then be closed (``open_index`` does it). Pickle always loads whole.
    """
    with path.open("rb") as f:
        head = f.read(len(MAGIC))
    if head == MAGIC:
        return _open_mapped(path) if lazy else _load_binary(path)
    if not head.startswith(b"\x80"):
        raise IndexFormatError(f"{path} is not a findex index file")
    # Trusted files only: see the module docstring.
    with path.open("rb") as f:
        index = pickle.load(f)
    if not isinstance(index, Index):
        raise IndexFormatError(f"{path} does not contain an Index")
    return index


@contextmanager
def open_index(path: Path) -> Generator[Index]:
    """``with open_index(path) as index:`` -- the index is closed when the block
    ends, whether it finishes, returns or raises."""
    index = load(path, lazy=True)
    try:
        yield index
    finally:
        index.close()


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


def _from_little_endian(data: bytes | memoryview) -> array[int]:
    a = array("I")
    a.frombytes(data)
    if sys.byteorder == "big":
        a.byteswap()
    return a


def _meta_row(
    doc_id: int, meta: DocMeta
) -> tuple[int, str, str] | tuple[int, str, str, str]:
    # Files without URLs keep the three-field rows, so their bytes don't change.
    if meta.url is None:
        return (doc_id, meta.path, meta.title)
    return (doc_id, meta.path, meta.title, meta.url)


def _save_binary(index: Index, path: Path) -> None:
    # Offsets come from the lengths alone, so the header is written first and
    # the postings are then streamed out one term at a time.
    terms: list[tuple[str, int, int, int]] = []
    offset = 0
    for term, plist in index.postings.items():
        n, m = _counts(plist)
        terms.append((term, offset, n, m))
        offset += (2 * n + m) * 4
    header: Header = {
        "positions": index.positions,
        "doc_lengths": list(index.doc_lengths.items()),
        "doc_meta": [_meta_row(i, m) for i, m in index.doc_meta.items()],
        "terms": terms,
    }
    encoded = json.dumps(header, ensure_ascii=False).encode("utf-8")
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(len(encoded).to_bytes(8, "little"))
        f.write(encoded)
        for plist in index.postings.values():
            arrays = to_arrays(plist)
            f.write(_little_endian(arrays.doc_ids))
            f.write(_little_endian(arrays.tfs))
            f.write(_little_endian(arrays.positions))


def _read_header(buf: Buffer) -> tuple[Header, int]:
    """(header, where the postings data starts)."""
    start = len(MAGIC) + 8
    header_len = int.from_bytes(buf[len(MAGIC) : start], "little")
    # json.loads returns Any. We wrote this JSON ourselves in _save_binary from a
    # Header, so cast says so (JSON arrays come back as lists; the code only
    # unpacks them, which works the same for lists and tuples).
    header = cast(Header, json.loads(bytes(buf[start : start + header_len])))
    return header, start + header_len


def _postings_at(buf: Buffer, at: int, n: int, m: int) -> ArrayPostings:
    tfs_at = at + 4 * n
    positions_at = tfs_at + 4 * n
    return ArrayPostings(
        _from_little_endian(buf[at:tfs_at]),
        _from_little_endian(buf[tfs_at:positions_at]),
        _from_little_endian(buf[positions_at : positions_at + 4 * m]),
    )


def _index_from(
    header: Header,
    postings: Mapping[str, PostingList],
    closer: Callable[[], None] | None = None,
) -> Index:
    return Index(
        postings,
        {doc_id: length for doc_id, length in header["doc_lengths"]},
        {row[0]: DocMeta(*row[1:]) for row in header["doc_meta"]},
        layout="array",
        positions=header["positions"],
        closer=closer,
    )


def _load_binary(path: Path) -> Index:
    data = memoryview(path.read_bytes())
    header, base = _read_header(data)
    postings: dict[str, PostingList] = {
        term: _postings_at(data, base + offset, n, m)
        for term, offset, n, m in header["terms"]
    }
    return _index_from(header, postings)


class MappedPostings(Mapping[str, ArrayPostings]):
    """term -> postings, read from a memory-mapped file only when asked for."""

    def __init__(
        self, buf: mmap.mmap, base: int, terms: list[tuple[str, int, int, int]]
    ) -> None:
        self._buf = buf
        self._base = base
        self._where = {term: (offset, n, m) for term, offset, n, m in terms}

    def __getitem__(self, term: str) -> ArrayPostings:
        offset, n, m = self._where[term]
        return _postings_at(self._buf, self._base + offset, n, m)

    def __len__(self) -> int:
        return len(self._where)

    def __iter__(self) -> Iterator[str]:
        return iter(self._where)

    def __contains__(self, term: object) -> bool:
        return term in self._where

    def df(self, term: str) -> int:
        """Document frequency straight from the header, without reading postings."""
        where = self._where.get(term)
        return 0 if where is None else where[1]


def _open_mapped(path: Path) -> Index:
    f = path.open("rb")
    try:
        buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    except BaseException:
        f.close()
        raise

    def close() -> None:
        buf.close()
        f.close()

    try:
        header, base = _read_header(buf)
        postings = MappedPostings(buf, base, header["terms"])
        return _index_from(header, postings, closer=close)
    except BaseException:
        close()
        raise
