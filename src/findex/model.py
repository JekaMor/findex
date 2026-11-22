"""Index records: postings, document metadata, and the index itself.

These live apart from ``index.py`` so that ``python -m findex.index`` (which runs
that file as ``__main__``) and ``findex.store`` see the same classes, and a
pickled index refers to ``findex.model.Posting``, not ``__main__.Posting``.
"""

from array import array
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

Layout = Literal["slots", "plain", "array"]
LAYOUTS: tuple[Layout, ...] = ("slots", "plain", "array")


@dataclass(frozen=True, slots=True)
class Posting:
    """One document in a term's postings list."""

    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True)
class PlainPosting:
    """Posting without ``slots``: every instance carries a ``__dict__``.

    Exists only for the memory study.
    """

    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


def _uint32s() -> array[int]:
    return array("I")


@dataclass(slots=True)
class ArrayPostings:
    """A postings list as parallel arrays of 4-byte unsigned ints, no objects.

    ``positions`` holds every posting's positions end to end; posting ``i`` owns
    the next ``tfs[i]`` of them.
    """

    doc_ids: array[int] = field(default_factory=_uint32s)
    tfs: array[int] = field(default_factory=_uint32s)
    positions: array[int] = field(default_factory=_uint32s)

    def __len__(self) -> int:
        return len(self.doc_ids)

    def append(self, doc_id: int, tf: int, positions: Sequence[int] = ()) -> None:
        self.doc_ids.append(doc_id)
        self.tfs.append(tf)
        self.positions.extend(positions)


type PostingList = list[Posting] | list[PlainPosting] | ArrayPostings


def to_arrays(plist: PostingList) -> ArrayPostings:
    """Any postings list as ArrayPostings (the same object if it already is one)."""
    if isinstance(plist, ArrayPostings):
        return plist
    out = ArrayPostings()
    for p in plist:
        out.append(p.doc_id, p.tf, p.positions)
    return out


@dataclass(frozen=True, slots=True)
class DocMeta:
    """What a search result shows for a document."""

    path: str
    title: str


@dataclass(slots=True)
class Index:
    postings: dict[str, PostingList]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]
    layout: Layout = "slots"
    positions: bool = False

    def doc_ids(self, term: str) -> Sequence[int]:
        """Sorted doc_ids of the documents containing ``term``."""
        plist = self.postings.get(term)
        if plist is None:
            return ()
        if isinstance(plist, ArrayPostings):
            return plist.doc_ids
        return [p.doc_id for p in plist]

    def all_doc_ids(self) -> list[int]:
        # doc_meta is filled in doc_id order and a dict keeps insertion order.
        return list(self.doc_meta)

    def posting_count(self) -> int:
        return sum(len(plist) for plist in self.postings.values())
