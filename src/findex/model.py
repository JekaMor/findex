"""Index records: postings, document metadata, and the index itself.

They live apart from ``index.py`` so that the builder, the store and the CLI all
import the same classes, and a pickled index refers to ``findex.model.Posting``.
"""

import functools
import logging
from array import array
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

from findex.boolean import EngineName
from findex.timing import timed

if TYPE_CHECKING:
    from findex.query import Node

log = logging.getLogger(__name__)

Layout = Literal["slots", "plain", "array"]
LAYOUTS: tuple[Layout, ...] = ("slots", "plain", "array")


class PostingLike(Protocol):
    """What scoring and phrase matching read from a posting.

    ``Posting`` and ``PlainPosting`` both fit without inheriting from it: frozen
    dataclass fields are read-only attributes, which is what the properties ask.
    """

    @property
    def doc_id(self) -> int: ...
    @property
    def tf(self) -> int: ...
    @property
    def positions(self) -> tuple[int, ...]: ...


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
class ArrayPostings(Sequence[Posting]):
    """A postings list as parallel arrays of 4-byte unsigned ints, no objects.

    ``positions`` holds every posting's positions end to end; posting ``i`` owns
    the next ``tfs[i]`` of them. Reading it as a sequence builds ``Posting``
    objects on the fly, so code that walks postings never needs to know the layout.
    """

    doc_ids: array[int] = field(default_factory=_uint32s)
    tfs: array[int] = field(default_factory=_uint32s)
    positions: array[int] = field(default_factory=_uint32s)

    def __len__(self) -> int:
        return len(self.doc_ids)

    @overload
    def __getitem__(self, i: int) -> Posting: ...
    @overload
    def __getitem__(self, i: slice) -> list[Posting]: ...
    def __getitem__(self, i: int | slice) -> Posting | list[Posting]:
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError("posting index out of range")
        tf = self.tfs[i]
        if not self.positions:
            return Posting(self.doc_ids[i], tf)
        start = sum(islice(self.tfs, i))
        return Posting(self.doc_ids[i], tf, tuple(self.positions[start : start + tf]))

    def __iter__(self) -> Iterator[Posting]:
        if not self.positions:
            for doc_id, tf in zip(self.doc_ids, self.tfs, strict=True):
                yield Posting(doc_id, tf)
            return
        start = 0
        for doc_id, tf in zip(self.doc_ids, self.tfs, strict=True):
            yield Posting(doc_id, tf, tuple(self.positions[start : start + tf]))
            start += tf

    def append(self, doc_id: int, tf: int, positions: Sequence[int] = ()) -> None:
        self.doc_ids.append(doc_id)
        self.tfs.append(tf)
        self.positions.extend(positions)


# list[Posting], list[PlainPosting] and ArrayPostings all fit.
type PostingList = Sequence[PostingLike]


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
    """What a search result shows for a document.

    ``path`` is the file the text was read from. For a crawled page that file is
    the crawl's JSON-lines output, and ``url`` says which record in it.
    """

    path: str
    title: str
    url: str | None = None


class MatchCache(Protocol):
    """What Index needs from its ``functools.lru_cache``-wrapped match function."""

    def __call__(self, node: Node, engine: EngineName, /) -> tuple[int, ...]: ...
    def cache_info(self) -> functools._CacheInfo: ...  # pyright: ignore[reportPrivateUsage]  # typeshed names the cache_info() result type only privately
    def cache_clear(self) -> None: ...


class Index(Mapping[str, PostingList]):
    """The inverted index. Reads like a dict of term -> postings::

        len(index)          # number of terms
        "whale" in index    # is the term indexed?
        index["whale"]      # its postings (KeyError if absent)
        for term in index   # every term

    plus per-document data: ``num_docs``, ``avg_doc_length``, ``doc_length()``,
    ``df()``. ``Mapping`` supplies ``get``, ``keys``, ``items``, ``values`` and
    ``==`` on top of the three methods it requires.

    ``close()`` releases whatever the index was loaded from (``open_index`` calls
    it); after that every lookup raises ``ValueError``.
    """

    def __init__(
        self,
        postings: Mapping[str, PostingList],
        doc_lengths: dict[int, int],
        doc_meta: dict[int, DocMeta],
        layout: Layout = "slots",
        positions: bool = False,
        *,
        closer: Callable[[], None] | None = None,
    ) -> None:
        self.postings = postings
        self.doc_lengths = doc_lengths
        self.doc_meta = doc_meta
        self.layout = layout
        self.positions = positions
        self.closed = False
        self._closer = closer
        self._cached_match = self._new_match_cache()

    # --- the Mapping protocol ---------------------------------------------------

    def __getitem__(self, term: str) -> PostingList:
        self._check_open()
        return self.postings[term]

    def __len__(self) -> int:
        self._check_open()
        return len(self.postings)

    def __iter__(self) -> Iterator[str]:
        self._check_open()
        return iter(self.postings)

    def __contains__(self, term: object) -> bool:
        # Mapping's default tries self[term], which would decode the postings of
        # a memory-mapped index; the term table answers without that.
        self._check_open()
        return term in self.postings

    def __repr__(self) -> str:
        if self.closed:
            return "Index(closed)"
        return (
            f"Index(terms={len(self):_}, docs={self.num_docs:_}, "
            f"layout={self.layout!r}, positions={self.positions})"
        )

    # --- per-document data --------------------------------------------------------

    @property
    def num_docs(self) -> int:
        return len(self.doc_lengths)

    @functools.cached_property
    def avg_doc_length(self) -> float:
        # BM25 needs this for every scored posting: computed once, then stored in
        # the instance __dict__, where the next lookup finds it first.
        return sum(self.doc_lengths.values()) / self.num_docs if self.num_docs else 0.0

    def doc_length(self, doc_id: int) -> int:
        return self.doc_lengths[doc_id]

    def df(self, term: str) -> int:
        """In how many documents ``term`` occurs (0 if never)."""
        self._check_open()
        from_header = getattr(self.postings, "df", None)  # memory-mapped postings
        if from_header is not None:
            return from_header(term)
        plist = self.postings.get(term)
        return 0 if plist is None else len(plist)

    def doc_ids(self, term: str) -> Sequence[int]:
        """Sorted doc_ids of the documents containing ``term``."""
        plist = self.get(term)
        if plist is None:
            return ()
        if isinstance(plist, ArrayPostings):
            return plist.doc_ids
        return [p.doc_id for p in plist]

    def all_doc_ids(self) -> list[int]:
        # doc_meta is filled in doc_id order and a dict keeps insertion order.
        return list(self.doc_meta)

    def posting_count(self) -> int:
        return sum(len(plist) for plist in self.values())

    # --- matching, cached ---------------------------------------------------------

    def _new_match_cache(self) -> MatchCache:
        # The cache belongs to this index: a module-level @lru_cache taking the
        # index as an argument would keep every index alive for as long as one
        # of its queries stays cached.
        @functools.lru_cache(maxsize=256)
        def cached_match(node: Node, engine: EngineName) -> tuple[int, ...]:
            return tuple(node.evaluate(self, engine))

        return cached_match

    def match(self, node: Node, engine: EngineName = "merge") -> tuple[int, ...]:
        """doc_ids matching a parsed query; repeated queries come from the cache."""
        hits = self._cached_match.cache_info().hits
        ids = self._lookup(node, engine)
        how = "cache hit" if self._cached_match.cache_info().hits > hits else "computed"
        # Logged after the timed call, so writing the log line is not timed.
        log.debug("match %s [%s]: %d docs, %s", node, engine, len(ids), how)
        return ids

    @timed
    def _lookup(self, node: Node, engine: EngineName) -> tuple[int, ...]:
        return self._cached_match(node, engine)

    # --- lifecycle ----------------------------------------------------------------

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._cached_match.cache_clear()
        if self._closer is not None:
            self._closer()

    def _check_open(self) -> None:
        if self.closed:
            raise ValueError("operation on a closed index")

    # pickle stores the data, not the cache (a function can't be pickled).
    def __getstate__(self) -> dict[str, Any]:
        return {
            "postings": dict(self.postings),
            "doc_lengths": self.doc_lengths,
            "doc_meta": self.doc_meta,
            "layout": self.layout,
            "positions": self.positions,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(**state)
