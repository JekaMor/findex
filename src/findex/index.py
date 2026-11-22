"""Build the inverted index in one streaming pass (``findex index`` on the CLI)."""

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping

from findex.corpus import Document
from findex.model import (
    ArrayPostings,
    DocMeta,
    Index,
    Layout,
    PlainPosting,
    Posting,
    PostingLike,
    PostingList,
)
from findex.timing import timed
from findex.tokenize import tokenize

_TITLE = re.compile(r"^Title:[ \t]*(.+?)\s*$", re.MULTILINE)

type TermStats = tuple[str, int, tuple[int, ...]]  # (term, tf, positions)
type AddPosting = Callable[[str, int, int, tuple[int, ...]], None]


def title_of(doc: Document) -> str:
    """The page's own title, the Gutenberg ``Title:`` header line, or the file name."""
    if doc.title:
        return doc.title
    match = _TITLE.search(doc.text, 0, 5000)
    return match.group(1) if match else doc.path.stem


def _terms(text: str, positions: bool) -> tuple[int, Iterator[TermStats]]:
    """(document length, (term, tf, positions) for each distinct term)."""
    if not positions:
        counts = Counter(tokenize(text))
        return counts.total(), ((term, tf, ()) for term, tf in counts.items())
    where: defaultdict[str, list[int]] = defaultdict(list)
    length = 0
    for length, term in enumerate(tokenize(text), 1):
        where[term].append(length - 1)
    return length, ((term, len(pos), tuple(pos)) for term, pos in where.items())


def _collector(layout: Layout) -> tuple[Mapping[str, PostingList], AddPosting]:
    """An empty postings table for ``layout`` and the function that appends to it."""
    if layout == "array":
        arrays: defaultdict[str, ArrayPostings] = defaultdict(ArrayPostings)

        def add_to_arrays(
            term: str, doc_id: int, tf: int, pos: tuple[int, ...]
        ) -> None:
            arrays[term].append(doc_id, tf, pos)

        return arrays, add_to_arrays

    record: type[Posting | PlainPosting] = (
        Posting if layout == "slots" else PlainPosting
    )
    lists: defaultdict[str, list[PostingLike]] = defaultdict(list)

    def add_to_lists(term: str, doc_id: int, tf: int, pos: tuple[int, ...]) -> None:
        lists[term].append(record(doc_id, tf, pos))

    return lists, add_to_lists


@timed
def build_index(
    docs: Iterable[Document], layout: Layout = "slots", positions: bool = False
) -> Index:
    """Consume ``docs`` once. Only the index grows; the corpus keeps streaming.

    Documents must arrive in increasing ``doc_id`` order (``iter_documents``
    guarantees it), so appending keeps every postings list sorted for free.
    """
    postings, add = _collector(layout)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    last = -1
    for doc in docs:
        if doc.doc_id <= last:
            raise ValueError(
                f"doc_id {doc.doc_id} after {last}: postings would be unsorted"
            )
        last = doc.doc_id
        # The current document's counter is flushed into the postings, then dropped.
        length, terms = _terms(doc.text, positions)
        for term, tf, pos in terms:
            add(term, doc.doc_id, tf, pos)
        doc_lengths[doc.doc_id] = length
        doc_meta[doc.doc_id] = DocMeta(str(doc.path), title_of(doc), doc.url)

    # A plain dict: looking up an unknown term must not insert it.
    return Index(dict(postings), doc_lengths, doc_meta, layout, positions)
