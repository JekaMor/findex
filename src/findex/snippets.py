"""Snippets: a window of text around the best match, with the terms marked.

Matches are found with one regular expression built from the query terms, not by
tokenizing the whole document: on a 3 MB book that is ~20 ms instead of ~500 ms.
It uses the tokenizer's notion of a word boundary and its apostrophes, and
ignores case; the one difference is that it cannot see ``casefold`` expansions
(``Straße`` is not found for the term ``strasse``).
"""

import os
import re
import unicodedata
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass
from functools import lru_cache
from itertools import islice
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from findex.corpus import document_text

if TYPE_CHECKING:
    from findex.model import Index

_WHITESPACE = re.compile(r"\s+")
# The best window is looked for among the first MAX_HITS matches: a word that
# occurs thousands of times in a long book would otherwise cost more than the
# rest of the search.
MAX_HITS = 1000
_APOSTROPHE = "['\u2019\u02bc]"  # ' and the two the tokenizer treats the same
_FOLD = str.maketrans({"\u2019": "'", "\u02bc": "'"})


def _term_pattern(terms: Collection[str], flags: int = 0) -> re.Pattern[str]:
    """Any of ``terms``, longest first ("dog's" before "dog"). Word boundaries
    are checked afterwards, in ``_whole_word``: a look-behind tried at every one
    of three million positions made the scan four times slower."""
    words = sorted(terms, key=len, reverse=True)
    body = "|".join(re.escape(w).replace("'", _APOSTROPHE) for w in words)
    return re.compile(body, flags)


def _is_word(ch: str) -> bool:
    return ch.isalnum()  # what [^\W_] means: \w without the underscore


def _whole_word(text: str, start: int, end: int) -> bool:
    """No word character, nor an apostrophe followed by one, on either side:
    the tokenizer would have cut a token exactly here."""
    before = text[start - 1] if start else ""
    after = text[end] if end < len(text) else ""
    if _is_word(before) or _is_word(after):
        return False
    glued_after = (
        after in "'\u2019\u02bc" and end + 1 < len(text) and _is_word(text[end + 1])
    )
    glued_before = (
        before in "'\u2019\u02bc" and start >= 2 and _is_word(text[start - 2])
    )
    return not (after and glued_after) and not (before and glued_before)


@dataclass(frozen=True, slots=True)
class Prepared:
    """A document's text made ready for snippet scans, once.

    ``text`` is NFC. ``folded`` is ``text.lower()`` when that keeps every
    character in place (almost always), else None and scans use re.IGNORECASE
    (a Turkish İ lower-cases to two characters). Profiling showed that doing
    these two passes over a 3 MB book for every query cost more than the regex.
    """

    text: str
    folded: str | None

    @classmethod
    def of(cls, text: str) -> Prepared:
        text = unicodedata.normalize("NFC", text)
        lowered = text.lower()
        return cls(text, lowered if len(lowered) == len(text) else None)


@lru_cache(maxsize=8)
def prepared_document(path: str, url: str | None = None) -> Prepared | None:
    """``Prepared`` text of a document, for the last few documents asked for.

    Eight long books are ~100 MB at most; the same books keep coming back as the
    best results for common words, so most queries find them here.
    """
    text = document_text(path, url)
    return None if text is None else Prepared.of(text)


def _hits(prepared: Prepared, terms: Collection[str]) -> list[Hit]:
    # A case-sensitive scan of the lower-cased text is three times faster than
    # re.IGNORECASE on the original.
    haystack, flags = (
        (prepared.folded, 0)
        if prepared.folded is not None
        else (prepared.text, re.IGNORECASE)
    )
    found = (
        m
        for m in _term_pattern(terms, flags).finditer(haystack)
        if _whole_word(haystack, m.start(), m.end())
    )
    return [
        Hit(m.group().casefold().translate(_FOLD), m.start(), m.end())
        for m in islice(found, MAX_HITS)
    ]


class Hit(NamedTuple):
    term: str
    start: int
    end: int


def _best_window(hits: list[Hit], width: int) -> tuple[int, int]:
    """(first, last + 1) indexes into ``hits`` of the run that fits in ``width``
    characters with the most distinct terms, then the most hits."""
    best, best_key = (0, 1), (0, 0)
    in_window: Counter[str] = Counter()
    j = 0
    for i, first in enumerate(hits):
        j = max(j, i)
        while j < len(hits) and (j == i or hits[j].end - first.start <= width):
            in_window[hits[j].term] += 1
            j += 1
        key = (len(+in_window), j - i)
        if key > best_key:
            best, best_key = (i, j), key
        in_window[first.term] -= 1
    return best


type Part = tuple[str, bool]  # (text, is it a query term?)


def snippet_parts(
    text: str | Prepared, terms: Collection[str], radius: int = 80
) -> list[Part]:
    """About ``radius`` characters each side of the best match, as pieces of text
    flagged when they are a query term. Without a match: the start of the text."""
    prepared = text if isinstance(text, Prepared) else Prepared.of(text)
    hits = _hits(prepared, terms) if terms else []
    if not hits:
        text = prepared.text
        tail = "…" if len(text) > 2 * radius else ""
        return [(_clean(text[: 2 * radius]) + tail, False)]
    return _window(prepared.text, hits, radius)


def _window(
    text: str,
    hits: list[Hit],
    radius: int,
    more_before: bool = False,
    more_after: bool = False,
) -> list[Part]:
    """The snippet around the best run of ``hits`` in ``text``. ``more_before`` /
    ``more_after``: ``text`` is a slice and the document goes on past that end."""
    i, j = _best_window(hits, 2 * radius)
    middle = (hits[i].start + hits[j - 1].end) // 2
    lo = max(0, min(middle - radius, hits[i].start))
    hi = min(len(text), max(middle + radius, hits[j - 1].end))
    # Don't cut words in half at the edges.
    while 0 < lo < hits[i].start and not text[lo - 1].isspace():
        lo += 1
    while hits[j - 1].end < hi < len(text) and not text[hi].isspace():
        hi -= 1

    parts: list[Part] = []
    pos = lo
    for hit in hits:
        if hit.start < lo or hit.end > hi:
            continue
        parts += [(text[pos : hit.start], False), (text[hit.start : hit.end], True)]
        pos = hit.end
    parts.append((text[pos:hi], False))
    # One space for any run of whitespace, none at the two ends.
    parts = [(_WHITESPACE.sub(" ", piece), is_term) for piece, is_term in parts]
    first, last = parts[0], parts[-1]
    parts[0] = (first[0].lstrip(), first[1])
    parts[-1] = (parts[-1][0].rstrip(), last[1])
    if lo > 0 or more_before:
        parts.insert(0, ("…", False))
    if hi < len(text) or more_after:
        parts.append(("…", False))
    return [part for part in parts if part[0]]


# --- snippets from the index's positions -----------------------------------------

REGION = 32 * 1024  # bytes read on each side of the estimated spot
CLUSTER = 30  # tokens: how close query terms must be to count as together


def locate(index: Index, doc_id: int, terms: Collection[str]) -> float | None:
    """Where in ``doc_id`` the query terms cluster best, as a fraction 0..1 of the
    document, from the positions stored in the index. None without positions.

    For every occurrence of every term, ``searchsorted`` counts how many distinct
    terms (and how many occurrences) fall in the next CLUSTER tokens; the best
    start wins. All of it in NumPy: a common word has tens of thousands of
    occurrences in a long book.
    """
    found = [pos for t in terms if (pos := index.positions_in(t, doc_id)).size]
    length = index.doc_length(doc_id)
    if not found or not length:
        return None
    starts = np.sort(np.concatenate(found))
    distinct = np.zeros(starts.size, dtype=np.int64)
    count = np.zeros(starts.size, dtype=np.int64)
    for pos in found:  # each array is sorted already
        first = np.searchsorted(pos, starts, side="left")
        last = np.searchsorted(pos, starts + CLUSTER, side="right")
        distinct += last > first
        count += last - first
    best = int(np.argmax(distinct * (count.max() + 1) + count))
    return (int(starts[best]) + CLUSTER / 2) / length


def snippet_for(
    index: Index, doc_id: int, terms: Collection[str], radius: int = 80
) -> list[Part]:
    """A result's snippet, reading only the part of the file around the match.

    Profiling showed snippets were 99.8 % of a search: every result meant reading,
    normalising, lower-casing and scanning a whole book (up to 3 MB). The index
    already knows where the terms are, so: estimate the byte offset from the token
    position, read REGION bytes either side, find the window there. If the
    estimate misses (no term in that slice), fall back to the whole document.
    """
    meta = index.doc_meta[doc_id]
    if meta.url is None and terms:
        where = locate(index, doc_id, terms)
        region = _read_region(meta.path, where) if where is not None else None
        if region is not None:
            text, more_before, more_after = region
            prepared = Prepared.of(text)
            hits = _hits(prepared, terms)
            if hits:
                return _window(prepared.text, hits, radius, more_before, more_after)
    whole = prepared_document(meta.path, meta.url)
    if whole is None:
        return [("(text not available)", False)]
    return snippet_parts(whole, terms, radius)


def _read_region(path: str, fraction: float) -> tuple[str, bool, bool] | None:
    try:
        with open(path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            start = max(0, min(size, int(fraction * size)) - REGION)
            f.seek(start)
            data = f.read(2 * REGION)
    except OSError:
        return None
    # A slice can cut a UTF-8 sequence at either end: drop the broken bytes.
    text = data.decode("utf-8", errors="ignore")
    return text, start > 0, start + len(data) < size


def snippet(
    text: str,
    terms: Collection[str],
    radius: int = 80,
    mark: tuple[str, str] = ("**", "**"),
) -> str:
    """``snippet_parts`` as one string, query terms wrapped in ``mark``."""
    return "".join(
        f"{mark[0]}{piece}{mark[1]}" if is_term else piece
        for piece, is_term in snippet_parts(text, terms, radius)
    )


def _clean(fragment: str) -> str:
    return " ".join(fragment.split())


def first_hit(text: str, terms: Collection[str]) -> int | None:
    """Where the first query term starts in NFC ``text``, or None."""
    hits = _hits(Prepared.of(text), terms) if terms else []
    return hits[0].start if hits else None


def highlight_parts(text: str, terms: Collection[str]) -> list[Part]:
    """All of NFC ``text``, cut into pieces flagged when they are a query term."""
    parts: list[Part] = []
    pos = 0
    for hit in _hits(Prepared.of(text), terms) if terms else []:
        parts += [(text[pos : hit.start], False), (text[hit.start : hit.end], True)]
        pos = hit.end
    parts.append((text[pos:], False))
    return [part for part in parts if part[0]]
