"""Snippets: a window of text around the best match, with the terms marked."""

import unicodedata
from collections import Counter
from collections.abc import Collection
from typing import NamedTuple

from findex.tokenize import token_spans


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


def snippet(
    text: str,
    terms: Collection[str],
    radius: int = 80,
    mark: tuple[str, str] = ("**", "**"),
) -> str:
    """About ``radius`` characters each side of the best match, terms wrapped in
    ``mark``. Without a match: the start of the text."""
    text = unicodedata.normalize("NFC", text)
    hits = [Hit(*span) for span in token_spans(text) if span[0] in terms]
    if not hits:
        return _clean(text[: 2 * radius]) + ("…" if len(text) > 2 * radius else "")

    i, j = _best_window(hits, 2 * radius)
    middle = (hits[i].start + hits[j - 1].end) // 2
    lo = max(0, min(middle - radius, hits[i].start))
    hi = min(len(text), max(middle + radius, hits[j - 1].end))
    # Don't cut words in half at the edges.
    while 0 < lo < hits[i].start and not text[lo - 1].isspace():
        lo += 1
    while hits[j - 1].end < hi < len(text) and not text[hi].isspace():
        hi -= 1

    pieces = []
    pos = lo
    for hit in hits:
        if hit.start < lo or hit.end > hi:
            continue
        pieces += [text[pos : hit.start], mark[0], text[hit.start : hit.end], mark[1]]
        pos = hit.end
    pieces.append(text[pos:hi])
    return (
        ("…" if lo > 0 else "")
        + _clean("".join(pieces))
        + ("…" if hi < len(text) else "")
    )


def _clean(fragment: str) -> str:
    return " ".join(fragment.split())
