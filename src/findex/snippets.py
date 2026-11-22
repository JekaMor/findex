"""Snippets: a window of text around the best match, with the terms marked."""

import re
import unicodedata
from collections import Counter
from collections.abc import Collection
from typing import NamedTuple

from findex.tokenize import token_spans

_WHITESPACE = re.compile(r"\s+")


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


def snippet_parts(text: str, terms: Collection[str], radius: int = 80) -> list[Part]:
    """About ``radius`` characters each side of the best match, as pieces of text
    flagged when they are a query term. Without a match: the start of the text."""
    text = unicodedata.normalize("NFC", text)
    hits = [Hit(*span) for span in token_spans(text) if span[0] in terms]
    if not hits:
        tail = "…" if len(text) > 2 * radius else ""
        return [(_clean(text[: 2 * radius]) + tail, False)]

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
    if lo > 0:
        parts.insert(0, ("…", False))
    if hi < len(text):
        parts.append(("…", False))
    return [part for part in parts if part[0]]


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
