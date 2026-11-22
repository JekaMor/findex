"""Snippets: a window of text around the best match, with the terms marked.

Matches are found with one regular expression built from the query terms, not by
tokenizing the whole document: on a 3 MB book that is ~20 ms instead of ~500 ms.
It uses the tokenizer's notion of a word boundary and its apostrophes, and
ignores case; the one difference is that it cannot see ``casefold`` expansions
(``Straße`` is not found for the term ``strasse``).
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Collection
from itertools import islice
from typing import NamedTuple

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


def _hits(text: str, terms: Collection[str]) -> list[Hit]:
    # lower() keeps the string's length almost always, and a case-sensitive
    # scan of it is three times faster than re.IGNORECASE on the original.
    # If lower() did change the length (a Turkish İ), fall back to IGNORECASE.
    lowered = text.lower()
    haystack, flags = (
        (lowered, 0) if len(lowered) == len(text) else (text, re.IGNORECASE)
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


def snippet_parts(text: str, terms: Collection[str], radius: int = 80) -> list[Part]:
    """About ``radius`` characters each side of the best match, as pieces of text
    flagged when they are a query term. Without a match: the start of the text."""
    text = unicodedata.normalize("NFC", text)
    hits = _hits(text, terms) if terms else []
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


def first_hit(text: str, terms: Collection[str]) -> int | None:
    """Where the first query term starts in NFC ``text``, or None."""
    hits = _hits(text, terms) if terms else []
    return hits[0].start if hits else None


def highlight_parts(text: str, terms: Collection[str]) -> list[Part]:
    """All of NFC ``text``, cut into pieces flagged when they are a query term."""
    parts: list[Part] = []
    pos = 0
    for hit in _hits(text, terms) if terms else []:
        parts += [(text[pos : hit.start], False), (text[hit.start : hit.end], True)]
        pos = hit.end
    parts.append((text[pos:], False))
    return [part for part in parts if part[0]]
