"""What the routes do, without HTTP: plain synchronous functions over the index.

All of it is CPU work (parsing, scoring, snippets) plus a little file reading,
which is why the routes calling it are plain ``def``: FastAPI runs them in its
thread pool instead of on the event loop.
"""

import math
import threading
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import document_text
from findex.model import Index
from findex.query import parse
from findex.search import SCORERS, ScorerName, SearchResult
from findex.semantic import Found, Mode, Semantic, find, found_snippet, query_terms
from findex.snippets import Part

SNIPPET_CACHE = 2048

type _SnippetKey = tuple[str, str | None, frozenset[str], int | None]
_snippets: OrderedDict[_SnippetKey, tuple[Part, ...]] = OrderedDict()
_snippets_lock = threading.Lock()  # the routes run in a thread pool


def cached_snippet(
    index: Index, found: Found, terms: frozenset[str], semantic: Semantic | None
) -> tuple[Part, ...]:
    """``found_snippet``, remembered for the last ``SNIPPET_CACHE`` (file, URL,
    terms, passage) keys: a popular query costs its snippets once.
    Lab 7 had the same cache; without it the server does ~10 ms of file
    reading per request."""
    key = (found.result.path, found.result.url, terms, found.passage)
    with _snippets_lock:
        parts = _snippets.get(key)
        if parts is not None:
            _snippets.move_to_end(key)
            return parts
    parts = tuple(found_snippet(index, found, terms, semantic))
    with _snippets_lock:
        _snippets[key] = parts
        if len(_snippets) > SNIPPET_CACHE:
            _snippets.popitem(last=False)
    return parts


@dataclass(frozen=True, slots=True)
class FoundHit:
    rank: int
    result: SearchResult
    parts: tuple[Part, ...]  # the snippet, pieces flagged when they are query terms

    @property
    def source(self) -> str:
        return self.result.url or Path(self.result.path).name


@dataclass(frozen=True, slots=True)
class Page:
    query: str
    parsed: str  # how the query was read (keyword mode), or the mode's name
    mode: Mode
    scorer: ScorerName
    page: int
    k: int
    total: int
    hits: tuple[FoundHit, ...]
    took_ms: float

    @property
    def pages(self) -> int:
        return max(1, math.ceil(self.total / self.k))


def search_page(
    index: Index,
    query: str,
    scorer: ScorerName,
    k: int,
    page: int,
    mode: Mode = "keyword",
    semantic: Semantic | None = None,
) -> Page:
    """Results ``(page - 1) * k + 1`` to ``page * k``. Raises ``QueryError`` (bad
    keyword syntax) and ``ValueError`` (a semantic mode without embeddings)."""
    started = time.perf_counter()
    found, total = find(index, query, mode, page * k, SCORERS[scorer], semantic)
    parsed = str(parse(query)) if mode == "keyword" else f"{mode} search"
    terms = query_terms(query)
    first = (page - 1) * k
    hits = tuple(
        FoundHit(first + i + 1, f.result, cached_snippet(index, f, terms, semantic))
        for i, f in enumerate(found[first:])
    )
    took = (time.perf_counter() - started) * 1000
    return Page(query, parsed, mode, scorer, page, k, total, hits, took)


def snippet_text(parts: tuple[Part, ...]) -> tuple[str, list[tuple[int, int]]]:
    """A snippet as plain text plus the character ranges of the matched terms."""
    text: list[str] = []
    ranges: list[tuple[int, int]] = []
    pos = 0
    for piece, is_term in parts:
        if is_term:
            ranges.append((pos, pos + len(piece)))
        text.append(piece)
        pos += len(piece)
    return "".join(text), ranges


@dataclass(frozen=True, slots=True)
class TextSlice:
    text: str | None
    chars: int | None
    offset: int
    next_offset: int | None


def document_slice(path: str, url: str | None, offset: int, length: int) -> TextSlice:
    raw = document_text(path, url)
    if raw is None:
        return TextSlice(None, None, offset, None)
    # NFC, like the snippet search, so offsets from first_hit() line up.
    text = unicodedata.normalize("NFC", raw)
    end = min(len(text), offset + length)
    return TextSlice(
        text[offset:end], len(text), offset, end if end < len(text) else None
    )
