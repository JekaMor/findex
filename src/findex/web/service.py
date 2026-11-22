"""What the routes do, without HTTP: plain synchronous functions over the index.

All of it is CPU work (parsing, scoring, snippets) plus a little file reading,
which is why the routes calling it are plain ``def``: FastAPI runs them in its
thread pool instead of on the event loop.
"""

import math
import time
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from findex.corpus import document_text
from findex.model import Index
from findex.query import Node, parse
from findex.search import SCORERS, ScorerName, SearchResult, search
from findex.snippets import Part, snippet_parts


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
    node: Node
    scorer: ScorerName
    page: int
    k: int
    total: int
    hits: tuple[FoundHit, ...]
    took_ms: float

    @property
    def pages(self) -> int:
        return max(1, math.ceil(self.total / self.k))


@lru_cache(maxsize=2048)
def cached_snippet(
    path: str, url: str | None, terms: frozenset[str]
) -> tuple[Part, ...]:
    """A result's snippet. The same top documents come back for popular
    queries, and a long book takes tens of milliseconds to scan."""
    text = document_text(path, url)
    if text is None:
        return (("(text not available)", False),)
    return tuple(snippet_parts(text, terms))


def search_page(
    index: Index, query: str, scorer: ScorerName, k: int, page: int
) -> Page:
    """Results ``(page - 1) * k + 1`` to ``page * k``. Raises ``QueryError``."""
    started = time.perf_counter()
    node = parse(query)
    total = len(index.match(node))  # cached in the index: search() reuses it
    best = search(index, node, SCORERS[scorer], k=page * k)
    terms = frozenset(node.terms())
    first = (page - 1) * k
    hits = tuple(
        FoundHit(first + i + 1, r, cached_snippet(r.path, r.url, terms))
        for i, r in enumerate(best[first:])
    )
    took = (time.perf_counter() - started) * 1000
    return Page(query, node, scorer, page, k, total, hits, took)


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
