"""Semantic search: passages as vectors, the query as a vector, cosine similarity.

    findex embed index.bin                 # -> index.emb.npy, .chunks.npy, .json
    findex search index.bin "..." --mode semantic | hybrid

Files next to the index (``<prefix>`` defaults to the index path minus its suffix
plus ``.emb``):

- ``<prefix>.npy`` — ``float32[n_passages, d]``, every row L2-normalised, so
  cosine similarity is a plain dot product: ``vectors @ q``;
- ``<prefix>.chunks.npy`` — ``int64[n_passages, 3]``: document number, start and
  end character of the passage in that document's text;
- ``<prefix>.json`` — the model, and the documents by *name* (file name or URL),
  not by doc_id: an index rebuilt elsewhere may number them differently.
"""

import json
import logging
import re
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final, Literal, Protocol

import numpy as np
import numpy.typing as npt

from findex.chunks import passages
from findex.corpus import document_text
from findex.model import Index
from findex.query import QueryError, parse
from findex.scoring import Scorer
from findex.search import DEFAULT_SCORER, SearchResult, search
from findex.snippets import Part, snippet_for, snippet_parts
from findex.tokenize import tokenize

log = logging.getLogger(__name__)

MODEL: Final = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K: Final = 60
type Mode = Literal["keyword", "semantic", "hybrid"]
MODES: Final[tuple[Mode, ...]] = ("keyword", "semantic", "hybrid")
type Vectors = npt.NDArray[np.float32]


class Embedder(Protocol):
    @property
    def name(self) -> str: ...
    def embed(self, texts: Sequence[str]) -> Vectors:
        """``float32[len(texts), d]``, each row of length 1."""
        ...


def normalise(vectors: npt.NDArray[np.floating]) -> Vectors:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.maximum(norms, 1e-12)).astype(np.float32)


class FastEmbedder:
    """all-MiniLM-L6-v2 (384 dimensions) through fastembed: ONNX Runtime, no
    PyTorch, small enough for a 512 MB server. ``pip install findex[semantic]``."""

    def __init__(self, model: str = MODEL, cache_dir: str | None = None) -> None:
        from fastembed import TextEmbedding  # pyright: ignore[reportMissingTypeStubs]  # no stubs; an optional extra

        self._name = model
        self._model = TextEmbedding(model, cache_dir=cache_dir)

    @property
    def name(self) -> str:
        return self._name

    def embed(self, texts: Sequence[str]) -> Vectors:
        rows = list(self._model.embed(list(texts), batch_size=32))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # untyped library
        return normalise(np.asarray(rows, dtype=np.float32))


class HashEmbedder:
    """A stand-in for tests: a bag of hashed tokens. Texts sharing words are close;
    it knows nothing about meaning, and needs no model or network."""

    def __init__(self, dims: int = 64) -> None:
        self.dims = dims

    @property
    def name(self) -> str:
        return f"hash-{self.dims}"

    def embed(self, texts: Sequence[str]) -> Vectors:
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                out[row, zlib.crc32(token.encode()) % self.dims] += 1.0
        return normalise(out)


def default_prefix(index_path: Path) -> Path:
    return index_path.with_suffix(".emb")


def document_name(index: Index, doc_id: int) -> str:
    meta = index.doc_meta[doc_id]
    return meta.url or Path(meta.path).name


# --- building -------------------------------------------------------------------


def embed_index(
    index: Index,
    embedder: Embedder,
    prefix: Path,
    words: int = 200,
    batch: int = 64,
    on_batch: Callable[[int], None] | None = None,
    on_start: Callable[[int], None] | None = None,
) -> int:
    """Embed every passage of every document; returns the number of passages.

    ``on_start(total)`` is called once the documents are cut into passages, and
    ``on_batch(n)`` after each batch of ``n`` passages is embedded.
    """
    names: list[str] = []
    spans: list[tuple[int, int, int]] = []
    texts: list[str] = []
    for doc_id in index.doc_meta:
        meta = index.doc_meta[doc_id]
        text = document_text(meta.path, meta.url)
        if text is None:
            log.warning("no text for %s; not embedded", document_name(index, doc_id))
            continue
        number = len(names)
        names.append(document_name(index, doc_id))
        for p in passages(doc_id, text, words):
            spans.append((number, p.start, p.end))
            texts.append(p.text)
    if on_start is not None:
        on_start(len(texts))
    vectors = np.zeros((0, 0), dtype=np.float32)
    parts: list[Vectors] = []
    for start in range(0, len(texts), batch):
        parts.append(embedder.embed(texts[start : start + batch]))
        if on_batch is not None:
            on_batch(min(batch, len(texts) - start))
    if parts:
        vectors = np.concatenate(parts)
    np.save(_paths(prefix)[0], vectors)
    dims = vectors.shape[1] if vectors.size else 0
    _save_meta(prefix, embedder.name, words, names, spans, dims)
    return len(texts)


def _paths(prefix: Path) -> tuple[Path, Path, Path]:
    base = str(prefix)
    return Path(f"{base}.npy"), Path(f"{base}.chunks.npy"), Path(f"{base}.json")


def _save_meta(
    prefix: Path,
    model: str,
    words: int,
    names: list[str],
    spans: list[tuple[int, int, int]],
    dims: int,
) -> None:
    _, chunks_path, meta_path = _paths(prefix)
    np.save(chunks_path, np.asarray(spans, dtype=np.int64).reshape(-1, 3))
    meta = {"model": model, "dims": dims, "words": words, "documents": names}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


# --- searching ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Passages:
    """Passage vectors and where each passage is, for one loaded index."""

    model: str
    vectors: Vectors  # memory-mapped from the .npy file
    doc_ids: npt.NDArray[np.int64]  # doc_id of each passage in *this* index
    starts: npt.NDArray[np.int64]
    ends: npt.NDArray[np.int64]
    first: npt.NDArray[np.int64]  # per doc_id: first passage (or -1)
    count: npt.NDArray[np.int64]  # per doc_id: number of passages

    @classmethod
    def load(cls, prefix: Path, index: Index) -> Passages:
        vectors_path, chunks_path, meta_path = _paths(prefix)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        by_name = {document_name(index, d): d for d in index.doc_meta}
        names: list[str] = meta["documents"]
        missing = [n for n in names if n not in by_name]
        if missing:
            raise ValueError(
                f"{len(missing)} embedded documents are not in the index, e.g. {missing[0]}"
            )
        to_doc = np.asarray([by_name[n] for n in names], dtype=np.int64)
        chunks = np.load(chunks_path)
        vectors: Vectors = np.load(vectors_path, mmap_mode="r")
        doc_ids = to_doc[chunks[:, 0]] if chunks.size else np.zeros(0, dtype=np.int64)
        size = index.doc_length_array.size
        first = np.full(size, -1, dtype=np.int64)
        count = np.zeros(size, dtype=np.int64)
        # Passages of a document are consecutive: its first one and how many.
        np.add.at(count, doc_ids, 1)
        seen = np.unique(doc_ids, return_index=True)
        first[seen[0]] = seen[1]
        return cls(
            meta["model"], vectors, doc_ids, chunks[:, 1], chunks[:, 2], first, count
        )

    def rank(self, query: Vectors, k: int) -> list[tuple[int, float, int]]:
        """``k`` best ``(doc_id, similarity, passage)``: each document scored by its
        best passage. One matrix-vector product for all passages."""
        if not self.doc_ids.size or k <= 0:
            return []
        sims = self.vectors @ query.ravel()  # cosine: all rows have length 1
        best = np.full(self.first.size, -np.inf, dtype=np.float32)
        np.maximum.at(best, self.doc_ids, sims)
        candidates = np.flatnonzero(np.isfinite(best))
        k = min(k, candidates.size)
        top = candidates[np.argpartition(-best[candidates], k - 1)[:k]]
        top = top[np.lexsort((top, -best[top]))]
        out: list[tuple[int, float, int]] = []
        for doc_id in top:
            start = int(self.first[doc_id])
            own = sims[start : start + int(self.count[doc_id])]
            out.append((int(doc_id), float(best[doc_id]), start + int(np.argmax(own))))
        return out

    def passage_text(self, index: Index, passage: int) -> str | None:
        meta = index.doc_meta[int(self.doc_ids[passage])]
        text = _raw_text(meta.path, meta.url)
        if text is None:
            return None
        return text[int(self.starts[passage]) : int(self.ends[passage])]


@lru_cache(maxsize=8)
def _raw_text(path: str, url: str | None) -> str | None:
    return document_text(path, url)


_SYNTAX = re.compile(r'\b(?:AND|OR|NOT)\b|[()"]')


def query_text(query: str) -> str:
    """The words of a findex query, without its operators, for the embedder."""
    return " ".join(_SYNTAX.sub(" ", query).split())


def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: ``Σ 1 / (k + rank)`` over the lists a document is in.

    Ranks, not raw scores: BM25 and cosine live on different scales, and adding
    them would let whichever is numerically larger decide alone.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


# --- one entry point for the three modes ----------------------------------------


@dataclass(frozen=True, slots=True)
class Found:
    result: SearchResult
    passage: int | None  # best passage (semantic and hybrid modes), if known


@dataclass(frozen=True, slots=True)
class Semantic:
    """What the semantic modes need: the passages of one index and a model."""

    passages: Passages
    embedder: Embedder

    def query_vector(self, query: str) -> Vectors:
        return self.embedder.embed([query_text(query) or query])[0]


POOL: Final = 100  # how deep each list goes before fusion


def find(
    index: Index,
    query: str,
    mode: Mode = "keyword",
    k: int = 10,
    scorer: Scorer = DEFAULT_SCORER,
    semantic: Semantic | None = None,
) -> tuple[list[Found], int]:
    """The ``k`` best documents for ``query`` in ``mode``, and how many qualify.

    keyword: the Boolean query filters, the scorer ranks (raises QueryError).
    semantic: every document competes, ranked by its best passage's cosine.
    hybrid: the keyword and semantic top-``POOL`` lists fused with RRF; a query
    that is not valid findex syntax is then simply not used for keywords.
    """
    if mode == "keyword":
        node = parse(query)
        results = search(index, node, scorer, k)
        return [Found(r, None) for r in results], len(index.match(node))
    if semantic is None:
        raise ValueError("semantic search is not available: no embeddings loaded")
    vector = semantic.query_vector(query)
    if mode == "semantic":
        ranked = semantic.passages.rank(vector, k)
        found = [Found(_result(index, d, s), p) for d, s, p in ranked]
        return found, int(np.count_nonzero(semantic.passages.count))
    meaning = semantic.passages.rank(vector, max(POOL, k))
    try:
        keywords = [r.doc_id for r in search(index, parse(query), scorer, max(POOL, k))]
    except QueryError:
        keywords = []
    best_passage = {d: p for d, _, p in meaning}
    fused = rrf([keywords, [d for d, _, _ in meaning]])
    found = [Found(_result(index, d, s), best_passage.get(d)) for d, s in fused[:k]]
    return found, len(fused)


def _result(index: Index, doc_id: int, score: float) -> SearchResult:
    meta = index.doc_meta[doc_id]
    return SearchResult(score, doc_id, meta.title, meta.path, meta.url)


def query_terms(query: str) -> frozenset[str]:
    """Words to highlight: the query's terms, or its tokens if it is not valid syntax."""
    try:
        return frozenset(parse(query).terms())
    except QueryError:
        return frozenset(tokenize(query_text(query)))


def found_snippet(
    index: Index, found: Found, terms: frozenset[str], semantic: Semantic | None
) -> list[Part]:
    """The best passage around a semantic hit, the keyword snippet otherwise."""
    if found.passage is not None and semantic is not None:
        text = semantic.passages.passage_text(index, found.passage)
        if text is not None:
            return snippet_parts(text, terms)
    return snippet_for(index, found.result.doc_id, terms)
