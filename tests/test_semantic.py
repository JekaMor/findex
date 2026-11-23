"""Semantic and hybrid search, with HashEmbedder standing in for the model."""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from findex import cli
from findex.corpus import Document
from findex.index import build_index
from findex.model import Index
from findex.semantic import (
    HashEmbedder,
    Passages,
    Semantic,
    embed_index,
    find,
    found_snippet,
    query_terms,
    query_text,
    rrf,
)
from findex.store import save
from findex.web import service
from findex.web.app import create_app, get_index, get_semantic
from findex.web.settings import Settings, get_settings

BOOKS = {
    "whales.txt": "Title: Whales\n\nThe white whale swam in the cold sea.\n\n"
    "Sailors hunted the great whale for its oil.",
    "castle.txt": "Title: Castle\n\nA vampire lived in a dark castle.\n\n"
    "He drank blood at night and slept in a coffin.",
    "garden.txt": "Title: Garden\n\nRoses and tulips grew in the garden.\n\n"
    "The gardener watered the flowers every morning.",
}


@pytest.fixture
def books(tmp_path: Path) -> tuple[Index, Path, Path]:
    """Three small books on disk, their index file and the embeddings prefix."""
    root = tmp_path / "books"
    root.mkdir()
    docs = []
    for i, (name, text) in enumerate(BOOKS.items()):
        (root / name).write_text(text, encoding="utf-8")
        docs.append(Document(i, root / name, text))
    index = build_index(docs, "array", positions=True)
    index_path = tmp_path / "books.bin"
    save(index, index_path)
    prefix = tmp_path / "books.emb"
    embed_index(index, HashEmbedder(), prefix, words=10)
    return index, index_path, prefix


def semantic_for(index: Index, prefix: Path) -> Semantic:
    return Semantic(Passages.load(prefix, index), HashEmbedder())


def test_embed_writes_normalised_vectors_and_passage_spans(
    books: tuple[Index, Path, Path],
) -> None:
    _, _, prefix = books

    vectors = np.load(f"{prefix}.npy")
    chunks = np.load(f"{prefix}.chunks.npy")
    meta = json.loads(Path(f"{prefix}.json").read_text(encoding="utf-8"))

    assert vectors.dtype == np.float32 and vectors.shape == (chunks.shape[0], 64)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert chunks.shape[0] >= 6  # every book is cut into several passages
    assert meta["documents"] == list(BOOKS) and meta["model"] == "hash-64"
    first_doc, start, end = chunks[0]
    assert BOOKS["whales.txt"][start:end].startswith("Title: Whales")
    assert first_doc == 0


def test_passages_follow_documents_by_name_not_number(
    books: tuple[Index, Path, Path],
) -> None:
    index, _, prefix = books
    # The same books indexed in the opposite order get other doc_ids.
    names = list(BOOKS)[::-1]
    root = Path(index.doc_meta[0].path).parent
    reversed_index = build_index(
        (Document(i, root / n, BOOKS[n]) for i, n in enumerate(names)),
        "array",
        positions=True,
    )

    passages = Passages.load(prefix, reversed_index)
    castle = names.index("castle.txt")

    found = passages.rank(HashEmbedder().embed(["vampire castle blood"])[0], 1)
    assert found[0][0] == castle


def test_rank_scores_each_document_by_its_best_passage(
    books: tuple[Index, Path, Path],
) -> None:
    index, _, prefix = books
    semantic = semantic_for(index, prefix)

    ranked = semantic.passages.rank(
        semantic.query_vector("gardener watered flowers"), 3
    )

    doc_id, score, passage = ranked[0]
    assert index.doc_meta[doc_id].title == "Garden"
    assert 0 < score <= 1.0
    text = semantic.passages.passage_text(index, passage)
    assert text is not None and "gardener" in text  # the second passage, not the first
    assert [r[1] for r in ranked] == sorted((r[1] for r in ranked), reverse=True)


def test_rrf_adds_reciprocal_ranks() -> None:
    fused = dict(rrf([[1, 2, 3], [3, 1]], k=60))

    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[2] == pytest.approx(1 / 62)
    assert [d for d, _ in rrf([[1, 2, 3], [3, 1]])] == [1, 3, 2]


def test_query_text_and_terms_drop_the_syntax() -> None:
    assert (
        query_text('whale AND (sea OR "dead man") NOT ship')
        == "whale sea dead man ship"
    )
    assert query_terms("Whale OR sea") == {"whale", "sea"}
    assert query_terms("a (broken") == {
        "a",
        "broken",
    }  # not valid syntax: just its words


def test_find_in_the_three_modes(books: tuple[Index, Path, Path]) -> None:
    index, _, prefix = books
    semantic = semantic_for(index, prefix)

    keyword, keyword_total = find(index, "whale", "keyword", 5)
    meaning, meaning_total = find(
        index, "vampire night blood", "semantic", 2, semantic=semantic
    )
    hybrid, _ = find(index, "whale OR vampire", "hybrid", 3, semantic=semantic)

    assert [f.result.title for f in keyword] == ["Whales"] and keyword_total == 1
    assert meaning[0].result.title == "Castle" and meaning[0].passage is not None
    assert meaning_total == 3 and len(meaning) == 2
    assert {f.result.title for f in hybrid[:2]} == {"Whales", "Castle"}
    with pytest.raises(ValueError, match="not available"):
        find(index, "whale", "semantic", 5)


def test_semantic_snippet_is_the_best_passage(books: tuple[Index, Path, Path]) -> None:
    index, _, prefix = books
    semantic = semantic_for(index, prefix)
    (hit,), _ = find(index, "coffin", "semantic", 1, semantic=semantic)

    parts = found_snippet(index, hit, frozenset({"coffin"}), semantic)

    assert ("coffin", True) in parts


def test_api_and_page_offer_the_modes(books: tuple[Index, Path, Path]) -> None:
    index, index_path, prefix = books
    app = create_app()
    app.dependency_overrides[get_index] = lambda: index
    app.dependency_overrides[get_settings] = lambda: Settings(index_path=index_path)
    app.dependency_overrides[get_semantic] = lambda: semantic_for(index, prefix)
    client = TestClient(app)

    body = client.get(
        "/search", params={"q": "vampire blood", "mode": "semantic"}
    ).json()
    page = client.get("/", params={"q": "vampire", "mode": "hybrid"})

    assert body["mode"] == "semantic" and body["results"][0]["title"] == "Castle"
    assert page.status_code == 200 and 'value="hybrid" selected' in page.text
    bad = client.get("/search", params={"q": "x", "mode": "telepathy"})
    assert bad.status_code == 422

    app.dependency_overrides[get_semantic] = lambda: None
    off = client.get("/search", params={"q": "vampire", "mode": "semantic"})
    assert off.status_code == 503
    assert "not available" in off.json()["detail"]
    assert 'name="mode"' not in client.get("/", params={"q": "vampire"}).text


def test_cli_semantic_mode(
    books: tuple[Index, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, index_path, prefix = books
    monkeypatch.setattr(cli, "_embedder", lambda model: HashEmbedder())
    runner = CliRunner()

    result = runner.invoke(
        cli.app,
        [
            "search",
            str(index_path),
            "flowers in the garden",
            "--mode",
            "semantic",
            "--json",
            "-k",
            "1",
        ],
    )
    missing = runner.invoke(
        cli.app,
        [
            "search",
            str(index_path),
            "x",
            "--mode",
            "hybrid",
            "--embeddings",
            str(prefix) + "-nope",
        ],
    )

    assert result.exit_code == 0, result.stderr
    (record,) = map(json.loads, result.stdout.splitlines())
    assert record["title"] == "Garden"
    assert missing.exit_code == 1 and "run `findex embed` first" in missing.stderr


def test_snippets_are_cached_per_document_terms_and_passage(
    books: tuple[Index, Path, Path],
) -> None:
    index, _, prefix = books
    semantic = semantic_for(index, prefix)
    (hit,), _ = find(index, "coffin", "semantic", 1, semantic=semantic)
    (plain,), _ = find(index, "coffin", "keyword", 1)

    first = service.cached_snippet(index, hit, frozenset({"coffin"}), semantic)
    again = service.cached_snippet(index, hit, frozenset({"coffin"}), semantic)
    keyword = service.cached_snippet(index, plain, frozenset({"coffin"}), None)

    assert again is first  # served from the cache
    assert keyword is not first and ("coffin", True) in keyword
