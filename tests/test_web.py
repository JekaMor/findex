"""The web service through TestClient. The index is injected with
dependency_overrides; nothing is loaded from an index file except in the two
lifespan tests at the end."""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from typer.testing import CliRunner

from findex.cli import app as cli_app
from findex.model import Index
from findex.web.app import create_app, get_index
from findex.web.settings import Settings, get_settings


@pytest.fixture
def web(corpus_index: Index, tmp_path: Path) -> Iterator[FastAPI]:
    fake_file = tmp_path / "index.bin"
    fake_file.write_bytes(b"x" * 1234)  # only /stats looks at it: its size
    application = create_app()
    application.dependency_overrides[get_index] = lambda: corpus_index
    application.dependency_overrides[get_settings] = lambda: Settings(
        index_path=fake_file
    )
    application.state.started = 0.0
    yield application


@pytest.fixture
def client(web: FastAPI) -> TestClient:
    # Not used as a context manager: the lifespan (which would load a real
    # index file) does not run.
    return TestClient(web)


# --- /search -----------------------------------------------------------------------


def test_search_returns_ranked_hits_with_highlights(client: TestClient) -> None:
    response = client.get("/search", params={"q": "whale"})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "whale" and body["parsed"] == "whale"
    assert body["total"] == 1 and body["pages"] == 1
    (hit,) = body["results"]
    assert hit["title"] == "The Whale" and hit["rank"] == 1 and hit["score"] > 0
    assert hit["source"] == "whale.txt"
    start, end = hit["highlights"][0]
    assert hit["snippet"][start:end].lower() == "whale"
    assert response.headers["x-request-id"]


def test_search_pages_through_results(client: TestClient) -> None:
    query = {"q": "the OR a OR кіт OR sea", "k": 2}

    first = client.get("/search", params=query).json()
    second = client.get("/search", params={**query, "page": 2}).json()
    beyond = client.get("/search", params={**query, "page": 9}).json()

    assert first["total"] == 4 and first["pages"] == 2
    assert [h["rank"] for h in first["results"]] == [1, 2]
    assert [h["rank"] for h in second["results"]] == [3, 4]
    assert {h["doc_id"] for h in first["results"]}.isdisjoint(
        h["doc_id"] for h in second["results"]
    )
    assert beyond["results"] == []


def test_search_scorer_and_phrases(client: TestClient) -> None:
    body = client.get("/search", params={"q": '"dead man"', "scorer": "tfidf"}).json()

    assert body["scorer"] == "tfidf"
    assert [h["title"] for h in body["results"]] == ["Pirates"]


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": ""},
        {"q": "x" * 201},
        {"q": "whale", "k": "abc"},
        {"q": "whale", "k": 0},
        {"q": "whale", "k": 51},
        {"q": "whale", "page": 0},
        {"q": "whale", "scorer": "pagerank"},
    ],
)
def test_bad_parameters_are_422(client: TestClient, params: dict[str, object]) -> None:
    response = client.get("/search", params=params)

    assert response.status_code == 422
    assert "detail" in response.json()


def test_bad_query_syntax_is_422_with_a_message(client: TestClient) -> None:
    response = client.get("/search", params={"q": "whale AND (sea"})

    assert response.status_code == 422
    assert response.json() == {"detail": "bad query: missing )"}


# --- /docs/{id}, /stats, /health -------------------------------------------------------


def test_document_and_its_text_in_slices(
    client: TestClient, corpus_index: Index
) -> None:
    doc_id = next(i for i, m in corpus_index.doc_meta.items() if m.title == "Pirates")

    whole = client.get(f"/docs/{doc_id}").json()
    head = client.get(f"/docs/{doc_id}", params={"length": 10}).json()

    assert whole["title"] == "Pirates" and whole["source"] == "pirates.txt"
    assert whole["text"].startswith("Title: Pirates") and whole["next_offset"] is None
    assert head["text"] == whole["text"][:10] and head["next_offset"] == 10
    assert whole["chars"] == len(whole["text"])


def test_unknown_document_is_404(client: TestClient) -> None:
    response = client.get("/docs/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "no document 999"}


def test_stats(client: TestClient) -> None:
    body = client.get("/stats").json()

    assert body["documents"] == 4
    assert body["positions"] is True
    assert body["index_bytes"] == 1234
    assert body["terms"] > 10


def test_health_is_ok_with_an_index(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "documents": 4}


def test_health_is_503_before_the_index_is_loaded() -> None:
    bare = TestClient(create_app())  # no override, lifespan not run: no index

    response = bare.get("/health")

    assert response.status_code == 503
    assert response.json() == {"detail": "index not loaded"}


def test_openapi_page_lists_the_routes(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/search", "/docs/{doc_id}", "/stats", "/health"} <= paths.keys()
    assert "/" not in paths  # HTML pages stay out of the API schema


# --- errors ----------------------------------------------------------------------------


def test_unexpected_error_is_a_clean_500_logged_with_the_request_id(
    web: FastAPI, caplog: pytest.LogCaptureFixture
) -> None:
    class Broken:
        def match(self, *args: object) -> object:
            raise RuntimeError("disk on fire")

    web.dependency_overrides[get_index] = lambda: Broken()
    client = TestClient(web, raise_server_exceptions=False)

    with caplog.at_level(logging.INFO, logger="findex.web"):
        response = client.get(
            "/search", params={"q": "whale"}, headers={"x-request-id": "abc123"}
        )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "internal server error",
        "request_id": "abc123",
    }
    assert "disk on fire" not in response.text  # no traceback for the client
    error = next(r for r in caplog.records if r.levelno == logging.ERROR)
    assert "[abc123]" in error.getMessage()
    assert error.exc_info is not None and "disk on fire" in str(error.exc_info[1])


# --- HTML pages ------------------------------------------------------------------------


def test_home_page_without_a_query_shows_examples(client: TestClient) -> None:
    page = client.get("/")

    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "Search 4 documents" in page.text
    assert "white whale" in page.text  # an example query


def test_home_page_renders_highlighted_results(client: TestClient) -> None:
    page = client.get("/", params={"q": "treasure"})

    assert page.status_code == 200
    assert "Pirates" in page.text
    assert "<mark>treasure</mark>" in page.text
    assert "/doc/" in page.text


def test_home_page_reports_a_bad_query(client: TestClient) -> None:
    page = client.get("/", params={"q": '"open'})

    assert page.status_code == 422
    assert "unclosed quote" in page.text


def test_document_page_highlights_the_query(
    client: TestClient, corpus_index: Index
) -> None:
    doc_id = next(i for i, m in corpus_index.doc_meta.items() if m.title == "The Whale")

    page = client.get(f"/doc/{doc_id}", params={"q": "ishmael"})

    assert page.status_code == 200
    assert "<mark>Ishmael</mark>" in page.text
    assert client.get("/doc/999").status_code == 404


def test_html_is_escaped(client: TestClient) -> None:
    page = client.get("/", params={"q": "<script>alert(1)</script>"})

    assert "<script>alert(1)</script>" not in page.text
    assert "&lt;script&gt;" in page.text


# --- lifespan with a real file, and the serve command ----------------------------------


def test_lifespan_loads_the_index_from_settings(
    index_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INDEX_PATH", str(index_file))
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as client:
            assert client.get("/health").json() == {"status": "ok", "documents": 4}
            assert client.get("/search", params={"q": "whale"}).json()["total"] == 1
    finally:
        get_settings.cache_clear()


def test_missing_index_file_fails_at_start_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INDEX_PATH", str(tmp_path / "nope.bin"))
    get_settings.cache_clear()
    try:
        with (
            pytest.raises(ValidationError, match="does not point to a file"),
            TestClient(create_app()),
        ):
            pass
    finally:
        get_settings.cache_clear()


def test_serve_rejects_a_missing_index_in_one_line(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli_app, ["serve", "--index", str(tmp_path / "nope.bin")]
    )

    assert result.exit_code == 1
    assert "bad configuration: INDEX_PATH" in result.stderr
    assert "Traceback" not in result.stderr
