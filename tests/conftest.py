"""Shared fixtures: a tiny corpus on disk, the index built from it, and its file.

The session-scoped ones are built once per test run, so tests must not change
them; a test that needs to mutate an index builds its own with ``make_index``.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from findex.corpus import Document, iter_documents
from findex.index import build_index
from findex.model import Index, Layout
from findex.store import save

CORPUS: dict[str, str] = {
    "whale.txt": "Title: The Whale\n\nThe white whale swam. Call me Ishmael. The whale!",
    "pirates.txt": "Title: Pirates\n\nA pirate buried the treasure; a dead man kept watch.",
    "sea.txt": "The sea, the sea: a ship, a captain and a long voyage over the sea.",
    "kit.txt": "Кіт і пес. Кіт спить, пес гавкає. Straße, café, don’t.",
}

type MakeIndex = Callable[..., Index]


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("corpus")
    for name, text in CORPUS.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def corpus_index(corpus_dir: Path) -> Index:
    """The tiny corpus, indexed with positions. Read-only: shared by many tests."""
    return build_index(iter_documents(corpus_dir), positions=True)


@pytest.fixture(scope="session")
def index_file(corpus_index: Index, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("index") / "index.bin"
    save(corpus_index, path)
    return path


@pytest.fixture
def make_index() -> MakeIndex:
    """``make_index("text 0", "text 1", layout=..., positions=...)``: in-memory
    documents d0.txt, d1.txt, ... indexed in order."""

    def make(*texts: str, layout: Layout = "slots", positions: bool = False) -> Index:
        docs = (Document(i, Path(f"d{i}.txt"), t) for i, t in enumerate(texts))
        return build_index(docs, layout, positions)

    return make
