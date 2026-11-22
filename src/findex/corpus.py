"""Lazy corpus intake: one document in memory at a time."""

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)


class Document(NamedTuple):
    doc_id: int
    path: Path
    text: str


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        log.warning("bad UTF-8 in %s; undecodable bytes replaced", path)
        return path.read_text(encoding="utf-8", errors="replace")


def corpus_paths(root: Path) -> list[Path]:
    """Every ``*.txt`` file under ``root``, in the order ``iter_documents`` uses.

    A file's position in this list is its doc_id.
    """
    return list(root.rglob("*.txt"))


def read_documents(paths: Iterable[Path], first_doc_id: int = 0) -> Iterator[Document]:
    """Read ``paths`` one at a time; the n-th path gets doc_id ``first_doc_id + n``.

    An unreadable file is logged and skipped, but still uses up its doc_id, so
    ids never depend on which files happened to fail.
    """
    for doc_id, path in enumerate(paths, first_doc_id):
        try:
            text = _read(path)
        except OSError as exc:
            log.warning("unreadable %s: %s", path, exc)
            continue
        yield Document(doc_id, path, text)


def iter_documents(root: Path) -> Iterator[Document]:
    """Yield every ``*.txt`` file under ``root`` as a Document, lazily."""
    yield from read_documents(root.rglob("*.txt"))
