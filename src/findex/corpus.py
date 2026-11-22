"""Lazy corpus intake: one document in memory at a time.

A corpus is either a directory of ``*.txt`` files or a JSON-lines file with one
document per line (what ``findex crawl`` writes): ``{"url", "title", "text", ...}``.
"""

import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple, cast

log = logging.getLogger(__name__)


class Document(NamedTuple):
    doc_id: int
    path: Path  # the file the text came from
    text: str
    url: str | None = None  # crawled pages: where the text was fetched
    title: str | None = None  # crawled pages: the HTML <title>


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        log.warning("bad UTF-8 in %s; undecodable bytes replaced", path)
        return path.read_text(encoding="utf-8", errors="replace")


def is_jsonl(path: Path) -> bool:
    return path.is_file() and path.suffix == ".jsonl"


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


def read_jsonl(path: Path) -> Iterator[Document]:
    """One Document per line of a JSON-lines file, read line by line.

    A line that is not a JSON object with a string ``text`` is logged and
    skipped (its line number is still its doc_id).
    """
    with path.open(encoding="utf-8", errors="replace") as lines:
        for doc_id, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("%s line %d: not JSON (%s)", path, doc_id + 1, exc.msg)
                continue
            if not isinstance(record, dict):
                log.warning("%s line %d: not a JSON object", path, doc_id + 1)
                continue
            fields = cast(dict[str, object], record)  # JSON object keys are str
            text = fields.get("text")
            if not isinstance(text, str):
                log.warning("%s line %d: no text field", path, doc_id + 1)
                continue
            url, title = fields.get("url"), fields.get("title")
            yield Document(
                doc_id,
                path,
                text,
                url if isinstance(url, str) else None,
                title if isinstance(title, str) else None,
            )


def iter_documents(root: Path) -> Iterator[Document]:
    """Every document of a corpus directory or JSON-lines file, lazily."""
    if is_jsonl(root):
        yield from read_jsonl(root)
    else:
        yield from read_documents(root.rglob("*.txt"))


def document_text(path: str, url: str | None = None) -> str | None:
    """A document's text again, for snippets: the file itself, or the line of a
    JSON-lines file whose ``url`` matches. None if it is gone."""
    file = Path(path)
    try:
        if url is None:
            return _read(file)
        return next((d.text for d in read_jsonl(file) if d.url == url), None)
    except OSError:
        return None
