import random
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import corpus_paths, iter_documents
from findex.index import build_index
from findex.parallel import (
    EXECUTOR_NAMES,
    Chunk,
    build_parallel,
    build_partial,
    chunks,
    gil_enabled,
    merge,
)
from findex.store import save


def saved_bytes(index, tmp_path: Path, name: str) -> bytes:
    path = tmp_path / name
    save(index, path)
    return path.read_bytes()


def lab4_bytes(corpus_dir: Path, tmp_path: Path) -> bytes:
    """The index exactly as Lab 4 built it: one serial pass over the corpus."""
    return saved_bytes(
        build_index(iter_documents(corpus_dir), positions=True), tmp_path, "lab4.bin"
    )


def test_chunks_cover_consecutive_ranges() -> None:
    paths = [Path(f"{i}.txt") for i in range(10)]

    parts = chunks(paths, 4)

    assert [c.first_doc_id for c in parts] == [0, 3, 6, 9]
    assert [p for c in parts for p in c.paths] == paths
    assert len(chunks(paths, 100)) == 10
    assert chunks(paths, 1) == [Chunk(0, tuple(paths))]
    assert chunks([], 4) == []


def test_serial_merge_of_one_partial_is_the_lab4_index(
    corpus_dir: Path, tmp_path: Path
) -> None:
    whole = Chunk(0, tuple(corpus_paths(corpus_dir)))

    index = merge([build_partial(whole)])

    assert saved_bytes(index, tmp_path, "one.bin") == lab4_bytes(corpus_dir, tmp_path)


@pytest.mark.parametrize(
    "executor",
    [
        pytest.param(name, marks=pytest.mark.slow) if name == "processes" else name
        for name in EXECUTOR_NAMES
    ],
)
@pytest.mark.parametrize("workers", [1, 3])
def test_every_executor_builds_the_same_bytes(
    executor, workers: int, corpus_dir: Path, tmp_path: Path
) -> None:
    report = build_parallel(corpus_paths(corpus_dir), executor, workers)

    assert saved_bytes(report.index, tmp_path, "p.bin") == lab4_bytes(
        corpus_dir, tmp_path
    )
    assert report.merge <= report.wall


def test_merge_does_not_care_in_which_order_chunks_finish(
    corpus_dir: Path, tmp_path: Path
) -> None:
    partials = [build_partial(c) for c in chunks(corpus_paths(corpus_dir), 4)]
    shuffled = partials[::-1]
    random.Random(1).shuffle(shuffled)

    assert saved_bytes(merge(shuffled), tmp_path, "m.bin") == lab4_bytes(
        corpus_dir, tmp_path
    )


def test_unreadable_file_keeps_everyone_elses_doc_ids(tmp_path: Path) -> None:
    paths = [tmp_path / "a.txt", tmp_path / "missing.txt", tmp_path / "c.txt"]
    paths[0].write_text("alpha", encoding="utf-8")
    paths[2].write_text("gamma", encoding="utf-8")

    index = build_parallel(paths, "threads", 3).index

    assert index.all_doc_ids() == [0, 2]
    assert list(index.doc_ids("gamma")) == [2]


@pytest.mark.parametrize(
    "executor",
    ["threads", pytest.param("processes", marks=pytest.mark.slow)],
)
def test_a_failing_worker_fails_the_build_with_its_traceback(executor) -> None:
    # A str instead of a Path: read_documents calls .read_text() on it, which
    # raises inside the worker, not in the parent.
    with pytest.raises(AttributeError, match="read_text") as caught:
        build_parallel(["oops.txt"], executor, 2)  # type: ignore[list-item]  # deliberately wrong

    if executor == "processes":
        # The worker's own traceback travels back as the exception's cause.
        assert "build_partial" in str(caught.value.__cause__)


def test_gil_enabled_is_a_bool() -> None:
    assert isinstance(gil_enabled(), bool)


def test_cli_index_with_threads(corpus_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "t.bin"

    result = CliRunner().invoke(
        app,
        [
            "index",
            str(corpus_dir),
            "-o",
            str(out),
            "--executor",
            "threads",
            "--workers",
            "2",
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert "threads, 2 workers" in result.stdout
    assert out.read_bytes() == lab4_bytes(corpus_dir, tmp_path)
