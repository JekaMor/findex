"""The ``findex`` command, driven through typer's CliRunner (stdout and stderr apart)."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.store import load

runner = CliRunner()


@pytest.fixture(autouse=True)
def run_from_corpus_parent(corpus_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Indexed paths are relative to where `findex index` ran; snippets read them.
    monkeypatch.chdir(corpus_dir.parent)


def test_help_lists_the_three_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("index", "search", "stats"):
        assert command in result.stdout


def test_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.startswith("findex 0.")


def test_index_builds_a_file(corpus_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "built.bin"

    result = runner.invoke(app, ["index", corpus_dir.name, "--out", str(out)])

    assert result.exit_code == 0, result.stderr
    assert out.is_file()
    assert "documents" in result.stdout
    assert "positions" in result.stdout


def test_index_progress_and_logs_go_to_stderr(corpus_dir: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["-v", "index", corpus_dir.name, "--out", str(tmp_path / "i.bin")]
    )

    assert result.exit_code == 0
    assert "indexing 4 documents" in result.stderr
    assert "indexing" not in result.stdout


def test_index_limit_and_pickle(corpus_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "two.pkl"

    result = runner.invoke(
        app, ["index", corpus_dir.name, "-o", str(out), "--limit", "2"]
    )

    assert result.exit_code == 0
    assert "pickle" in result.stdout
    assert load(out).num_docs == 2


def _search(*args: str) -> tuple[int, str, str]:
    result = runner.invoke(app, ["search", *args])
    return result.exit_code, result.stdout, result.stderr


def test_search_prints_a_ranked_table(index_file: Path) -> None:
    code, stdout, _ = _search(str(index_file), "whale")

    assert code == 0
    assert "The Whale" in stdout
    assert "score" in stdout
    assert "Ishmael" in stdout  # the snippet


def test_search_json_lines_are_pure_data(index_file: Path) -> None:
    code, stdout, stderr = _search(
        str(index_file), "whale OR sea", '"dead man"', "--json", "-k", "5"
    )

    assert code == 0
    records = [json.loads(line) for line in stdout.splitlines()]
    assert [r["query"] for r in records] == ["whale OR sea"] * 2 + ['"dead man"']
    assert records[0]["rank"] == 1
    assert records[0]["score"] >= records[1]["score"]
    assert records[2]["title"] == "Pirates"
    assert "dead man" in records[2]["snippet"]
    assert stderr == ""


def test_search_scorer_and_no_snippets(index_file: Path) -> None:
    code, stdout, _ = _search(
        str(index_file), "sea", "--scorer", "tfidf", "--no-snippets", "--json"
    )

    assert code == 0
    (record,) = map(json.loads, stdout.splitlines())
    assert record["snippet"] is None


def test_search_with_nothing_found(index_file: Path) -> None:
    code, stdout, _ = _search(str(index_file), "unicorn")

    assert code == 0
    assert "nothing found" in stdout


def test_repeated_query_is_a_cache_hit_in_the_debug_log(index_file: Path) -> None:
    result = runner.invoke(
        app, ["-vv", "search", str(index_file), "whale", "whale", "--json"]
    )

    assert result.exit_code == 0
    assert "computed" in result.stderr
    assert "cache hit" in result.stderr
    assert "took" in result.stderr  # @timed at DEBUG
    assert "cache hit" not in result.stdout


@pytest.mark.parametrize(
    ("args", "code", "message"),
    [
        (["search", "missing.bin", "whale"], 1, "index file not found: missing.bin"),
        (
            ["search", "INDEX", "whale AND (sea"],
            2,
            "bad query 'whale AND (sea': missing )",
        ),
        (["search", "INDEX", '"open quote'], 2, "unclosed quote"),
        (["search", "NOT_AN_INDEX", "whale"], 1, "is not a findex index file"),
        (["index", "no-such-dir"], 1, "corpus directory not found: no-such-dir"),
        (["stats", "nowhere"], 1, "no such index file or corpus directory: nowhere"),
        (["stats", "NOT_AN_INDEX"], 1, "is not a findex index file"),
    ],
)
def test_expected_errors_are_one_line_and_nonzero(
    args: list[str], code: int, message: str, index_file: Path, tmp_path: Path
) -> None:
    junk = tmp_path / "junk.txt"
    junk.write_text("not an index", encoding="utf-8")
    replace = {"INDEX": str(index_file), "NOT_AN_INDEX": str(junk)}
    args = [replace.get(a, a) for a in args]

    result = runner.invoke(app, args)

    assert result.exit_code == code
    assert result.stdout == ""
    assert message in result.stderr
    assert len(result.stderr.strip().splitlines()) == 1
    assert "Traceback" not in result.stderr


def test_phrase_on_index_without_positions_is_an_error(
    corpus_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "flat.bin"
    runner.invoke(app, ["index", corpus_dir.name, "-o", str(out), "--no-positions"])

    code, _, stderr = _search(str(out), '"dead man"')

    assert code == 2
    assert "positions" in stderr


def test_stats_from_index(index_file: Path) -> None:
    result = runner.invoke(app, ["stats", str(index_file), "--top", "3"])

    assert result.exit_code == 0
    for needle in ("documents", "tokens", "vocabulary", "top 3 terms", "the"):
        assert needle in result.stdout


@pytest.mark.parametrize("mode", [[], ["--eager"]])
def test_stats_from_corpus_directory(corpus_dir: Path, mode: list[str]) -> None:
    result = runner.invoke(app, ["stats", corpus_dir.name, *mode])

    assert result.exit_code == 0
    assert ("eager" if mode else "lazy") in result.stdout
    assert "peak memory" in result.stdout


def test_missing_document_file_gives_a_dash_and_a_warning(
    corpus_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "relative.bin"
    runner.invoke(app, ["index", corpus_dir.name, "-o", str(out)])  # relative paths
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # ...which no longer resolve from here

    result = runner.invoke(app, ["search", str(out), "whale"])

    assert result.exit_code == 0
    assert "The Whale" in result.stdout
    assert "—" in result.stdout  # an em dash instead of the snippet
    assert "no snippet for" in result.stderr


def test_table_without_snippets_has_no_snippet_column(index_file: Path) -> None:
    code, stdout, _ = _search(str(index_file), "whale", "--no-snippets")

    assert code == 0
    assert "The Whale" in stdout
    assert "snippet" not in stdout
    assert "Ishmael" not in stdout
