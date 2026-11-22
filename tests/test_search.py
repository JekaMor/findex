import random
from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import build_index
from findex.search import ENGINES, difference, intersect, main, parse, search, union
from findex.store import save

TEXTS = [
    "Title: Cats\nthe cat sat",  # 0
    "the dog ran",  # 1
    "cat and dog",  # 2
    "Кіт і пес",  # 3
]


@pytest.fixture(scope="module")
def index():
    return build_index(Document(i, Path(f"d{i}.txt"), t) for i, t in enumerate(TEXTS))


def test_merges_on_examples():
    a, b = [1, 3, 5, 7], [3, 4, 5, 8]

    assert intersect(a, b) == [3, 5]
    assert union(a, b) == [1, 3, 4, 5, 7, 8]
    assert difference(a, b) == [1, 7]
    assert intersect([], b) == union([], []) == difference([], b) == []


def test_merges_agree_with_sets():
    rng = random.Random(0)
    for _ in range(200):
        a = sorted(rng.sample(range(60), rng.randint(0, 30)))
        b = sorted(rng.sample(range(60), rng.randint(0, 30)))
        assert intersect(a, b) == sorted(set(a) & set(b))
        assert union(a, b) == sorted(set(a) | set(b))
        assert difference(a, b) == sorted(set(a) - set(b))


def test_parse():
    assert parse("Cat dog") == [("AND", ["cat"]), ("AND", ["dog"])]
    assert parse("a OR b NOT c") == [("AND", ["a"]), ("OR", ["b"]), ("NOT", ["c"])]
    assert parse("well-known") == [("AND", ["well", "known"])]
    assert parse("a or b") == [("AND", ["a"]), ("AND", ["or"]), ("AND", ["b"])]


@pytest.mark.parametrize("engine", sorted(ENGINES))
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("cat", [0, 2]),
        ("CAT dog", [2]),
        ("cat AND dog", [2]),
        ("cat OR dog", [0, 1, 2]),
        ("dog NOT cat", [1]),
        ("NOT the", [2, 3]),
        ("кіт", [3]),
        ("cat OR кіт NOT dog", [0, 3]),
        ("unicorn", []),
        ("cat unicorn", []),
        ("", []),
    ],
)
def test_search(index, engine, query, expected):
    assert search(index, query, engine) == expected


def test_main_searches_a_saved_index(index, tmp_path, capsys):
    path = tmp_path / "index.bin"
    save(index, path)

    main([str(path), "cat", "--engine", "set"])

    out = capsys.readouterr().out
    for needle in ("hits:         2", "Cats  (d0.txt)", "set", "load", "peak memory"):
        assert needle in out
