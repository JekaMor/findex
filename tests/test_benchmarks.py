"""pytest-benchmark: search and build_index on the fixture corpus.

A normal ``pytest`` run executes each of these once, untimed (``--benchmark-disable``
in pyproject.toml). CI runs them timed and publishes the table:

    pytest tests/test_benchmarks.py --benchmark-enable --benchmark-only

The generous ceilings below turn a large slowdown (a Python loop sneaking back
into the hot path, say) into a failing test, on any machine.
"""

from pathlib import Path

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from findex.corpus import corpus_paths, read_documents
from findex.index import build_index
from findex.model import Index
from findex.query import parse
from findex.scoring import BM25
from findex.search import rank_numpy, rank_python, search

QUERY = "sea OR ship OR captain OR the"


def ceiling(benchmark: BenchmarkFixture, seconds: float) -> None:
    stats = benchmark.stats
    if stats is not None:  # None when benchmarks are disabled
        assert stats.stats.median < seconds


@pytest.mark.benchmark(group="search")
def test_bench_search(benchmark: BenchmarkFixture, corpus_index: Index) -> None:
    results = benchmark(search, corpus_index, QUERY, k=10)

    assert results
    ceiling(benchmark, 0.005)


@pytest.mark.benchmark(group="search")
def test_bench_rank_numpy(benchmark: BenchmarkFixture, corpus_index: Index) -> None:
    node = parse(QUERY)

    benchmark(rank_numpy, corpus_index, node, BM25(), 10)
    ceiling(benchmark, 0.005)


@pytest.mark.benchmark(group="search")
def test_bench_rank_python_lab3(
    benchmark: BenchmarkFixture, corpus_index: Index
) -> None:
    node = parse(QUERY)

    benchmark(rank_python, corpus_index, node, BM25(), 10)
    ceiling(benchmark, 0.005)


@pytest.mark.benchmark(group="build")
def test_bench_build_index(benchmark: BenchmarkFixture, corpus_dir: Path) -> None:
    paths = corpus_paths(corpus_dir)

    index = benchmark(
        lambda: build_index(read_documents(paths), "array", positions=True)
    )

    assert index.num_docs == 4
    ceiling(benchmark, 0.05)
