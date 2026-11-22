from findex.stats import eager_stats, index_stats, lazy_stats


def make_corpus(tmp_path):
    (tmp_path / "a.txt").write_text("the cat and the dog", encoding="utf-8")
    (tmp_path / "b.txt").write_text("The end. Кіт і кіт!", encoding="utf-8")
    return tmp_path


def test_lazy_stats_counts(tmp_path):
    stats = lazy_stats(make_corpus(tmp_path), top=2)

    assert stats.documents == 2
    assert stats.tokens == 10
    assert stats.vocabulary == 7
    assert stats.top[0] == ("the", 3)
    assert len(stats.top) == 2


def test_limit_takes_first_n_documents(tmp_path):
    stats = lazy_stats(make_corpus(tmp_path), limit=1)

    assert stats.documents == 1


def test_eager_matches_lazy(tmp_path):
    root = make_corpus(tmp_path)

    assert eager_stats(root, top=50) == lazy_stats(root, top=50)


def test_index_stats_match_corpus_stats(corpus_dir, corpus_index):
    from_corpus = lazy_stats(corpus_dir, top=5)
    from_index = index_stats(corpus_index, top=5)

    assert from_index.documents == from_corpus.documents
    assert from_index.tokens == from_corpus.tokens
    assert from_index.vocabulary == from_corpus.vocabulary
    assert dict(from_index.top) == dict(from_corpus.top)
