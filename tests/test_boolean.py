import random

from findex.boolean import difference, intersect, union


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
