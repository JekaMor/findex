"""Boolean set operations on sorted doc_id lists, and the two engines using them."""

import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

# --- two-pointer merges over lists sorted by doc_id: O(len(a) + len(b)) ------


def intersect(a: Sequence[int], b: Sequence[int]) -> list[int]:
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        x, y = a[i], b[j]
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            i += 1
        else:
            j += 1
    return out


def union(a: Sequence[int], b: Sequence[int]) -> list[int]:
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        x, y = a[i], b[j]
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            out.append(x)
            i += 1
        else:
            out.append(y)
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def difference(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Items of ``a`` that are not in ``b``."""
    out: list[int] = []
    i = j = 0
    while i < len(a):
        if j == len(b) or a[i] < b[j]:
            out.append(a[i])
            i += 1
        elif a[i] == b[j]:
            i += 1
            j += 1
        else:
            j += 1
    return out


# --- engines -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Engine:
    """How to turn postings into an operand and combine operands."""

    fetch: Callable[[Sequence[int]], Any]
    and_: Callable[[Any, Any], Any]
    or_: Callable[[Any, Any], Any]
    not_: Callable[[Any, Any], Any]
    result: Callable[[Any], list[int]]


ENGINES: dict[str, Engine] = {
    # Works on the sorted lists as they are; every result stays sorted.
    "merge": Engine(lambda ids: ids, intersect, union, difference, list),
    # Hashes each postings list into a set first, sorts once at the end.
    "set": Engine(set, operator.and_, operator.or_, operator.sub, sorted),
}
