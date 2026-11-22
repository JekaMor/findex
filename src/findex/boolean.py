"""Boolean set operations on sorted doc_id lists, and the two engines using them."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal

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

type EngineName = Literal["merge", "set"]
ENGINE_NAMES: Final[tuple[EngineName, ...]] = ("merge", "set")


@dataclass(frozen=True, slots=True)
class Engine[T]:
    """How to turn postings into an operand of type ``T`` and combine operands."""

    fetch: Callable[[Sequence[int]], T]
    and_: Callable[[T, T], T]
    or_: Callable[[T, T], T]
    not_: Callable[[T, T], T]
    result: Callable[[T], list[int]]


def _as_is(ids: Sequence[int]) -> Sequence[int]:
    return ids


def _as_set(ids: Sequence[int]) -> set[int]:
    return set(ids)


def _set_and(a: set[int], b: set[int]) -> set[int]:
    return a & b


def _set_or(a: set[int], b: set[int]) -> set[int]:
    return a | b


def _set_sub(a: set[int], b: set[int]) -> set[int]:
    return a - b


# Works on the sorted lists as they are; every result stays sorted.
MERGE: Final = Engine[Sequence[int]](_as_is, intersect, union, difference, list)
# Hashes each postings list into a set first, sorts once at the end.
SET: Final = Engine[set[int]](_as_set, _set_and, _set_or, _set_sub, sorted)
