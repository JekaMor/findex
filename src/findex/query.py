"""The query language: a recursive-descent parser building a tree of nodes.

Grammar, from the loosest operator to the tightest::

    or_expr  := and_expr ("OR" and_expr)*
    and_expr := not_expr (["AND"] not_expr)*      two atoms side by side mean AND
    not_expr := "NOT" not_expr | atom
    atom     := WORD | '"' words '"' | "(" or_expr ")"

So ``a OR b c`` is ``Or(a, And(b, c))``: AND binds tighter than OR, the way ``*``
binds tighter than ``+``. Operators must be upper case (``or`` is a word). Words
go through the corpus tokenizer, so ``Alice's`` finds ``alice's``; a word that
splits into several tokens (``well-known``) needs all of them.

Nodes can also be built without a string: ``Term("a") | Term("b") & ~Term("c")``.
"""

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING

from findex.boolean import MERGE, SET, Engine, EngineName, intersect
from findex.tokenize import tokenize

if TYPE_CHECKING:
    from findex.model import Index


class QueryError(ValueError):
    """The query string can't be parsed or evaluated."""


class Node:
    """A query tree node. Subclasses are frozen dataclasses: equal trees compare
    equal and hash equal, so a tree can be a cache key."""

    __slots__ = ()

    def __and__(self, other: object) -> Node:
        return And(self, other) if isinstance(other, Node) else NotImplemented

    def __or__(self, other: object) -> Node:
        return Or(self, other) if isinstance(other, Node) else NotImplemented

    def __invert__(self) -> Node:
        return Not(self)

    def evaluate(self, index: Index, engine: EngineName = "merge") -> list[int]:
        """Sorted doc_ids of the documents matching this query."""
        if engine == "set":
            return SET.result(self.operand(index, SET))
        return MERGE.result(self.operand(index, MERGE))

    def operand[T](self, index: Index, eng: Engine[T]) -> T:
        """This node's documents in the engine's own representation."""
        raise NotImplementedError

    def terms(self) -> Iterator[str]:
        """Terms that make a document match (not the ones under NOT)."""
        return iter(())


@dataclass(frozen=True, slots=True)
class Term(Node):
    text: str

    def __str__(self) -> str:
        return self.text

    def operand[T](self, index: Index, eng: Engine[T]) -> T:
        return eng.fetch(index.doc_ids(self.text))

    def terms(self) -> Iterator[str]:
        yield self.text


def _followed_by(first: Sequence[int], second: Sequence[int]) -> list[int]:
    """Positions in ``second`` that come right after a position in ``first``."""
    out: list[int] = []
    i = j = 0
    while i < len(first) and j < len(second):
        want = first[i] + 1
        if want == second[j]:
            out.append(second[j])
            i += 1
            j += 1
        elif want < second[j]:
            i += 1
        else:
            j += 1
    return out


@dataclass(frozen=True, slots=True)
class Phrase(Node):
    words: tuple[str, ...]

    def __str__(self) -> str:
        return '"' + " ".join(self.words) + '"'

    def operand[T](self, index: Index, eng: Engine[T]) -> T:
        if not index.positions:
            raise QueryError("phrase queries need an index built with positions")
        # Only documents having every word can have the phrase.
        candidates = reduce(intersect, (index.doc_ids(w) for w in self.words))
        wanted = set(candidates)
        where = [
            {p.doc_id: p.positions for p in index.get(w, ()) if p.doc_id in wanted}
            for w in self.words
        ]
        found: list[int] = []
        for doc_id in candidates:
            ends: Sequence[int] = where[0][doc_id]
            for positions in where[1:]:
                ends = _followed_by(ends, positions[doc_id])
                if not ends:
                    break
            if ends:
                found.append(doc_id)
        return eng.fetch(found)

    def terms(self) -> Iterator[str]:
        yield from self.words


@dataclass(frozen=True, slots=True)
class And(Node):
    left: Node
    right: Node

    def __str__(self) -> str:
        return f"({self.left} AND {self.right})"

    def operand[T](self, index: Index, eng: Engine[T]) -> T:
        left = self.left.operand(index, eng)
        if isinstance(self.right, Not):  # a AND NOT b: subtract b, no complement needed
            return eng.not_(left, self.right.child.operand(index, eng))
        return eng.and_(left, self.right.operand(index, eng))

    def terms(self) -> Iterator[str]:
        yield from self.left.terms()
        yield from self.right.terms()


@dataclass(frozen=True, slots=True)
class Or(Node):
    left: Node
    right: Node

    def __str__(self) -> str:
        return f"({self.left} OR {self.right})"

    def operand[T](self, index: Index, eng: Engine[T]) -> T:
        return eng.or_(self.left.operand(index, eng), self.right.operand(index, eng))

    def terms(self) -> Iterator[str]:
        yield from self.left.terms()
        yield from self.right.terms()


@dataclass(frozen=True, slots=True)
class Not(Node):
    child: Node

    def __str__(self) -> str:
        return f"NOT {self.child}"

    def operand[T](self, index: Index, eng: Engine[T]) -> T:
        everything = eng.fetch(index.all_doc_ids())
        return eng.not_(everything, self.child.operand(index, eng))


# --- lexer ---------------------------------------------------------------------

type Token = tuple[
    str, Node | None
]  # ("(" | ")" | "AND" | "OR" | "NOT" | "ATOM", node)

_LEXEME = re.compile(
    r'\s*(?:(?P<paren>[()])|"(?P<phrase>[^"]*)(?P<closed>"?)|(?P<word>[^\s()"]+))'
)
_OPERATORS = frozenset({"AND", "OR", "NOT"})


def _atom(tokens: Sequence[str], phrase: bool) -> Node | None:
    if not tokens:
        return None
    if len(tokens) == 1:
        return Term(tokens[0])
    if phrase:
        return Phrase(tuple(tokens))
    return reduce(And, map(Term, tokens))


def _lex(query: str) -> Iterator[Token]:
    pos = 0
    query = query.rstrip()
    while pos < len(query):
        m = _LEXEME.match(query, pos)
        if m is None:  # pragma: no cover - the pattern accepts any non-space char
            raise QueryError(f"can't read the query at {query[pos:]!r}")
        pos = m.end()
        if m["paren"]:
            yield m["paren"], None
        elif m["phrase"] is not None:
            if not m["closed"]:
                raise QueryError("unclosed quote")
            node = _atom(list(tokenize(m["phrase"])), phrase=True)
            if node is not None:
                yield "ATOM", node
        elif m["word"] in _OPERATORS:
            yield m["word"], None
        else:
            node = _atom(list(tokenize(m["word"])), phrase=False)
            if node is not None:  # "!!!" has no tokens: drop it
                yield "ATOM", node


# --- parser: one method per grammar rule -----------------------------------------


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos][0] if self.pos < len(self.tokens) else None

    def take(self) -> Token:
        if self.pos == len(self.tokens):
            raise QueryError("the query ends too early")
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def or_expr(self) -> Node:
        node = self.and_expr()
        while self.peek() == "OR":
            self.take()
            node = Or(node, self.and_expr())
        return node

    def and_expr(self) -> Node:
        node = self.not_expr()
        while self.peek() not in (None, "OR", ")"):
            if self.peek() == "AND":
                self.take()
            node = And(node, self.not_expr())
        return node

    def not_expr(self) -> Node:
        if self.peek() == "NOT":
            self.take()
            return Not(self.not_expr())
        return self.atom()

    def atom(self) -> Node:
        kind, node = self.take()
        if kind == "ATOM" and node is not None:
            return node
        if kind == "(":
            inner = self.or_expr()
            if self.peek() != ")":
                raise QueryError("missing )")
            self.take()
            return inner
        raise QueryError(f"unexpected {kind}")


def parse(query: str) -> Node:
    tokens = list(_lex(query))
    if not tokens:
        raise QueryError("empty query")
    parser = _Parser(tokens)
    node = parser.or_expr()
    if parser.peek() is not None:
        raise QueryError(f"unexpected {parser.peek()}")
    return node
