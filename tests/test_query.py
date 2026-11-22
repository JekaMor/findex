from pathlib import Path

import pytest

from findex.boolean import ENGINES
from findex.corpus import Document
from findex.index import build_index
from findex.query import And, Not, Or, Phrase, QueryError, Term, parse

a, b, c, d = Term("a"), Term("b"), Term("c"), Term("d")


# --- the parser: compare trees, not strings ------------------------------------


@pytest.mark.parametrize(
    ("query", "tree"),
    [
        ("a", a),
        ("a b", And(a, b)),
        ("a AND b", And(a, b)),
        ("a b c", And(And(a, b), c)),
        ("a OR b", Or(a, b)),
        ("a OR b c", Or(a, And(b, c))),  # AND binds tighter than OR
        ("a b OR c", Or(And(a, b), c)),
        ("(a OR b) c", And(Or(a, b), c)),
        ("NOT a", Not(a)),
        ("a NOT b", And(a, Not(b))),
        ("NOT NOT a", Not(Not(a))),
        ("NOT (a OR b)", Not(Or(a, b))),
        ("((a))", a),
        ('"event loop"', Phrase(("event", "loop"))),
        ('"a"', a),
        ("well-known", And(Term("well"), Term("known"))),
        ("Alice's CAT", And(Term("alice's"), Term("cat"))),
        ("a or b", And(And(a, Term("or")), b)),  # lower-case "or" is a word
        ("a !!! b", And(a, b)),  # a word with no tokens is dropped
        (
            'python AND (async OR await) NOT java "event loop"',
            And(
                And(
                    And(Term("python"), Or(Term("async"), Term("await"))),
                    Not(Term("java")),
                ),
                Phrase(("event", "loop")),
            ),
        ),
    ],
)
def test_parse(query, tree):
    assert parse(query) == tree


def test_operators_build_the_same_trees():
    assert a & b == And(a, b)
    assert a | b & c == parse("a OR b c")
    assert a & ~b == parse("a NOT b")
    assert ~(a | b) == parse("NOT (a OR b)")


def test_trees_are_hashable_and_printable():
    assert len({parse("a b"), parse("a  AND  b"), And(a, b)}) == 1
    assert str(parse('a OR b NOT "c d"')) == '(a OR (b AND NOT "c d"))'


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ("", "empty"),
        ("!!!", "empty"),
        ("a AND", "ends too early"),
        ("(a OR b", "missing )"),
        ("a)", "unexpected )"),
        ("()", "unexpected )"),
        ("a OR OR b", "unexpected OR"),
        ('"event loop', "unclosed quote"),
    ],
)
def test_parse_errors(query, message):
    with pytest.raises(QueryError, match=message.replace(")", r"\)")):
        parse(query)


def test_query_error_is_a_value_error():
    assert issubclass(QueryError, ValueError)


# --- evaluation -------------------------------------------------------------------

TEXTS = [
    "python event loop with async code",  # 0
    "java has an event loop too",  # 1
    "python await the loop event",  # 2  (words reversed: no phrase)
    "Кіт і пес",  # 3
]


@pytest.fixture(scope="module")
def index():
    docs = (Document(i, Path(f"d{i}.txt"), t) for i, t in enumerate(TEXTS))
    return build_index(docs, positions=True)


@pytest.mark.parametrize("engine", sorted(ENGINES))
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("python", [0, 2]),
        ("python loop", [0, 2]),
        ("python OR java", [0, 1, 2]),
        ("loop NOT python", [1]),
        ("NOT loop", [3]),
        ('"event loop"', [0, 1]),
        ('"loop event"', [2]),
        ('python "event loop"', [0]),
        ('"event loop" NOT java', [0]),
        ("python AND (async OR await) NOT java", [0, 2]),
        ('python AND (async OR await) NOT java "event loop"', [0]),
        ("кіт OR java", [1, 3]),
        ("unicorn", []),
        ('"no such phrase"', []),
    ],
)
def test_evaluate(index, engine, query, expected):
    assert parse(query).evaluate(index, engine) == expected


def test_phrase_needs_positions():
    plain = build_index([Document(0, Path("x"), "event loop")])

    with pytest.raises(QueryError, match="positions"):
        parse('"event loop"').evaluate(plain)


def test_terms_skip_negated_parts():
    tree = parse('python (async OR await) NOT java "event loop"')

    assert list(tree.terms()) == ["python", "async", "await", "event", "loop"]
