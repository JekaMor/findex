import inspect
import unicodedata

import pytest

from findex.tokenize import token_spans, tokenize


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        pytest.param("", [], id="empty"),
        pytest.param("   \n\t ", [], id="only-whitespace"),
        pytest.param("!?…—", [], id="only-punctuation"),
        pytest.param(
            "Hello, World! It works.",
            ["hello", "world", "it", "works"],
            id="mixed-case",
        ),
        pytest.param(
            "Привіт, СВІТЕ! Це тест.", ["привіт", "світе", "це", "тест"], id="cyrillic"
        ),
        pytest.param("Straße STRASSE", ["strasse", "strasse"], id="casefold-not-lower"),
        pytest.param("café", ["café"], id="combining-mark"),
        pytest.param("don't п'ять", ["don't", "п'ять"], id="apostrophe-inside"),
        pytest.param("don’t п’ять", ["don't", "п'ять"], id="typographic-apostrophe"),
        pytest.param("'quoted' words'", ["quoted", "words"], id="apostrophe-at-edges"),
        pytest.param(
            "well-known state-of-the-art",
            ["well", "known", "state", "of", "the", "art"],
            id="hyphen-splits",
        ),
        pytest.param(
            "Python 3.14 in 2026, abc123",
            ["python", "3", "14", "in", "2026", "abc123"],
            id="digits",
        ),
        pytest.param("snake_case", ["snake", "case"], id="underscore-splits"),
        pytest.param("東京 タワー", ["東京", "タワー"], id="cjk"),
        pytest.param("ΣΊΣΥΦΟΣ", ["σίσυφοσ"], id="greek-final-sigma-casefolded"),
    ],
)
def test_tokenize(text: str, tokens: list[str]) -> None:
    assert list(tokenize(text)) == tokens


def test_combining_mark_matches_precomposed() -> None:
    (token,) = tokenize("café")
    assert token == next(tokenize("café"))
    assert unicodedata.is_normalized("NFC", token)


def test_is_lazy() -> None:
    assert inspect.isgeneratorfunction(tokenize)
    gen = tokenize("a b c")
    assert next(gen) == "a"


def test_token_spans_agree_with_tokenize() -> None:
    text = "Straße, don’t п'ять well-known 3.14 café “Quoted”"

    spans = list(token_spans(text))

    assert [t for t, _, _ in spans] == list(tokenize(text))
    assert spans[0] == ("strasse", 0, 6)
