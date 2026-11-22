from findex.snippets import snippet


def test_marks_terms_and_trims_with_ellipses():
    text = "word " * 50 + "the White Whale swam" + " word" * 50

    out = snippet(text, {"white", "whale"}, radius=30)

    assert "**White** **Whale**" in out
    assert out.startswith("…") and out.endswith("…")
    assert len(out) < 100


def test_short_text_has_no_ellipses():
    assert snippet("A white whale.", {"whale"}) == "A white **whale**."


def test_window_prefers_more_distinct_terms():
    text = "whale " + "x " * 200 + "white whale here " + "y " * 200 + "whale"

    out = snippet(text, {"white", "whale"}, radius=20)

    assert "**white** **whale** here" in out


def test_whitespace_is_collapsed():
    assert (
        snippet("one\n\n  two   whale\nthree", {"whale"}) == "one two **whale** three"
    )


def test_does_not_cut_words_at_the_edges():
    text = "abcdefghij " * 20 + "whale" + " abcdefghij" * 20

    out = snippet(text, {"whale"}, radius=25)

    assert all(
        w in {"abcdefghij", "**whale**", "…"} for w in out.replace("…", " … ").split()
    )


def test_no_match_shows_the_start():
    assert snippet("Just some text here.", {"whale"}) == "Just some text here."
    assert snippet("x" * 500, {"whale"}, radius=10) == "x" * 20 + "…"
