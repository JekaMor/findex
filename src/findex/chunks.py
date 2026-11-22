"""Passages: documents cut into pieces of about ``words`` words.

Semantic search embeds passages, not whole books (one vector for 500,000 words
means nothing); the same passages make a many-document corpus for benchmarks.
Cuts fall on paragraph breaks when there is one near, otherwise between words.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass

_PARAGRAPH = re.compile(r"\n\s*\n")
_WORD = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class Passage:
    doc_id: int
    number: int  # 0, 1, 2 ... within the document
    start: int  # character offsets into the document's text
    end: int
    text: str


def passages(doc_id: int, text: str, words: int = 200) -> Iterator[Passage]:
    """Consecutive passages of ``text``, each about ``words`` words long.

    Paragraphs are packed together until the next one would overflow the size;
    a paragraph longer than ``words`` is cut between words.
    """
    number = 0
    start = end = None
    count = 0
    for para in _paragraphs(text):
        n = len(_WORD.findall(text, para[0], para[1]))
        if start is not None and count + n > words:
            yield Passage(doc_id, number, start, end or start, text[start:end].strip())
            number += 1
            start, count = None, 0
        if n > words:  # a giant paragraph: cut it by words
            for piece_start, piece_end in _word_pieces(text, para[0], para[1], words):
                yield Passage(
                    doc_id, number, piece_start, piece_end, text[piece_start:piece_end]
                )
                number += 1
            continue
        if start is None:
            start = para[0]
        end = para[1]
        count += n
    if start is not None and count:
        yield Passage(doc_id, number, start, end or start, text[start:end].strip())


def _paragraphs(text: str) -> Iterator[tuple[int, int]]:
    pos = 0
    for gap in _PARAGRAPH.finditer(text):
        if gap.start() > pos:
            yield pos, gap.start()
        pos = gap.end()
    if pos < len(text):
        yield pos, len(text)


def _word_pieces(
    text: str, start: int, end: int, words: int
) -> Iterator[tuple[int, int]]:
    spans = [(m.start(), m.end()) for m in _WORD.finditer(text, start, end)]
    for i in range(0, len(spans), words):
        group = spans[i : i + words]
        yield group[0][0], group[-1][1]
