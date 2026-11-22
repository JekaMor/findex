"""Title, visible text and links of an HTML page, with the standard ``html.parser``.

This is CPU work (a large page takes tens of milliseconds), so the crawler runs
it in a worker thread with ``asyncio.to_thread``, never on the event loop.
"""

from dataclasses import dataclass
from html.parser import HTMLParser

_SKIP = frozenset({"script", "style", "noscript", "template", "svg", "head", "nav"})
_VOID = frozenset(
    [
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "source",
        "wbr",
    ]
)
_BLOCK = frozenset(
    [
        "p",
        "div",
        "br",
        "li",
        "ul",
        "ol",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "tr",
        "td",
        "th",
        "pre",
        "section",
        "article",
        "header",
        "footer",
        "nav",
        "dt",
        "dd",
        "table",
        "blockquote",
    ]
)


@dataclass(frozen=True, slots=True)
class ParsedPage:
    title: str
    text: str
    links: tuple[str, ...]  # href values as written, not yet resolved
    nofollow: bool  # <meta name="robots" content="nofollow">


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: list[str] = []
        self.text: list[str] = []
        self.links: list[str] = []
        self.main_text: list[str] = []  # inside <main> or role="main", if any
        self.nofollow = False
        self._skip = 0  # depth inside script/style/...
        self._main: list[str] = []  # the tag that opened the main content
        self._depth: list[str] = []  # tags opened inside it, to find its end
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name: value or "" for name, value in attrs}
        if tag == "a" and values.get("href"):
            self.links.append(values["href"])
        elif tag == "title":
            self._in_title = True
        elif tag == "meta" and values.get("name", "").lower() == "robots":
            self.nofollow |= "nofollow" in values.get("content", "").lower()
        if not self._main and (tag == "main" or values.get("role") == "main"):
            self._main.append(tag)
        elif self._main and tag not in _VOID:
            self._depth.append(tag)
        if tag in _SKIP:
            self._skip += 1
        elif tag in _BLOCK:
            self._add("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in _SKIP and self._skip:
            self._skip -= 1
        elif tag in _BLOCK:
            self._add("\n")
        if self._main:
            if self._depth and self._depth[-1] == tag:
                self._depth.pop()
            elif not self._depth and tag == self._main[0]:
                self._main.clear()  # main content ended: stop collecting it

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title.append(data)
        elif not self._skip:
            self._add(data)

    def _add(self, text: str) -> None:
        self.text.append(text)
        if self._main:
            self.main_text.append(text)


def parse_html(html: str) -> ParsedPage:
    extractor = _Extractor()
    extractor.feed(html)
    extractor.close()
    # The page's main content if it marks one (Sphinx and most CMSes do): menus,
    # sidebars and footers would otherwise show up in every page's text.
    main = "".join(extractor.main_text)
    body = main if main.strip() else "".join(extractor.text)
    lines = (" ".join(line.split()) for line in body.splitlines())
    return ParsedPage(
        title=" ".join("".join(extractor.title).split()),
        text="\n".join(line for line in lines if line),
        links=tuple(extractor.links),
        nofollow=extractor.nofollow,
    )
