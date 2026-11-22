"""One spelling per URL, and which URLs the crawl may visit."""

from collections.abc import Iterable
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

_DEFAULT_PORTS = {"http": 80, "https": 443}
# Links that are surely not HTML pages; skipping them saves a request each.
_NOT_PAGES = frozenset(
    [
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".webp",
        ".ico",
        ".css",
        ".js",
        ".json",
        ".xml",
        ".pdf",
        ".zip",
        ".gz",
        ".bz2",
        ".xz",
        ".tar",
        ".tgz",
        ".whl",
        ".exe",
        ".msi",
        ".dmg",
        ".epub",
        ".mp3",
        ".mp4",
        ".woff",
        ".woff2",
        ".ttf",
        ".txt",
    ]
)


def canonical(url: str, base: str | None = None) -> str | None:
    """The canonical form of ``url`` (resolved against ``base``), or None if it is
    not an http(s) URL.

    - scheme and host lower case, default port dropped;
    - the ``#fragment`` dropped (same page);
    - an empty path becomes ``/``;
    - query parameters sorted, so ``?b=2&a=1`` and ``?a=1&b=2`` are one URL.

    A trailing slash is kept here, because this is the URL that gets fetched and
    ``/docs`` may well be a 404 where ``/docs/`` is a page. ``url_key`` is what
    treats the two as one for de-duplication.
    """
    absolute = urljoin(base, url.strip()) if base else url.strip()
    try:
        parts = urlsplit(absolute)
        port = parts.port
    except ValueError:  # e.g. a port that is not a number
        return None
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if scheme not in _DEFAULT_PORTS or not host:
        return None
    netloc = host if port in (None, _DEFAULT_PORTS[scheme]) else f"{host}:{port}"
    path = parts.path or "/"
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, query, ""))


def url_key(url: str) -> str:
    """De-duplication key of a canonical URL: ``/docs`` and ``/docs/`` are one page."""
    parts = urlsplit(url)
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def host_of(url: str) -> str:
    """``scheme://host[:port]``, the unit robots.txt and politeness work per."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def scope_of(seed: str) -> str:
    """The seed's directory: ``https://h/3/library/a.html`` -> ``https://h/3/library/``."""
    canon = canonical(seed) or seed
    head, _, _ = canon.partition("?")
    return head if head.endswith("/") else head.rsplit("/", 1)[0] + "/"


def in_scope(url: str, scopes: Iterable[str]) -> bool:
    """Is ``url`` under one of the ``scopes`` prefixes (or one of the scopes itself)?"""
    return any(url.startswith(s) or url + "/" == s for s in scopes)


def looks_like_page(url: str) -> bool:
    path = urlsplit(url).path.lower()
    dot = path.rfind(".")
    return dot <= path.rfind("/") or path[dot:] not in _NOT_PAGES
