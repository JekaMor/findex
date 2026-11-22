"""The crawler against fake sites built with httpx.MockTransport: no network."""

import asyncio
import contextlib
import email.utils
import json
import logging
from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from functools import partial
from itertools import pairwise
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.crawler import USER_AGENT, Backoff, CrawlStats, RobotsCache, crawl, fetch
from findex.crawler.html import parse_html
from findex.crawler.urls import canonical, in_scope, looks_like_page, scope_of, url_key

type Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]


def client_for(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
        max_redirects=5,
        headers={"User-Agent": USER_AGENT},
    )


class Sleeps:
    """Stands in for asyncio.sleep in fetch(): records delays, waits for nothing."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def run_fetch(handler: Handler, url: str = "https://site.test/", **kwargs) -> tuple:
    sleeps = Sleeps()

    async def go():
        async with client_for(handler) as client:
            return await fetch(client, url, sleep=sleeps, rng=lambda: 0.0, **kwargs)

    return asyncio.run(go()), sleeps.delays


def responses(
    *answers: httpx.Response | Exception,
) -> tuple[Handler, list[httpx.Request]]:
    """A handler giving ``answers`` in order (the last one repeats)."""
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        answer = answers[min(len(seen), len(answers)) - 1]
        if isinstance(answer, Exception):
            raise answer
        return answer

    return handler, seen


# --- fetch ------------------------------------------------------------------------


def test_fetch_records_status_bytes_and_time() -> None:
    handler, seen = responses(
        httpx.Response(200, text="hello", headers={"content-type": "text/html"})
    )

    result, delays = run_fetch(handler)

    assert (result.status, result.body, result.attempts) == (200, b"hello", 1)
    assert result.ok and result.content_type == "text/html"
    assert result.elapsed >= 0
    assert delays == []
    assert seen[0].headers["user-agent"] == USER_AGENT


def test_404_is_not_retried() -> None:
    handler, seen = responses(httpx.Response(404))

    result, delays = run_fetch(handler)

    assert (result.status, result.attempts, len(seen), delays) == (404, 1, 1, [])
    assert not result.ok


def test_503_then_200_retries_with_exponential_backoff() -> None:
    handler, seen = responses(
        httpx.Response(503), httpx.Response(503), httpx.Response(200)
    )

    result, delays = run_fetch(handler)

    assert (result.status, result.attempts, len(seen)) == (200, 3, 3)
    assert delays == [0.5, 1.0]  # base 0.5 * 2**attempt, jitter 0 here


def test_gives_up_after_three_retries() -> None:
    handler, seen = responses(httpx.Response(500))

    result, delays = run_fetch(handler)

    assert (result.status, result.attempts, len(seen)) == (500, 4, 4)
    assert result.error == "HTTP 500"
    assert delays == [0.5, 1.0, 2.0]


def test_429_waits_as_long_as_retry_after_says() -> None:
    handler, _ = responses(
        httpx.Response(429, headers={"retry-after": "3"}), httpx.Response(200)
    )

    result, delays = run_fetch(handler)

    assert result.status == 200
    assert delays == [3.0]


def test_retry_after_as_an_http_date() -> None:
    when = email.utils.format_datetime(
        datetime.now(UTC) + timedelta(seconds=30), usegmt=True
    )
    handler, _ = responses(
        httpx.Response(503, headers={"retry-after": when}), httpx.Response(200)
    )

    _, delays = run_fetch(handler)

    assert 25 < delays[0] <= 30


def test_retry_after_too_long_gives_up() -> None:
    handler, seen = responses(httpx.Response(429, headers={"retry-after": "3600"}))

    result, delays = run_fetch(handler)

    assert (result.status, len(seen), delays) == (429, 1, [])


def test_timeout_is_retried() -> None:
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1)
        return httpx.Response(200)

    result, delays = run_fetch(slow, timeout=0.02, retries=2)

    assert result.status is None
    assert result.error == "timeout after 0.02 s"
    assert result.attempts == 3
    assert len(delays) == 2


def test_connection_error_is_not_retried() -> None:
    handler, seen = responses(httpx.ConnectError("refused"))

    result, delays = run_fetch(handler)

    assert result.status is None and len(seen) == 1 and delays == []
    assert result.error and "ConnectError" in result.error


def test_redirect_loop_is_reported() -> None:
    async def loop(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "/again"})

    result, _ = run_fetch(loop)

    assert result.error == "too many redirects"


def test_backoff_doubles_up_to_the_cap_and_adds_jitter() -> None:
    backoff = Backoff(base=0.5, cap=4.0, jitter=1.0)

    assert [backoff.delay(a, rng=lambda: 0.0) for a in range(5)] == [0.5, 1, 2, 4, 4]
    assert backoff.delay(0, rng=lambda: 0.5) == 1.0


# --- robots.txt ----------------------------------------------------------------------


def robots_site(robots: httpx.Response) -> tuple[Handler, Counter[str]]:
    hits: Counter[str] = Counter()

    async def handler(request: httpx.Request) -> httpx.Response:
        hits[request.url.path] += 1
        if request.url.path == "/robots.txt":
            await asyncio.sleep(0.01)  # let concurrent askers pile up
            return robots
        return httpx.Response(200, text="page")

    return handler, hits


def ask(handler: Handler, *urls: str) -> list[bool]:
    async def go():
        async with client_for(handler) as client:
            robots = RobotsCache(client, retries=0)  # a 5xx is not retried here
            return list(await asyncio.gather(*(robots.allowed(u) for u in urls)))

    return asyncio.run(go())


def test_robots_is_fetched_once_per_host_and_obeyed() -> None:
    rules = "User-agent: *\nDisallow: /private/\n"
    handler, hits = robots_site(httpx.Response(200, text=rules))

    answers = ask(
        handler, "https://a.test/x", "https://a.test/private/y", "https://a.test/z"
    )

    assert answers == [True, False, True]
    assert hits["/robots.txt"] == 1


def test_missing_robots_allows_everything() -> None:
    handler, _ = robots_site(httpx.Response(404))

    assert ask(handler, "https://a.test/private/y") == [True]


def test_broken_robots_allows_nothing() -> None:
    handler, _ = robots_site(httpx.Response(500))

    assert ask(handler, "https://a.test/") == [False]


def test_crawl_delay_is_read() -> None:
    handler, _ = robots_site(
        httpx.Response(200, text="User-agent: *\nCrawl-delay: 2\n")
    )

    async def go():
        async with client_for(handler) as client:
            return await RobotsCache(client).crawl_delay("https://a.test/")

    assert asyncio.run(go()) == 2.0


# --- URLs and HTML ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("HTTPS://Docs.Python.ORG/3/Library/", "https://docs.python.org/3/Library/"),
        ("https://h.test", "https://h.test/"),
        ("https://h.test:443/a#section", "https://h.test/a"),
        ("http://h.test:8080/a/", "http://h.test:8080/a/"),
        ("https://h.test/s?b=2&a=1&a=0", "https://h.test/s?a=0&a=1&b=2"),
        ("mailto:someone@h.test", None),
        ("javascript:void(0)", None),
        ("https://h.test:notaport/", None),
    ],
)
def test_canonical(url: str, expected: str | None) -> None:
    assert canonical(url) == expected


def test_canonical_resolves_relative_links() -> None:
    base = "https://h.test/3/library/asyncio.html"

    assert canonical("../tutorial/#x", base) == "https://h.test/3/tutorial/"
    assert (
        canonical("asyncio-task.html", base)
        == "https://h.test/3/library/asyncio-task.html"
    )


def test_url_key_treats_a_trailing_slash_as_the_same_page() -> None:
    assert url_key("https://h.test/docs/") == url_key("https://h.test/docs")
    assert url_key("https://h.test/") == "https://h.test/"
    assert url_key("https://h.test/a/?q=1") == "https://h.test/a?q=1"


def test_scope_and_page_filters() -> None:
    scope = scope_of("https://h.test/3/library/index.html")

    assert scope == "https://h.test/3/library/"
    assert in_scope("https://h.test/3/library/os.html", [scope])
    assert in_scope("https://h.test/3/library", [scope])
    assert not in_scope("https://h.test/3/tutorial/", [scope])
    assert looks_like_page("https://h.test/a/b.html")
    assert looks_like_page("https://h.test/a/b")
    assert not looks_like_page("https://h.test/_static/logo.png")


def test_parse_html_takes_title_text_and_links() -> None:
    page = parse_html(
        "<html><head><title> The  Title </title><style>p{}</style></head>"
        "<body><h1>Head</h1><p>Some <b>bold</b> text.</p><script>var x;</script>"
        '<a href="/one">one</a> <a href="two.html">two</a> <a>no href</a></body></html>'
    )

    assert page.title == "The Title"
    assert page.text.splitlines() == ["Head", "Some bold text.", "one two no href"]
    assert page.links == ("/one", "two.html")
    assert not page.nofollow
    assert parse_html('<meta name="robots" content="noindex, nofollow">').nofollow


# --- crawl -----------------------------------------------------------------------------


def html(title: str, *links: str) -> httpx.Response:
    body = "".join(f'<a href="{link}">{link}</a>' for link in links)
    return httpx.Response(
        200,
        text=f"<html><title>{title}</title><body><p>{title} text</p>{body}</body></html>",
        headers={"content-type": "text/html; charset=utf-8"},
    )


SITE: dict[str, Callable[[], httpx.Response]] = {
    "/robots.txt": lambda: httpx.Response(
        200, text="User-agent: *\nDisallow: /docs/private\n"
    ),
    "/docs/": lambda: html(
        "Home", "a.html", "b.html?y=2&x=1", "b.html?x=1&y=2#top", "/elsewhere"
    ),
    "/docs/a.html": lambda: html(
        "A", "/docs/", "private/secret.html", "https://other.test/docs/"
    ),
    "/docs/b.html": lambda: html(
        "B", "a.html", "img.png", "missing.html", "flaky.html"
    ),
    "/docs/missing.html": lambda: httpx.Response(404),
    "/docs/img.png": lambda: httpx.Response(200, content=b"\x89PNG"),
}


def site(
    flaky_failures: int = 1, latency: float = 0.0
) -> tuple[Handler, Counter[str], dict]:
    hits: Counter[str] = Counter()
    load = {"now": 0, "max": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        hits[
            f"{request.url.host}{path}{'?' + request.url.query.decode() if request.url.query else ''}"
        ] += 1
        load["now"] += 1
        load["max"] = max(load["max"], load["now"])
        try:
            await asyncio.sleep(latency)
            if path == "/docs/flaky.html":
                if hits[f"{request.url.host}{path}"] <= flaky_failures:
                    return httpx.Response(503)
                return html("Flaky")
            if path.startswith("/docs/page"):
                n = int(path.removeprefix("/docs/page"))
                return html(f"P{n}", *(f"page{n * 3 + i}" for i in range(1, 4)))
            make = SITE.get(path)
            return make() if make else httpx.Response(404)
        finally:
            load["now"] -= 1

    return handler, hits, load


def collect(
    handler: Handler, seed: str = "https://h.test/docs/", **options
) -> tuple[list, CrawlStats]:
    stats = CrawlStats()

    async def go():
        async with client_for(handler) as client:
            options.setdefault("delay", 0.0)
            async with contextlib.aclosing(
                crawl([seed], client=client, stats=stats, **options)
            ) as pages:
                return [p async for p in pages]

    return asyncio.run(go()), stats


def test_crawl_visits_each_allowed_page_once(caplog: pytest.LogCaptureFixture) -> None:
    handler, hits, _ = site()

    with caplog.at_level(logging.INFO, logger="findex.crawler"):
        pages, stats = collect(handler, concurrency=3)

    titles = sorted(p.title for p in pages)
    assert titles == ["A", "B", "Flaky", "Home"]
    assert all(count == 1 for path, count in hits.items() if "flaky" not in path)
    assert hits["h.test/docs/b.html?x=1&y=2"] == 1  # two spellings, one fetch
    assert "h.test/docs/private/secret.html" not in hits  # robots.txt
    assert not any(h.startswith("other.test") for h in hits)  # off-domain
    assert "h.test/elsewhere" not in hits  # outside the seed's directory
    assert "h.test/docs/img.png" not in hits  # not a page
    assert hits["h.test/docs/flaky.html"] == 2  # one 503, then a retry
    assert stats.pages == 4 and stats.errors == 1  # the 404
    assert stats.statuses["404"] == 1
    logged = [r.getMessage() for r in caplog.records]
    assert any("flaky.html: HTTP 503, retry 1/3" in m for m in logged)
    assert any(m.startswith("404 https://h.test/docs/missing.html") for m in logged)
    assert any("robots.txt disallows" in m for m in logged)


def test_crawl_stops_at_max_pages() -> None:
    handler, _, _ = site()

    pages, stats = collect(
        handler, seed="https://h.test/docs/page0", max_pages=7, concurrency=4
    )

    assert len(pages) == 7 == stats.pages
    assert len({p.url for p in pages}) == 7
    assert stats.in_flight == 0


@pytest.mark.parametrize(("concurrency", "per_host"), [(10, 2), (3, 5)])
def test_concurrency_limits_hold(concurrency: int, per_host: int) -> None:
    handler, _, load = site(latency=0.01)

    collect(
        handler,
        seed="https://h.test/docs/page0",
        max_pages=30,
        concurrency=concurrency,
        per_host=per_host,
    )

    assert load["max"] <= min(concurrency, per_host)
    assert load["max"] >= 2  # and it really did overlap requests


def test_per_host_delay_spaces_requests_out() -> None:
    starts: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/robots.txt":
            starts.append(asyncio.get_running_loop().time())
        return html("P", "x1", "x2", "x3", "x4")

    collect(
        handler,
        seed="https://h.test/docs/x0",
        max_pages=5,
        concurrency=5,
        per_host=5,
        delay=0.03,
    )

    gaps = [b - a for a, b in pairwise(starts)]
    assert len(starts) >= 5
    assert min(gaps) >= 0.025


def test_breaking_out_early_leaves_no_tasks_behind() -> None:
    handler, _, _ = site(latency=0.01)

    async def go() -> int:
        async with client_for(handler) as client:
            async with contextlib.aclosing(
                crawl(
                    ["https://h.test/docs/page0"],
                    client=client,
                    delay=0.0,
                    max_pages=100,
                )
            ) as pages:
                async for _ in pages:
                    break
            return len(asyncio.all_tasks()) - 1  # minus this one

    assert asyncio.run(go()) == 0


@pytest.mark.slow  # starts the parser process pool
def test_jsonl_from_a_crawl_is_a_corpus_for_index_and_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handler, _, _ = site()
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        partial(httpx.AsyncClient, transport=httpx.MockTransport(handler)),
    )
    jsonl, log = tmp_path / "crawl.jsonl", tmp_path / "crawl.log"
    runner = CliRunner()

    crawled = runner.invoke(
        app,
        [
            "crawl",
            "https://h.test/docs/",
            "-o",
            str(jsonl),
            "--log",
            str(log),
            "--delay",
            "0",
        ],
    )
    assert crawled.exit_code == 0, crawled.stderr
    lines = [
        json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert sorted(r["title"] for r in lines) == ["A", "B", "Flaky", "Home"]
    assert {"url", "title", "text", "fetched_at", "status"} <= lines[0].keys()
    log_text = log.read_text(encoding="utf-8")
    assert "page" in log_text and "404 https://h.test/docs/missing.html" in log_text

    index = tmp_path / "web.bin"
    assert runner.invoke(app, ["index", str(jsonl), "-o", str(index)]).exit_code == 0
    found = runner.invoke(app, ["search", str(index), "flaky", "--json"])
    (record,) = map(json.loads, found.stdout.splitlines())
    assert record["url"] == "https://h.test/docs/flaky.html"
    assert record["title"] == "Flaky"
    assert "Flaky text" in record["snippet"]


def test_crawl_rejects_a_non_http_seed() -> None:
    result = CliRunner().invoke(app, ["crawl", "ftp://h.test/"])

    assert result.exit_code == 1
    assert "not an http(s) URL" in result.stderr


@pytest.mark.slow
def test_pages_can_be_parsed_in_a_process_pool() -> None:
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    handler, _, _ = site()
    with ProcessPoolExecutor(2, mp_context=get_context("spawn")) as pool:
        pages, _ = collect(handler, parse_pool=pool)

    assert sorted(p.title for p in pages) == ["A", "B", "Flaky", "Home"]


def test_parse_html_keeps_only_the_main_content_when_marked() -> None:
    page = parse_html(
        "<body><nav>Menu Home About</nav><div class='side'>Sidebar</div>"
        "<div role='main'><h1>Real</h1><p>Body<br>text <img src=x> here</p>"
        "<div><p>nested</p></div></div><footer>Footer</footer></body>"
    )

    assert page.text.splitlines() == ["Real", "Body", "text here", "nested"]
    assert parse_html("<nav>menu</nav><p>plain page</p>").text == "plain page"
