"""The crawl: a frontier queue, worker tasks in a TaskGroup, politeness limits.

    async with aclosing(crawl([seed], max_pages=200)) as pages:
        async for page in pages:
            ...

Shape: a ``frontier`` queue of URLs; ``concurrency`` workers each take a URL,
wait for its host (a per-host semaphore and a minimum gap between request
starts), then for a global slot, fetch it, parse it in a thread, and put new
links back on the frontier. Pages travel to the consumer through a second
queue. The generator itself never yields inside the TaskGroup: the workers live
in a separate task, so a consumer that stops early (``break``, an exception,
cancellation) only cancels that task, and the group shuts the workers down.
"""

import asyncio
import codecs
import contextlib
import logging
from collections.abc import AsyncGenerator, Iterable
from concurrent.futures import Executor
from dataclasses import dataclass, field
from functools import partial

import httpx

from findex.crawler.fetch import USER_AGENT, FetchResult, fetch, now_iso
from findex.crawler.html import ParsedPage, parse_html
from findex.crawler.robots import RobotsCache
from findex.crawler.urls import (
    canonical,
    host_of,
    in_scope,
    looks_like_page,
    scope_of,
    url_key,
)

log = logging.getLogger(__name__)
# One line per URL, whatever happened to it; the CLI writes it to crawl.log.
url_log = logging.getLogger("findex.crawler.urls")


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: str  # ISO 8601, UTC
    status: int


@dataclass(slots=True)
class CrawlStats:
    """Live counters; the crawl updates them, the caller may read them any time."""

    pages: int = 0  # HTML pages accepted for the consumer
    requests: int = 0  # finished fetches (robots.txt not counted)
    in_flight: int = 0  # requests sent, response not complete yet
    errors: int = 0  # no response, or a 4xx/5xx in the end
    skipped: int = 0  # robots.txt said no, not HTML, off-scope redirect
    queued: int = 0  # URLs waiting in the frontier
    statuses: dict[str, int] = field(default_factory=dict[str, int])


class HostGate:
    """At most ``limit`` requests to one host at a time, starts ``delay`` apart."""

    def __init__(self, limit: int, delay: float) -> None:
        self._slots = asyncio.Semaphore(limit)
        self._spacing = asyncio.Lock()
        self._delay = delay
        self._last_start = float("-inf")

    @contextlib.asynccontextmanager
    async def __call__(self) -> AsyncGenerator[None]:
        async with self._slots:
            # One worker at a time decides when it may start, and measures from
            # when the previous request really started: timers wake up late
            # (about 15 ms on Windows), so pre-booked start times could bunch up.
            async with self._spacing:
                loop = asyncio.get_running_loop()
                wait = self._last_start + self._delay - loop.time()
                if wait > 0:
                    await asyncio.sleep(wait)
                self._last_start = loop.time()
            yield


def _record(stats: CrawlStats, result: FetchResult, outcome: str) -> None:
    key = str(result.status) if result.status is not None else "error"
    stats.statuses[key] = stats.statuses.get(key, 0) + 1
    url_log.info(
        "%s %s bytes=%d elapsed=%.3f attempts=%d %s%s",
        result.status if result.status is not None else "---",
        result.url,
        len(result.body),
        result.elapsed,
        result.attempts,
        outcome,
        f" ({result.error})" if result.error else "",
    )


async def crawl(
    seeds: Iterable[str],
    *,
    max_pages: int = 200,
    concurrency: int = 10,
    per_host: int = 2,
    delay: float = 0.25,
    allowed_domains: Iterable[str] | None = None,
    stay_under_seed: bool = True,
    timeout: float = 10.0,
    user_agent: str = USER_AGENT,
    client: httpx.AsyncClient | None = None,
    stats: CrawlStats | None = None,
    parse_pool: Executor | None = None,
) -> AsyncGenerator[Page]:
    """Pages reachable from ``seeds``, as they are fetched, at most ``max_pages``.

    ``allowed_domains`` (default: the seeds' hosts) and, with ``stay_under_seed``,
    the seeds' directories bound where the crawl may go. ``robots.txt`` is obeyed
    and its ``Crawl-delay`` raises ``delay`` for that host.

    Pages are parsed in ``parse_pool`` (default: the loop's thread pool). A
    process pool keeps parsing from competing with the event loop for the GIL.
    """
    stats = stats if stats is not None else CrawlStats()
    canonical_seeds = [c for s in seeds if (c := canonical(s)) is not None]
    domains = (
        {d.lower() for d in allowed_domains}
        if allowed_domains
        else {host_of(s).split("://", 1)[1].split(":")[0] for s in canonical_seeds}
    )
    scopes = [scope_of(s) for s in canonical_seeds] if stay_under_seed else []

    wanted = LinkFilter(frozenset(domains), tuple(scopes))

    own_client = client is None
    # Building a client loads the CA certificates from disk: 0.2-0.3 s of
    # blocking work (asyncio debug mode caught it). Do it in a worker thread.
    http = client or await asyncio.to_thread(
        partial(
            httpx.AsyncClient,
            headers={"User-Agent": user_agent},
            follow_redirects=True,
            max_redirects=5,
            timeout=timeout,
            limits=httpx.Limits(max_connections=concurrency),
        )
    )
    robots = RobotsCache(http, user_agent, timeout)
    frontier: asyncio.Queue[str] = asyncio.Queue()
    # Bounded: if the consumer is slow, workers wait instead of piling up pages.
    out: asyncio.Queue[Page] = asyncio.Queue(maxsize=concurrency * 2)
    seen: set[str] = set()
    gates: dict[str, HostGate] = {}
    global_slots = asyncio.Semaphore(concurrency)
    enough = asyncio.Event()

    def enqueue(url: str, key: str) -> None:
        if key not in seen:
            seen.add(key)
            frontier.put_nowait(url)
            stats.queued = frontier.qsize()

    async def gate_for(url: str) -> HostGate:
        host = host_of(url)
        gate = gates.get(host)
        if gate is None:
            crawl_delay = await robots.crawl_delay(url)
            gate = gates.get(host)  # another worker may have made it meanwhile
            if gate is None:
                gate = HostGate(per_host, max(delay, crawl_delay or 0.0))
                gates[host] = gate
        return gate

    async def visit(url: str) -> None:
        if not await robots.allowed(url):
            stats.skipped += 1
            url_log.info("--- %s robots.txt disallows", url)
            return
        async with (await gate_for(url))(), global_slots:
            stats.in_flight += 1
            try:
                result = await fetch(http, url, timeout=timeout)
            finally:
                stats.in_flight -= 1
        stats.requests += 1
        final = canonical(result.final_url) or url
        if not result.ok:
            stats.errors += 1
            _record(stats, result, "error")
            return
        if final != url:  # redirected: count the target as seen too
            seen.add(url_key(final))
            if not wanted(final):
                stats.skipped += 1
                _record(stats, result, f"redirected off-scope to {final}")
                return
        if "html" not in result.content_type.lower():
            stats.skipped += 1
            _record(stats, result, f"skipped {result.content_type or 'no type'}")
            return
        # Decoding, parsing and resolving hundreds of links is CPU work: it runs
        # in the default thread pool, so the event loop stays free.
        # The loop is left with set lookups only: under the GIL a long step
        # here would also wait for the parser threads, and stall every worker.
        parsed, links = await asyncio.get_running_loop().run_in_executor(
            parse_pool, _digest, result.body, result.content_type, final, wanted
        )
        if enough.is_set():
            return
        _record(stats, result, "page")
        for target, key in links:
            enqueue(target, key)
        if stats.pages >= max_pages:
            return
        # Count the page before the first await below: no other worker can run
        # in between, so the crawl never hands out more than max_pages.
        stats.pages += 1
        if stats.pages >= max_pages:
            enough.set()
        await out.put(
            Page(final, parsed.title, parsed.text, now_iso(), result.status or 0)
        )

    async def worker() -> None:
        while True:
            url = await frontier.get()
            stats.queued = frontier.qsize()
            try:
                if not enough.is_set():
                    await visit(url)
            except Exception:  # one bad page must not stop the crawl
                log.exception("crawling %s failed", url)
                stats.errors += 1
            finally:
                frontier.task_done()

    async def run() -> None:
        async with asyncio.TaskGroup() as group:
            workers = [group.create_task(worker()) for _ in range(concurrency)]
            drained = group.create_task(frontier.join())
            stop = group.create_task(enough.wait())
            await asyncio.wait({drained, stop}, return_when=asyncio.FIRST_COMPLETED)
            for task in (*workers, drained, stop):
                task.cancel()  # cancelled children are not errors for the group

    for seed in canonical_seeds:
        if wanted(seed):
            enqueue(seed, url_key(seed))
    runner = asyncio.create_task(run())
    try:
        # Hand pages over until the workers are done and the queue is empty.
        while not (runner.done() and out.empty()):
            getter = asyncio.ensure_future(out.get())
            await asyncio.wait({getter, runner}, return_when=asyncio.FIRST_COMPLETED)
            if getter.done():
                yield getter.result()
            else:
                getter.cancel()
        await runner  # re-raise anything the workers' group raised
    finally:
        if not runner.done():
            runner.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await runner
        if own_client:
            await http.aclose()


@dataclass(frozen=True, slots=True)
class LinkFilter:
    """May the crawl visit ``url``? On an allowed domain, under a seed's
    directory (if ``scopes``), and not obviously a non-page. A module-level
    class, so it can travel to a parser process with the page."""

    domains: frozenset[str]
    scopes: tuple[str, ...]

    def __call__(self, url: str) -> bool:
        host = host_of(url).split("://", 1)[1].split(":")[0]
        on_domain = any(host == d or host.endswith("." + d) for d in self.domains)
        return (
            on_domain
            and (not self.scopes or in_scope(url, self.scopes))
            and looks_like_page(url)
        )


def _digest(
    body: bytes, content_type: str, url: str, wanted: LinkFilter
) -> tuple[ParsedPage, list[tuple[str, str]]]:
    """Decode and parse a page. Its links come back resolved, canonical, filtered
    by ``wanted`` and paired with their de-duplication key (none if nofollow).
    Runs in a worker thread or process, never on the event loop."""
    parsed = parse_html(body.decode(_charset(content_type), errors="replace"))
    if parsed.nofollow:
        return parsed, []
    links = (canonical(link, url) for link in parsed.links)
    return parsed, [(c, url_key(c)) for c in links if c is not None and wanted(c)]


def _charset(content_type: str) -> str:
    """The charset named in a Content-Type header, if Python knows it; else UTF-8."""
    for part in content_type.split(";")[1:]:
        name, _, value = part.strip().partition("=")
        if name.lower() == "charset" and value:
            try:
                return codecs.lookup(value.strip("\"'")).name
            except LookupError:
                break
    return "utf-8"
