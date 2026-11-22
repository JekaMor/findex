"""robots.txt, fetched once per host and remembered."""

import asyncio
import logging
from urllib.robotparser import RobotFileParser

import httpx

from findex.crawler.fetch import USER_AGENT, fetch
from findex.crawler.urls import host_of

log = logging.getLogger(__name__)


class RobotsCache:
    """``await robots.allowed(url)``: may ``user_agent`` fetch it?

    Each host's robots.txt is fetched the first time a URL on that host is
    asked about; concurrent questions about the same host wait for that one
    fetch instead of starting their own. What the answer means, as Google's
    crawlers read it: 2xx -> obey the file; 4xx (no robots.txt) -> everything
    is allowed; 5xx, timeouts, network errors -> nothing is allowed for now.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        user_agent: str = USER_AGENT,
        timeout: float = 10.0,
        retries: int = 3,
    ) -> None:
        self._client = client
        self._retries = retries
        self._agent = user_agent
        self._timeout = timeout
        self._parsers: dict[str, asyncio.Task[RobotFileParser]] = {}

    async def allowed(self, url: str) -> bool:
        return (await self._parser(url)).can_fetch(self._agent, url)

    async def crawl_delay(self, url: str) -> float | None:
        delay = (await self._parser(url)).crawl_delay(self._agent)
        return None if delay is None else float(delay)

    def _parser(self, url: str) -> asyncio.Task[RobotFileParser]:
        host = host_of(url)
        task = self._parsers.get(host)
        if task is None:  # no await between the check and the insert: no race
            task = asyncio.ensure_future(self._load(host))
            self._parsers[host] = task
        return task

    async def _load(self, host: str) -> RobotFileParser:
        parser = RobotFileParser(f"{host}/robots.txt")
        result = await fetch(
            self._client,
            f"{host}/robots.txt",
            timeout=self._timeout,
            retries=self._retries,
        )
        if result.ok:
            parser.parse(result.body.decode("utf-8", errors="replace").splitlines())
            log.info("robots.txt of %s: %d bytes", host, len(result.body))
        elif result.status is not None and 400 <= result.status < 500:
            parser.parse([])  # no rules: everything allowed
            log.info(
                "robots.txt of %s: HTTP %d, everything allowed", host, result.status
            )
        else:
            parser.parse(["User-agent: *", "Disallow: /"])
            log.warning(
                "robots.txt of %s unavailable (%s): not crawling it",
                host,
                result.error or result.status,
            )
        return parser
