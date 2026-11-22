"""One polite HTTP GET: a timeout, bounded retries with backoff, a record of what happened."""

import asyncio
import email.utils
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import httpx

log = logging.getLogger(__name__)

USER_AGENT: Final = (
    "findex-crawler/0.6 (course project; +https://github.com/JekaMor/findex)"
)
MAX_BYTES: Final = 5 * 2**20  # a page larger than this is cut off
RETRY_AFTER_CAP: Final = 60.0  # a server asking us to wait longer: give up instead

type Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str  # what was asked for
    final_url: str  # after redirects
    status: int | None  # None: no HTTP response at all (timeout, connection error)
    content_type: str
    body: bytes
    elapsed: float  # seconds, all attempts and waits included
    attempts: int
    error: str | None = None  # why there is no usable response

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300


@dataclass(frozen=True, slots=True)
class Backoff:
    """``min(cap, base * 2**attempt) + uniform(0, jitter)`` seconds before a retry."""

    base: float = 0.5
    cap: float = 8.0
    jitter: float = 1.0

    def delay(self, attempt: int, rng: Callable[[], float] = random.random) -> float:
        return min(self.cap, self.base * 2**attempt) + rng() * self.jitter


def retry_after(response: httpx.Response, now: datetime | None = None) -> float | None:
    """Seconds asked for by a ``Retry-After`` header (seconds or an HTTP date)."""
    value = response.headers.get("retry-after")
    if value is None:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except TypeError, ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - (now or datetime.now(UTC))).total_seconds())


def _retryable(status: int) -> bool:
    return status == 429 or 500 <= status < 600


async def _get(client: httpx.AsyncClient, url: str) -> tuple[httpx.Response, bytes]:
    async with client.stream("GET", url) as response:
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
            size += len(chunk)
            if size >= MAX_BYTES:
                log.warning("%s: cut off at %d bytes", url, size)
                break
        return response, b"".join(chunks)


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    retries: int = 3,
    backoff: Backoff = Backoff(),  # noqa: B008  # frozen, so one shared instance is fine
    sleep: Sleep = asyncio.sleep,
    rng: Callable[[], float] = random.random,
) -> FetchResult:
    """GET ``url``; retry 429, 5xx and timeouts up to ``retries`` times.

    Other 4xx answers are final (a 404 will not become a 200 by asking again);
    so are connection errors and redirect loops. ``Retry-After`` on a 429 or 503
    replaces the computed backoff, unless it asks for more than a minute.
    """
    started = time.perf_counter()
    status: int | None = None
    error: str | None = None
    attempts = 0
    for attempt in range(retries + 1):
        attempts = attempt + 1
        wait: float | None = None
        try:
            async with asyncio.timeout(timeout):
                response, body = await _get(client, url)
        except TimeoutError, httpx.TimeoutException:
            status, error = None, f"timeout after {timeout:g} s"
        except httpx.TooManyRedirects:
            return _failed(url, started, attempt + 1, "too many redirects")
        except httpx.HTTPError as exc:  # connection refused, DNS, TLS, ...
            return _failed(url, started, attempt + 1, f"{type(exc).__name__}: {exc}")
        else:
            status = response.status_code
            if not _retryable(status):
                return FetchResult(
                    url,
                    str(response.url),
                    status,
                    response.headers.get("content-type", ""),
                    body,
                    time.perf_counter() - started,
                    attempt + 1,
                )
            error = f"HTTP {status}"
            wait = retry_after(response)
            if wait is not None and wait > RETRY_AFTER_CAP:
                log.warning("%s: Retry-After %.0f s is too long, giving up", url, wait)
                break
        if attempt == retries:
            break
        delay = wait if wait is not None else backoff.delay(attempt, rng)
        log.info(
            "%s: %s, retry %d/%d in %.2f s%s",
            url,
            error,
            attempt + 1,
            retries,
            delay,
            " (Retry-After)" if wait is not None else "",
        )
        await sleep(delay)
    return FetchResult(
        url, url, status, "", b"", time.perf_counter() - started, attempts, error
    )


def _failed(url: str, started: float, attempts: int, error: str) -> FetchResult:
    return FetchResult(
        url, url, None, "", b"", time.perf_counter() - started, attempts, error
    )


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
