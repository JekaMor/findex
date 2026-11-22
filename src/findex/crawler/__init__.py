"""An asyncio web crawler that feeds pages into the findex corpus (Lab 6)."""

from findex.crawler.crawl import CrawlStats, Page, crawl
from findex.crawler.fetch import USER_AGENT, Backoff, FetchResult, fetch
from findex.crawler.robots import RobotsCache
from findex.crawler.urls import canonical

__all__ = [
    "USER_AGENT",
    "Backoff",
    "CrawlStats",
    "FetchResult",
    "Page",
    "RobotsCache",
    "canonical",
    "crawl",
    "fetch",
]
