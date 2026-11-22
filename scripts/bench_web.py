"""def vs async def for /search, measured.

The real app keeps ``def search_api`` (FastAPI runs it in its thread pool). For
the comparison this module adds ``/search-async``: the same work, but in an
``async def`` route, i.e. on the event loop.

    # terminal 1: one worker serving both routes
    INDEX_PATH=index.bin uvicorn bench_web:app --app-dir scripts --port 8765
    # terminal 2: 20 requests at once to each route, 5 rounds
    python scripts/bench_web.py http://127.0.0.1:8765
"""

import asyncio
import itertools
import statistics
import sys
import time
from collections.abc import Iterator
from typing import Annotated

import httpx
from fastapi import Query

from findex.search import ScorerName
from findex.web.app import IndexDep, create_app, search_page

app = create_app()


@app.get("/search-async", include_in_schema=False)
async def search_async(
    index: IndexDep,
    q: Annotated[str, Query(min_length=1, max_length=200)],
    k: Annotated[int, Query(ge=1, le=50)] = 10,
    scorer: ScorerName = "bm25",
    page: Annotated[int, Query(ge=1, le=100)] = 1,
) -> dict[str, object]:
    # Deliberately wrong: CPU work inside a coroutine blocks the event loop.
    found = search_page(index, q, scorer, k, page)
    return {"total": found.total, "took_ms": found.took_ms}


WORDS = [
    "house",
    "water",
    "night",
    "child",
    "heart",
    "light",
    "letter",
    "friend",
    "money",
    "garden",
    "window",
    "evening",
    "morning",
    "river",
    "mountain",
    "forest",
    "horse",
    "table",
    "paper",
    "street",
    "church",
    "country",
    "father",
    "mother",
    "brother",
    "sister",
    "daughter",
    "husband",
    "doctor",
    "captain",
    "soldier",
    "prince",
    "king",
    "queen",
    "lady",
    "master",
    "servant",
    "stranger",
    "fire",
    "stone",
    "bread",
    "wine",
    "ship",
    "island",
    "sea",
    "storm",
    "silence",
    "voice",
    "music",
    "dream",
    "fear",
    "hope",
    "anger",
    "beauty",
    "truth",
    "power",
    "death",
    "life",
    "war",
    "peace",
    "love",
    "honour",
    "duty",
    "glory",
    "dog",
    "cat",
    "bird",
    "wolf",
    "snow",
    "rain",
    "winter",
    "summer",
    "spring",
    "autumn",
    "sun",
    "moon",
    "star",
    "gold",
    "silver",
    "iron",
    "glass",
    "blood",
    "bone",
    "hand",
    "face",
    "eye",
    "mouth",
    "hair",
    "skin",
    "head",
    "foot",
    "road",
    "bridge",
    "wall",
    "door",
    "gate",
    "field",
    "farm",
    "village",
    "city",
    "castle",
    "tower",
    "prison",
    "school",
    "book",
    "story",
    "song",
    "poem",
    "picture",
    "word",
    "name",
    "question",
    "answer",
    "secret",
    "promise",
    "reason",
    "chance",
    "danger",
    "journey",
    "battle",
    "victory",
    "enemy",
    "army",
    "sword",
    "gun",
    "carriage",
    "train",
    "coat",
    "hat",
    "dress",
    "shoe",
    "bed",
    "chair",
    "candle",
    "lamp",
    "clock",
    "key",
    "ring",
    "box",
    "bottle",
    "cup",
    "plate",
    "knife",
    "fork",
    "kitchen",
    "room",
    "flower",
    "tree",
    "leaf",
    "grass",
    "apple",
    "sugar",
    "salt",
    "milk",
    "tea",
    "coffee",
    "meat",
    "fish",
    "egg",
    "butter",
    "cheese",
]


def unique_queries() -> Iterator[str]:
    """Never the same query twice, so neither the match cache nor the snippet
    cache answers for a route: every request does the full work."""
    for a, b in itertools.combinations(WORDS, 2):
        yield f"{a} OR {b}"


QUERIES = unique_queries()


async def burst(base: str, route: str, n: int = 20) -> tuple[float, list[float], float]:
    """n different requests at once, while /health is probed every 20 ms.

    Returns (wall time, each request's latency, worst /health latency): the
    last one says whether the server could do anything else meanwhile.
    """
    queries = [next(QUERIES) for _ in range(n)]

    async def one(client: httpx.AsyncClient, q: str) -> float:
        started = time.perf_counter()
        r = await client.get(f"{base}{route}", params={"q": q, "k": 10})
        r.raise_for_status()
        return time.perf_counter() - started

    async def probe(client: httpx.AsyncClient, stop: asyncio.Event) -> float:
        worst = 0.0
        while not stop.is_set():
            started = time.perf_counter()
            await client.get(f"{base}/health")
            worst = max(worst, time.perf_counter() - started)
            await asyncio.sleep(0.02)
        return worst

    limits = httpx.Limits(max_connections=n + 5)
    async with httpx.AsyncClient(timeout=60, limits=limits) as client:
        await client.get(f"{base}/health")
        stop = asyncio.Event()
        started = time.perf_counter()
        async with asyncio.TaskGroup() as group:
            prober = group.create_task(probe(client, stop))
            tasks = [group.create_task(one(client, q)) for q in queries]
            await asyncio.gather(*tasks)
            wall = time.perf_counter() - started
            stop.set()
        return wall, [t.result() for t in tasks], prober.result()


async def main(base: str) -> None:
    print(
        "| Route | 20 at once: wall (median of 5) | request p50 | request max | worst /health meanwhile |"
    )
    print("|---|---|---|---|---|")
    for route, label in (
        ("/search", "`def` (thread pool)"),
        ("/search-async", "`async def` (event loop)"),
    ):
        walls: list[float] = []
        latencies: list[float] = []
        probes: list[float] = []
        for _ in range(5):
            wall, each, worst = await burst(base, route)
            walls.append(wall)
            latencies += each
            probes.append(worst)
        print(
            f"| {label} | {statistics.median(walls) * 1000:.0f} ms | "
            f"{statistics.median(latencies) * 1000:.0f} ms | {max(latencies) * 1000:.0f} ms | "
            f"{max(probes) * 1000:.0f} ms |"
        )


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"))
