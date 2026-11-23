# Profiling findex (Lab 8)

Machine: the Lab 5 laptop (Intel Core i5-1035G1, 4 cores / 8 threads, Windows 10,
CPython 3.14.5). Corpus: 40 Gutenberg books, 5.3 M tokens, index with positions.

## Part 1 — before any change

Written down **before** touching the code (see the git history: this file was
committed first).

### What was run

| What | Command | Result |
|---|---|---|
| search, steady state, with snippets | `python scripts/profile_search.py index.bin -n 100` | **300–366 ms / query** |
| search, steady state, ranking only | `python scripts/profile_search.py index.bin -n 3000 --no-snippets` | **0.5 ms / query** |
| one `findex search` from the shell | `findex search index.bin "whale OR ship OR captain" --no-snippets --json` | 1.3–1.5 s |
| index build, one process | `findex index data -o x.bin --executor serial` | 7.1–7.5 s |

`profile_search.py` opens the index once and runs different three-term queries
(`sea OR ship OR captain`, …), the way the web service does.

Profiles in this folder:

- `flame-search.svg` — py-spy, search with snippets ([open it](flame-search.svg));
- `flame-search-ranking.svg` — py-spy, ranking only;
- `flame-index.svg` — py-spy, `findex index`;
- `search.prof`, `index.prof` — cProfile (`python -m pstats`, or `snakeviz`);
- `scalene-search.json`, `scalene-ranking.json` — Scalene, CPU only
  (`scalene view docs/scalene-search.json`).

py-spy needs `--subprocesses` on Windows: `.venv\Scripts\python.exe` is a small
launcher that starts the real interpreter as a child process. Scalene runs on
Windows but puts most samples on its own sampling thread (`threading.py:1024`), so
its per-line split is only usable for the lines it does attribute; cProfile and
py-spy carry the ranking below.

### The three hot spots

**1. Snippets re-read and re-scan whole books on every query.**
Search with snippets is 300–366 ms per query; without them, 0.5 ms. So 99.8 % of
a web search is snippets. cProfile (`tottime`, 200 queries): the regex scan over
the book 27 s, `str.lower()` of the whole book 19.7 s, `unicodedata.normalize`
17.7 s, reading the file 10 s — out of 84 s. py-spy: `snippet_parts` is 58 % of
the samples, `document_text`/`_read` 16 %. Scalene marks the `lower()`,
`normalize()` and regex lines as **native** time.
*Hypothesis:* the time is not in Python bytecode but in C routines run over
3 MB of text, ten times per query, for books that are the same every time. NumPy
cannot help; doing the work **once per document** instead of once per query will.

**2. Ranking walks Python objects it does not need.**
In the ranking-only flame graph, `ArrayPostings.__iter__` is 37 % of `search`:
for every posting it builds a `Posting` and a `tuple` of that posting's
*positions* — for a word like "sea" in a 570,000-word book, thousands of numbers
— although BM25 only reads `doc_id` and `tf`. Then `scores[doc_id] +=
scorer(...)` (25 %) and `BM25.__call__` (12 %) run once per posting in Python.
*Hypothesis:* Python-level per-posting overhead (object creation, a function call,
a dict update). Vectorising with NumPy over the `doc_ids`/`tfs` arrays removes all
three. On this corpus every postings list has at most 40 entries, so the gain
per query will be small in absolute terms; it should grow with the number of
documents.

**3. Indexing: one Python generator step per token.**
`tokenize` is the top function by own time in `findex index` (3.8 s `tottime`,
5.3 M calls, plus 1.2 s in `Match.group`): a generator resuming 5.3 million times.
*Hypothesis:* Python time (frame resume, a `group()` call per token). Lab 5
already parallelises it across processes; making each step cheaper is possible
(`re.findall` builds the whole list in C) but trades memory for speed — left as is.

**Also seen:** a single `findex search` from the shell spends 0.97 s just importing
`findex.cli`; `asyncio` (0.38 s) and `httpx` (0.17 s) are imported for the
crawler even when nobody crawls. *Hypothesis:* lazy imports cut CLI start-up by
about half.

## Part 2 — after

Same machine, same commands.

| What | Before | After | Change |
|---|---|---|---|
| search with snippets, per query | 300–366 ms | **21.6–22.5 ms** | ~15× |
| ranking only, per query | 0.5 ms | 0.2 ms | 2.5× |
| `import findex.cli` (every CLI command) | 0.97 s | 0.60 s | −0.37 s |
| index build, one process | 7.1–7.5 s | unchanged | — |

**1. Snippets.** The first fix, caching the prepared (NFC, lower-cased) text of
the last 8 documents, barely helped (300–329 ms): results spread over more than 8
books, and even a cached book is still scanned whole by the regex. The real fix
uses what the index already knows: the token positions of the query terms in
that document. `snippets.locate` finds the densest cluster of terms with NumPy
(`searchsorted` over the position arrays), turns the token position into an
estimated byte offset, and `snippet_for` reads only 64 KB of the file around it.
Snippets went from tens of ms to ~2 ms each; in 2,000 snippets the estimate
never missed (the whole-document fallback ran 0 times).

**2. Ranking.** `rank_numpy` scores each term's postings as int32 arrays
(`np.frombuffer` views onto the memory-mapped file, positions never decoded) and
takes the top k with `np.argpartition`. It ranks exactly like Lab 3's loop
(tested, including a Hypothesis property and ties at the k boundary). Speed-ups,
from `scripts/bench_scoring.py`:

| Index (documents) | Query | Docs matched | Lab 3, ms | NumPy, ms | Speedup |
|---|---|---|---|---|---|
| books (40) | `the` | 40 | 10.064 | 0.065 | 155× |
| books (40) | `publick` (rare) | 1 | 0.012 | 0.045 | 0.3× |
| books (40) | `sea OR ship OR captain` | 37 | 0.554 | 0.115 | 4.8× |
| passages (34,429) | `the` | 33,130 | 102.471 | 3.249 | 31.5× |
| passages (34,429) | `cabbin` (rare) | 1 | 0.017 | 0.074 | 0.2× |
| passages (34,429) | `sea OR ship OR captain` | 1,891 | 10.801 | 0.255 | 42.4× |

Without positions in the index (`--no-positions`) the books row for `the` is only
1.4× (0.104 → 0.072 ms): the 155× was mostly Lab 3 decoding 294,000 positions it
never needed. On the passages index without positions, pure vectorisation:
26× (`the`) and 22× (3 terms). A rare term is 3–5× *slower* in NumPy: one
posting costs less than the fixed cost of a few NumPy calls (~40 µs). On the
4-document test corpus (pytest-benchmark) NumPy is 5× slower for the same reason.

**3. Indexing** — not changed (see Part 1).

**CLI start-up.** The crawler (`asyncio`, `httpx`) is now imported inside
`findex crawl` only.

## Part 3 — under load, after adding semantic search

The cProfile numbers above are per call in one thread. `oha` against the running
server (`bench/oha-lab8-*.json`, 20 s, 50 connections) caught two things they did not.

1. **Lab 7's 420 req/s was a cache benchmark.** Lab 7 kept the snippets in an
   `lru_cache(2048)`, and the load test sent the same query (`white whale`) every
   time. A query it had not seen cost 230 ms. With a mix of 225 different queries
   (`--rand-regex-url`), Lab 7 serves **6 req/s** (p50 9.6 s); Lab 8 serves **223**
   on one worker and **579** on four.
2. **Rewriting the snippets dropped that cache.** The first Lab 8 run of the
   fixed query got 77 req/s: every request read and searched 9 × 64 KB of text
   (11 ms). The cache is back (`service.cached_snippet`, keyed by file, URL, terms
   and passage, guarded by a lock since the routes run in a thread pool), and the
   fixed query is at 293 req/s on one worker, 937 on four.

One false lead is worth recording: a run with the embedding model loaded showed
213 req/s against 501–557 without it. Alternating the two servers four times
gave 565, 471, 584 and 553: no difference, the 213 was a laptop that had been
under full load for minutes. The model and 34,429 × 384 embeddings cost 297 MB of
working set per worker; a semantic query costs ~2.7 ms of ONNX inference and ~2.2 ms
for `vectors @ q`.
