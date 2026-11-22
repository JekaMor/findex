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

(filled in after the changes, with the same commands)
