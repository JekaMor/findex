"""precision@5 of TF-IDF vs BM25 on hand-labelled queries.

``python scripts/evaluate.py index.bin [eval/queries.toml]`` prints a Markdown table.
"""

import sys
import tomllib
from pathlib import Path

from findex.scoring import BM25, TfIdf
from findex.search import search
from findex.store import open_index

K = 5
SCORERS = {"TF-IDF": TfIdf(), "BM25": BM25()}


def precision_at_k(found: list[str], relevant: set[str], k: int = K) -> float:
    return sum(name in relevant for name in found[:k]) / k


def main(index_path: Path, queries_path: Path) -> None:
    queries = tomllib.loads(queries_path.read_text(encoding="utf-8"))["query"]
    totals = dict.fromkeys(SCORERS, 0.0)
    best_total = 0.0
    names = " | ".join(SCORERS)
    print(f"| Query | Relevant | {names} | Best possible |")
    print("|---|---|" + "---|" * len(SCORERS) + "---|")
    with open_index(index_path) as index:
        for item in queries:
            relevant = set(item["relevant"])
            row = [f"`{item['q']}`", str(len(relevant))]
            for label, scorer in SCORERS.items():
                found = [
                    Path(r.path).name for r in search(index, item["q"], scorer, k=K)
                ]
                p = precision_at_k(found, relevant)
                totals[label] += p
                row.append(f"{p:.1f}")
            best = min(len(relevant), K) / K
            best_total += best
            row.append(f"{best:.1f}")
            print("| " + " | ".join(row) + " |")
    n = len(queries)
    means = " | ".join(f"**{totals[label] / n:.2f}**" for label in SCORERS)
    print(f"| **mean** | | {means} | {best_total / n:.2f} |")


if __name__ == "__main__":
    main(
        Path(sys.argv[1] if len(sys.argv) > 1 else "index.bin"),
        Path(sys.argv[2] if len(sys.argv) > 2 else "eval/queries.toml"),
    )
