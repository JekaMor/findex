"""precision@5 of keyword, semantic and hybrid search on hand-labelled queries.

    python scripts/evaluate_modes.py index.bin            # needs index.emb.* (findex embed)

The 10 queries from Lab 3 plus 5 paraphrases (eval/queries.toml). Keyword mode
gets the words joined with OR, as in Lab 3; the semantic modes get the text.
"""

import sys
import tomllib
from pathlib import Path

from findex.semantic import FastEmbedder, Passages, Semantic, default_prefix, find
from findex.store import open_index
from findex.tokenize import tokenize

K = 5
DEPTH = 40  # how far to look for the first relevant book (for MRR)
MODES = ("keyword", "semantic", "hybrid")


def keyword_form(q: str) -> str:
    return q if " OR " in q else " OR ".join(dict.fromkeys(tokenize(q)))


def main(index_path: Path, queries_path: Path) -> None:
    queries = tomllib.loads(queries_path.read_text(encoding="utf-8"))["query"]
    with open_index(index_path) as index:
        passages = Passages.load(default_prefix(index_path), index)
        semantic = Semantic(passages, FastEmbedder(passages.model))
        groups = ("original", "paraphrase")
        p_sum = {g: dict.fromkeys(MODES, 0.0) for g in groups}
        rr_sum = {g: dict.fromkeys(MODES, 0.0) for g in groups}
        best = dict.fromkeys(groups, 0.0)
        counts = dict.fromkeys(groups, 0)
        print(
            "| Query | Kind | Relevant | P@5 keyword | P@5 semantic | P@5 hybrid "
            "| Best P@5 | 1st relevant: keyword | semantic | hybrid |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|")
        for item in queries:
            group = "paraphrase" if item.get("paraphrase") else "original"
            relevant = set(item["relevant"])
            row = [f"`{item['q']}`", group, str(len(relevant))]
            ranks: list[str] = []
            for mode in MODES:
                query = (
                    keyword_form(item["q"])
                    if mode == "keyword"
                    else item["q"].replace(" OR ", " ")
                )
                found, _ = find(index, query, mode, DEPTH, semantic=semantic)
                names = [Path(f.result.path).name for f in found]
                p = sum(n in relevant for n in names[:K]) / K
                first = next((i for i, n in enumerate(names, 1) if n in relevant), 0)
                p_sum[group][mode] += p
                rr_sum[group][mode] += 1 / first if first else 0.0
                row.append(f"{p:.1f}")
                ranks.append(str(first) if first else f">{DEPTH}")
            possible = min(len(relevant), K) / K
            best[group] += possible
            counts[group] += 1
            print("| " + " | ".join([*row, f"{possible:.1f}", *ranks]) + " |")
        for group in groups:
            n = counts[group]
            ps = " | ".join(f"**{p_sum[group][m] / n:.2f}**" for m in MODES)
            mrr = " | ".join(f"MRR **{rr_sum[group][m] / n:.2f}**" for m in MODES)
            print(
                f"| **mean, {group}** ({n}) | | | {ps} | {best[group] / n:.2f} | {mrr} |"
            )


if __name__ == "__main__":
    main(
        Path(sys.argv[1] if len(sys.argv) > 1 else "index.bin"),
        Path(sys.argv[2] if len(sys.argv) > 2 else "eval/queries.toml"),
    )
