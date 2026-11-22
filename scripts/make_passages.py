"""Cut a corpus into ~200-word passages, one JSON line each.

    python scripts/make_passages.py data data/passages.jsonl

Each line: {"url": "<file>#<passage>", "title": "<book title> (passage N)", "text": ...},
which ``findex index`` reads like a crawl: a corpus of many short documents.
"""

import json
import sys
from pathlib import Path

from findex.chunks import passages
from findex.corpus import iter_documents
from findex.index import title_of


def main(corpus: Path, out: Path) -> None:
    count = 0
    with out.open("w", encoding="utf-8") as sink:
        for doc in iter_documents(corpus):
            title = title_of(doc)
            for p in passages(doc.doc_id, doc.text):
                record = {
                    "url": f"{doc.path.name}#{p.number}",
                    "title": f"{title} (passage {p.number + 1})",
                    "text": p.text,
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    print(f"{count:,} passages -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
