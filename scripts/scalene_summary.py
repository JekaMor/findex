"""Lines that took the most CPU in a Scalene profile, split into Python / native / system.

    scalene run --cpu-only --profile-only findex -o docs/scalene-search.json \
        scripts/profile_search.py --- index.bin -n 40
    python scripts/scalene_summary.py docs/scalene-search.json
"""

import json
import sys
from pathlib import Path


def main(path: Path, top: int = 12) -> None:
    profile = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for name, data in profile.get("files", {}).items():
        short = name.replace("\\", "/").split("/src/")[-1].split("/scripts/")[-1]
        for line in data.get("lines", []):
            py = line.get("n_cpu_percent_python", 0.0)
            native = line.get("n_cpu_percent_c", 0.0)
            system = line.get("n_sys_percent", 0.0)
            if py + native + system >= 1.0:
                rows.append(
                    (
                        py + native + system,
                        py,
                        native,
                        system,
                        short,
                        line["lineno"],
                        line["line"].strip(),
                    )
                )
    print("| Line | Total | Python | Native | System | Code |")
    print("|---|---|---|---|---|---|")
    for total, py, native, system, short, lineno, code in sorted(rows, reverse=True)[
        :top
    ]:
        print(
            f"| `{short}:{lineno}` | {total:.0f}% | {py:.0f}% | {native:.0f}% | {system:.0f}% | `{code[:60]}` |"
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
