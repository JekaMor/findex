"""pytest-benchmark JSON -> a Markdown table (for the CI run summary)."""

import json
import sys
from pathlib import Path


def main(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    print("### Benchmarks (fixture corpus)\n")
    print("| Benchmark | Median | Min | Max | Rounds |")
    print("|---|---|---|---|---|")
    for bench in sorted(
        data["benchmarks"], key=lambda b: (b["group"] or "", b["name"])
    ):
        stats = bench["stats"]
        print(
            f"| `{bench['name']}` | {stats['median'] * 1e6:.1f} µs | "
            f"{stats['min'] * 1e6:.1f} µs | {stats['max'] * 1e6:.1f} µs | {stats['rounds']} |"
        )
    print(
        f"\n{data['machine_info'].get('cpu', {}).get('brand_raw', '')}, Python {data['machine_info']['python_version']}"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
