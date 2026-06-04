#!/usr/bin/env python3
"""
Refresh bench_baseline.txt to match the current engine binary.

Run this after any intentional change that affects node counts:
    cargo build --release
    python ci/update_bench.py

The script builds and runs the bench at depth 10, then overwrites
bench_baseline.txt with the new output.  Commit the updated file so
CI passes on future runs.

Usage:
    python ci/update_bench.py [--depth N]   (default: 10)
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


BASELINE_PATH = Path("bench_baseline.txt")
BINARY_CANDIDATES = [
    Path("target/release/checksmith"),
    Path("target/release/checksmith.exe"),
]


def find_binary() -> Path:
    for p in BINARY_CANDIDATES:
        if p.exists():
            return p
    return None


def fmt_nodes(n: int) -> str:
    s = str(n)
    groups = []
    while len(s) > 3:
        groups.append(s[-3:])
        s = s[:-3]
    groups.append(s)
    return ",".join(reversed(groups))


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh bench_baseline.txt")
    ap.add_argument("--depth", type=int, default=10,
                    help="Bench depth (default: 10, same as CI)")
    args = ap.parse_args()

    # Build first.
    print(f"Building release binary …")
    build = subprocess.run(["cargo", "build", "--release"], capture_output=True, text=True)
    if build.returncode != 0:
        print("Build failed:")
        print(build.stderr[-2000:])
        sys.exit(1)
    print("Build OK.")

    binary = find_binary()
    if binary is None:
        print("ERROR: release binary not found after build.")
        sys.exit(1)

    # Run bench.
    print(f"Running: {binary} bench {args.depth}")
    result = subprocess.run(
        [str(binary), "bench", str(args.depth)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Bench failed:")
        print(result.stderr[-1000:])
        sys.exit(1)

    output = result.stdout

    # Extract and display key metrics.
    nodes_match = re.search(r"Total nodes\s*:\s*([\d,]+)", output)
    depth_match  = re.search(r"Depth\s*:\s*(\d+)", output)
    nps_match    = re.search(r"NPS\s*:\s*([\d,]+)", output)

    if not nodes_match:
        print("ERROR: Could not parse node count from bench output.")
        print(output)
        sys.exit(1)

    nodes = int(nodes_match.group(1).replace(",", ""))
    depth = depth_match.group(1) if depth_match else "?"
    nps   = nps_match.group(1) if nps_match else "?"

    # Warn if a previous baseline exists.
    if BASELINE_PATH.exists():
        old_text = BASELINE_PATH.read_text()
        old_match = re.search(r"Total nodes\s*:\s*([\d,]+)", old_text)
        if old_match:
            old_nodes = int(old_match.group(1).replace(",", ""))
            delta = nodes - old_nodes
            pct   = delta / old_nodes * 100 if old_nodes else 0.0
            print(f"  Old baseline : {fmt_nodes(old_nodes)} nodes")
            print(f"  New baseline : {fmt_nodes(nodes)} nodes  ({pct:+.2f}%)")
            if delta > 0:
                print("  WARNING: new baseline has MORE nodes (search regression?).")
            elif delta < 0:
                print("  Fewer nodes — search efficiency improved.")
            else:
                print("  Unchanged.")
    else:
        print(f"  Nodes : {fmt_nodes(nodes)}  depth={depth}  NPS={nps}")

    BASELINE_PATH.write_text(output)
    print(f"\nUpdated {BASELINE_PATH}")
    print("Commit this file so CI uses the new baseline:")
    print(f"  git add {BASELINE_PATH} && git commit -m 'ci: update bench baseline'")


if __name__ == "__main__":
    main()
