#!/usr/bin/env python3
"""
Compare the current bench output against the stored baseline.

Usage:
    python ci/check_bench.py <current_output.txt> <bench_baseline.txt>

Exit codes:
    0 — nodes unchanged or decreased (improvement or no regression)
    1 — nodes increased beyond the tolerance threshold (search regression)

The script always prints a clear human-readable summary regardless of exit code.
Node counts are fully deterministic for a given engine binary: same source →
same count on every run.  An increase means pruning or move ordering got worse.
A decrease means efficiency improved — inspect to confirm it is not a bug that
causes premature cutoffs.

Tolerance:
    Changes within TOLERANCE_PCT are considered noise (platform or compiler
    differences) and do not fail CI.  The default is 1%.  Set to 0 to require
    an exact match.
"""

import re
import sys

TOLERANCE_PCT = 1.0   # percent above baseline that still passes


def parse_nodes(text: str) -> int:
    """Extract the 'Total nodes' figure from bench output."""
    m = re.search(r"Total nodes\s*:\s*([\d,]+)", text)
    if not m:
        raise ValueError(
            "Could not find 'Total nodes : ...' in bench output.\n"
            "Make sure the file contains output from: checksmith bench <depth>\n"
            f"First 300 chars of file:\n{text[:300]}"
        )
    return int(m.group(1).replace(",", ""))


def parse_depth(text: str) -> str:
    """Extract the depth label for display purposes (best-effort)."""
    m = re.search(r"Depth\s*:\s*(\d+)", text)
    return m.group(1) if m else "?"


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <current_output.txt> <bench_baseline.txt>")
        sys.exit(2)

    current_file, baseline_file = sys.argv[1], sys.argv[2]

    try:
        with open(current_file) as f:
            current_text = f.read()
    except FileNotFoundError:
        print(f"ERROR: current bench output not found: {current_file}")
        sys.exit(2)

    try:
        with open(baseline_file) as f:
            baseline_text = f.read()
    except FileNotFoundError:
        print(f"ERROR: baseline file not found: {baseline_file}")
        print("Run  python ci/update_bench.py  to create it.")
        sys.exit(2)

    current_nodes  = parse_nodes(current_text)
    baseline_nodes = parse_nodes(baseline_text)
    depth          = parse_depth(baseline_text)

    delta = current_nodes - baseline_nodes
    pct   = delta / baseline_nodes * 100 if baseline_nodes else 0.0

    print()
    print("=" * 60)
    print(f"  Bench regression check (depth {depth})")
    print("=" * 60)
    print(f"  Baseline : {baseline_nodes:>14,} nodes")
    print(f"  Current  : {current_nodes:>14,} nodes")
    print(f"  Delta    : {delta:>+14,} nodes  ({pct:+.2f}%)")
    print()

    if delta == 0:
        print("  RESULT: UNCHANGED — no regression.")
        print("=" * 60)
        sys.exit(0)
    elif delta < 0:
        print(f"  RESULT: IMPROVED — {abs(delta):,} fewer nodes ({abs(pct):.2f}% gain).")
        print("  If this is unexpected, verify there is no premature-cutoff bug.")
        print("  To accept as the new baseline: python ci/update_bench.py")
        print("=" * 60)
        sys.exit(0)
    else:
        if pct <= TOLERANCE_PCT:
            print(f"  RESULT: WITHIN TOLERANCE (+{pct:.2f}% <= {TOLERANCE_PCT}%) — pass.")
            print("=" * 60)
            sys.exit(0)
        else:
            print(f"  RESULT: REGRESSION — {delta:,} more nodes ({pct:+.2f}%).")
            print("  Search efficiency has decreased.  Possible causes:")
            print("    • A pruning condition was weakened or disabled.")
            print("    • Move ordering got worse (history/killer/TT move change).")
            print("    • A new search feature searches more nodes than it should.")
            print()
            print("  If this is intentional, update the baseline:")
            print("    python ci/update_bench.py")
            print("=" * 60)
            sys.exit(1)


if __name__ == "__main__":
    main()
