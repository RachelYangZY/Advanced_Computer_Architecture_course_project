#!/usr/bin/env python3
"""Summarize per-op Spike counters of generated-model runs (scripts/run/run_model.sh).

usage: model_profile.py CFG=LOG [CFG=LOG ...]     (first CFG is the speedup baseline)
Prints Markdown: totals and correctness, per op kind, per model section.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict

PROF = re.compile(r"^PROF op=(\d+) kind=(\S+) macs=(\d+) instret=(\d+) cycles=(\d+)")
CHECK = re.compile(r"^CHECK t(\d+) n=(\d+) mismatches=(\d+) max_diff=(\d+) top1=(\d+) golden_top1=(\d+)")


def section(op: int) -> str:
    """DeiT-Tiny layout of this export: stem 0-7, blocks of 52 ops (block 11: 64 incl. final norm)."""
    if op < 8:
        return "stem"
    if op < 580:
        return f"block {(op - 8) // 52}"
    return "block 11 + final norm" if op < 644 else "head"


def parse(path: str):
    ops, checks, result = [], [], "?"
    for line in open(path, encoding="utf-8", errors="replace"):
        if m := PROF.match(line):
            ops.append((int(m[1]), m[2], int(m[3]), int(m[4]), int(m[5])))
        elif m := CHECK.match(line):
            checks.append(tuple(int(x) for x in m.groups()))
        elif line.startswith("RESULT"):
            result = line.split()[1]
    return ops, checks, result


def fmt(n: float) -> str:
    return f"{n:,.0f}"


def main() -> None:
    runs = {}
    for arg in sys.argv[1:]:
        cfg, path = arg.split("=", 1)
        runs[cfg] = parse(path)
    cfgs = list(runs)
    base = cfgs[0]
    total = {c: sum(o[3] for o in runs[c][0]) for c in cfgs}
    macs = sum(o[2] for o in runs[base][0])

    print("# Generated model on Spike\n")
    print(f"Ops profiled: {len(runs[base][0])}; MACs: {fmt(macs)}. Counts are retired instructions"
          " (`minstret`) between op boundaries; Spike retires one instruction per cycle.\n")
    print("| config | instructions | speedup vs " + base + " | instr/MAC | logits vs TFLite | top-1 |")
    print("|---|---|---|---|---|---|")
    for c in cfgs:
        ops, checks, result = runs[c]
        chk = "; ".join(f"{m} of {n} differ (max {d})" if m else "bit-exact" for _, n, m, d, _, _ in checks)
        top = "; ".join(f"{t} (golden {g})" for *_, t, g in checks)
        print(f"| {c} | {fmt(total[c])} | {total[base] / total[c]:.2f}x | {total[c] / macs:.2f} | "
              f"{chk or result} | {top} |")

    kinds = defaultdict(lambda: defaultdict(int))
    kmacs = defaultdict(int)
    for c in cfgs:
        for _, kind, m, ins, _ in runs[c][0]:
            kinds[kind][c] += ins
            if c == base:
                kmacs[kind] += m
    print("\n## By op kind\n")
    print("| op kind | MACs | " + " | ".join(f"{c} instr | {c} %" for c in cfgs) + " |")
    print("|---|---|" + "---|---|" * len(cfgs))
    for kind in sorted(kinds, key=lambda k: -kinds[k][base]):
        cells = " | ".join(f"{fmt(kinds[kind][c])} | {100 * kinds[kind][c] / total[c]:.1f}" for c in cfgs)
        print(f"| {kind} | {fmt(kmacs[kind])} | {cells} |")

    secs = defaultdict(lambda: defaultdict(int))
    order: list[str] = []
    for c in cfgs:
        for op, _, _, ins, _ in runs[c][0]:
            s = section(op)
            if s not in order:
                order.append(s)
            secs[s][c] += ins
    print("\n## By model section\n")
    print("| section | " + " | ".join(cfgs) + " |")
    print("|---|" + "---|" * len(cfgs))
    for s in order:
        print(f"| {s} | " + " | ".join(fmt(secs[s][c]) for c in cfgs) + " |")


if __name__ == "__main__":
    main()
