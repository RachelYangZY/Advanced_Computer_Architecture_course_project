#!/usr/bin/env python3
"""Instruction-mix and per-function breakdown from Spike `-l` traces.

usage: spike_metrics.py --macs N [--outputs M] label=trace.log:prog.elf ...

Counts only retired instructions in program memory (pc >= 0x80000000) and
strips the trailing tohost spin loop, so every configuration is bounded the
same way. Per-function contribution comes from the ELF symbol table.
"""

from __future__ import annotations

import argparse
import bisect
import re
import subprocess
from collections import Counter
from pathlib import Path

LINE_RE = re.compile(r"0x([0-9a-f]{16}) \(0x[0-9a-f]+\)\s+(\S+)\s*(.*)")
LOADS = {"lb", "lbu", "lh", "lhu", "lw", "lwu", "ld"}
STORES = {"sb", "sh", "sw", "sd"}
MULS = {"mul", "mulw", "mulh", "mulhu", "mulhsu"}
CTRL = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "jal", "jalr", "j", "jr", "ret", "call"}
PROGRAM_BASE = 0x80000000


def strip_c(mnemonic: str) -> str:
    return mnemonic[2:] if mnemonic.startswith("c.") else mnemonic


def classify(mnemonic: str) -> str:
    base = strip_c(mnemonic)
    if base == "xqmac8":
        return "xqmac8"
    if base == "xrequant":
        return "xrequant"
    if base in MULS:
        return "multiply"
    if base in LOADS:
        return "load"
    if base in STORES:
        return "store"
    if base in CTRL or base.startswith("b"):
        return "branch/jump"
    return "int_alu/other"


def load_symbols(elf: str) -> tuple[list[int], list[str]]:
    out = subprocess.run(
        ["riscv64-unknown-elf-nm", "-n", elf], capture_output=True, text=True, check=True
    ).stdout
    addrs, names = [], []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] in ("T", "t"):
            addrs.append(int(parts[0], 16))
            names.append(parts[2])
    return addrs, names


def is_spin(mnemonic: str, operands: str) -> bool:
    return mnemonic == "c.nop" or (
        mnemonic in ("c.j", "j") and operands in ("pc + 0", "pc - 2")
    )


def summarize(label: str, trace: str, elf: str, macs: int, outputs: int) -> dict:
    """Stream the trace (can be >10M lines) and drop the trailing tohost spin loop.

    Spin-loop candidates are held back in `pending`; they are only counted if a
    real instruction follows them, so whatever is still pending at EOF (the
    completion spin) is excluded.
    """
    addrs, names = load_symbols(elf)
    mix: Counter[str] = Counter()
    per_func: Counter[str] = Counter()
    compressed = 0
    total = 0
    pending: Counter[tuple[str, str, bool]] = Counter()

    def commit(cls: str, func: str, is_c: bool, n: int = 1) -> None:
        nonlocal compressed, total
        mix[cls] += n
        per_func[func] += n
        compressed += n if is_c else 0
        total += n

    with open(trace, errors="replace") as handle:
        for line in handle:
            if not line.startswith("core"):
                continue
            m = LINE_RE.search(line)
            if not m:
                continue
            pc = int(m.group(1), 16)
            if pc < PROGRAM_BASE:
                continue
            mnemonic, operands = m.group(2), m.group(3).strip()
            idx = bisect.bisect_right(addrs, pc) - 1
            func = names[idx] if idx >= 0 else "?"
            key = (classify(mnemonic), func, mnemonic.startswith("c."))
            if is_spin(mnemonic, operands):
                pending[key] += 1
                continue
            for (cls, fn, is_c), n in pending.items():
                commit(cls, fn, is_c, n)
            pending.clear()
            commit(*key)
    mem = mix["load"] + mix["store"]
    return {
        "label": label, "total": total, "mix": mix, "per_func": per_func,
        "compressed": compressed, "macs": macs, "outputs": outputs,
        "macs_per_instr": macs / total if total else 0.0,
        "mem_per_mac": mem / macs if macs else 0.0,
        "instr_per_output": total / outputs if outputs else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--macs", type=int, required=True)
    parser.add_argument("--outputs", type=int, required=True)
    parser.add_argument("entries", nargs="+", help="label=trace.log:prog.elf")
    args = parser.parse_args()

    results = []
    for entry in args.entries:
        label, rest = entry.split("=", 1)
        trace, elf = rest.split(":", 1)
        results.append(summarize(label, trace, elf, args.macs, args.outputs))

    cats = ["multiply", "xqmac8", "xrequant", "load", "store", "branch/jump", "int_alu/other"]
    print("| config | retired | MACs/instr | (ld+st)/MAC | instr/output | compressed | " + " | ".join(cats) + " |")
    print("|---|---:|---:|---:|---:|---:|" + "---:|" * len(cats))
    for r in results:
        print(
            f"| {r['label']} | {r['total']} | {r['macs_per_instr']:.4f} | {r['mem_per_mac']:.4f} | "
            f"{r['instr_per_output']:.1f} | {r['compressed']} | "
            + " | ".join(str(r["mix"][c]) for c in cats) + " |"
        )
    print()
    print("Per-function retired instructions:")
    print()
    funcs = sorted({f for r in results for f in r["per_func"]})
    print("| function | " + " | ".join(r["label"] for r in results) + " |")
    print("|---|" + "---:|" * len(results))
    for f in funcs:
        print(f"| {f} | " + " | ".join(str(r["per_func"][f]) for r in results) + " |")
    print()
    base = results[0]["total"]
    for r in results[1:]:
        print(f"{r['label']} vs {results[0]['label']}: {100.0 * (1 - r['total'] / base):.2f}% fewer "
              f"({base / r['total']:.2f}x)")
    for prev, cur in zip(results[1:], results[2:]):
        print(f"{cur['label']} vs {prev['label']}: {100.0 * (1 - cur['total'] / prev['total']):.2f}% fewer "
              f"({prev['total'] / cur['total']:.2f}x)")


if __name__ == "__main__":
    main()
