#!/usr/bin/env python3
"""Whole-program dynamic instruction mix from a Spike PC histogram (spike -g) and the ELF disassembly.

usage: instr_mix.py CFG=ELF:HIST [CFG=ELF:HIST ...]
  ELF   program (ELF.with .elf replaced by .dis must exist: riscv64-unknown-elf-objdump -d)
  HIST  stderr of `spike -g ...` ("PC Histogram size:N" then "<hex pc> <count>" lines)

Classes: load, store, multiply (mul*), divide (div*/rem*), branch (conditional), jump (jal/jalr/j/jr),
custom (xqmac8 = custom-0 funct3 0, xrequant = custom-0 funct3 1), csr, alu (everything else).
Prints Markdown: totals, mix, memory instructions per MAC, code size, custom-instruction frequency,
and the top functions (with their own mix).
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import defaultdict

LOADS = {"lb", "lbu", "lh", "lhu", "lw", "lwu", "ld", "c.lw", "c.ld", "c.lwsp", "c.ldsp"}
STORES = {"sb", "sh", "sw", "sd", "c.sw", "c.sd", "c.swsp", "c.sdsp"}
BRANCHES = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez", "bltz", "bgez", "blez", "bgtz",
            "bgt", "ble", "bgtu", "bleu", "c.beqz", "c.bnez"}
JUMPS = {"jal", "jalr", "j", "jr", "ret", "call", "tail", "c.j", "c.jr", "c.jalr", "c.jal"}
MULS = {"mul", "mulh", "mulhu", "mulhsu", "mulw", "c.mul"}
DIVS = {"div", "divu", "rem", "remu", "divw", "divuw", "remw", "remuw"}
CLASSES = ["load", "store", "alu", "multiply", "branch", "jump", "xqmac8", "xrequant", "csr", "divide"]
LINE = re.compile(r"^\s*([0-9a-f]+):\s+([0-9a-f]+)\s+(\S+)\s*(.*)$")
MACS = 1_253_683_200


def classify(enc: str, mn: str) -> str:
    word = int(enc, 16)
    if len(enc) == 8 and (word & 0x7F) == 0x0B:          # custom-0
        f3 = (word >> 12) & 7
        if f3 == 0:
            return "xqmac8"
        if f3 == 1:
            return "xrequant"
    base = mn.split(".")[-1] if mn.startswith("c.") and mn not in LOADS | STORES | BRANCHES | JUMPS | MULS else mn
    if mn in LOADS:
        return "load"
    if mn in STORES:
        return "store"
    if mn in BRANCHES:
        return "branch"
    if mn in JUMPS:
        return "jump"
    if mn in MULS:
        return "multiply"
    if mn in DIVS:
        return "divide"
    if mn.startswith("csr") or base.startswith("csr"):
        return "csr"
    return "alu"


def load_dis(path: str) -> dict[int, tuple[str, str]]:
    insns = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = LINE.match(line)
        if m and re.fullmatch(r"[0-9a-f]{4}|[0-9a-f]{8}", m[2]):
            insns[int(m[1], 16)] = (m[2], m[3])
    return insns


def load_symbols(elf: str) -> list[tuple[int, int, str]]:
    out = subprocess.run(["riscv64-unknown-elf-nm", "-S", "-n", "--defined-only", elf],
                         capture_output=True, text=True, check=True).stdout
    syms = []
    for line in out.splitlines():
        p = line.split()
        if len(p) == 4 and p[2] in ("T", "t"):
            syms.append((int(p[0], 16), int(p[1], 16), p[3]))
    return syms


def text_size(elf: str) -> dict[str, int]:
    out = subprocess.run(["riscv64-unknown-elf-size", "-A", elf], capture_output=True, text=True,
                         check=True).stdout
    sizes = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 2 and p[0].startswith(".") and p[1].isdigit():
            sizes[p[0]] = int(p[1])
    return sizes


def load_hist(path: str) -> dict[int, int]:
    h = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        p = line.split()
        if len(p) == 2 and re.fullmatch(r"[0-9a-f]+", p[0]) and p[1].isdigit():
            h[int(p[0], 16)] = int(p[1])
    return h


def analyse(elf: str, hist_path: str):
    insns = load_dis(elf.removesuffix(".elf") + ".dis")
    syms = load_symbols(elf)
    hist = load_hist(hist_path)
    starts = [s[0] for s in syms]
    import bisect

    mix = defaultdict(int)
    fmix = defaultdict(lambda: defaultdict(int))
    unknown = 0
    for pc, n in hist.items():
        if pc not in insns:          # boot ROM etc.
            unknown += n
            continue
        enc, mn = insns[pc]
        c = classify(enc, mn)
        mix[c] += n
        i = bisect.bisect_right(starts, pc) - 1
        fname = syms[i][2] if i >= 0 and pc < syms[i][0] + max(syms[i][1], 1) else "?"
        fmix[fname][c] += n
    return mix, fmix, unknown, syms, text_size(elf)


def fmt(n: float) -> str:
    return f"{n:,.0f}"


def main() -> None:
    runs = {}
    for arg in sys.argv[1:]:
        cfg, rest = arg.split("=", 1)
        elf, hist = rest.split(":", 1)
        runs[cfg] = (elf, *analyse(elf, hist))
    cfgs = list(runs)
    total = {c: sum(runs[c][1].values()) for c in cfgs}

    print("# Dynamic instruction mix (whole model, Spike PC histogram)\n")
    print(f"MACs per inference: {fmt(MACS)}. Counts exclude {', '.join(f'{c}: {fmt(runs[c][3])}' for c in cfgs)} "
          "instructions executed outside the program text (Spike boot ROM).\n")
    print("| config | total | " + " | ".join(CLASSES) + " |")
    print("|---|---|" + "---|" * len(CLASSES))
    for c in cfgs:
        mix = runs[c][1]
        print(f"| {c} | {fmt(total[c])} | " + " | ".join(fmt(mix[k]) for k in CLASSES) + " |")
    print("\nShare of all instructions (%):\n")
    print("| config | " + " | ".join(CLASSES) + " |")
    print("|---|" + "---|" * len(CLASSES))
    for c in cfgs:
        mix = runs[c][1]
        print(f"| {c} | " + " | ".join(f"{100 * mix[k] / total[c]:.2f}" for k in CLASSES) + " |")

    print("\n## Per-MAC metrics\n")
    print("| config | instr/MAC | MAC/instr | loads+stores | (ld+st)/MAC | loads/MAC | multiplies/MAC | "
          "branch+jump/instr (%) |")
    print("|---|---|---|---|---|---|---|---|")
    for c in cfgs:
        mix = runs[c][1]
        mem = mix["load"] + mix["store"]
        bj = mix["branch"] + mix["jump"]
        print(f"| {c} | {total[c] / MACS:.3f} | {MACS / total[c]:.3f} | {fmt(mem)} | {mem / MACS:.3f} | "
              f"{mix['load'] / MACS:.3f} | {mix['multiply'] / MACS:.3f} | {100 * bj / total[c]:.1f} |")

    print("\n## Custom-instruction utilization\n")
    print("(Each xqmac8 does 8 MAC slots; the total can exceed 100% because the reduction length is "
          "padded to a multiple of 8, e.g. 197-long rows in attention.)\n")
    print("| config | xqmac8 executed | MACs done by xqmac8 | % of all MACs | xrequant executed | "
          "custom % of instructions |")
    print("|---|---|---|---|---|---|")
    for c in cfgs:
        mix = runs[c][1]
        xq, xr = mix["xqmac8"], mix["xrequant"]
        print(f"| {c} | {fmt(xq)} | {fmt(8 * xq)} | {100 * 8 * xq / MACS:.1f} | {fmt(xr)} | "
              f"{100 * (xq + xr) / total[c]:.2f} |")

    print("\n## Code size (bytes)\n")
    print("| config | .text | .rodata (weights, LUTs, params) | .data+.bss |")
    print("|---|---|---|---|")
    for c in cfgs:
        sz = runs[c][5]
        print(f"| {c} | {fmt(sz.get('.text', 0))} | {fmt(sz.get('.rodata', 0))} | "
              f"{fmt(sz.get('.data', 0) + sz.get('.bss', 0))} |")

    print("\n## Kernel code size (bytes of the functions that differ between configs)\n")
    names = ["op_fc_i8", "op_fc_i8_ref", "op_bmm_i8", "op_bmm_i8_ref", "op_conv2d_i8", "op_conv2d_i8_ref",
             "model_run"]
    print("| function | " + " | ".join(cfgs) + " |")
    print("|---|" + "---|" * len(cfgs))
    for nme in names:
        row = []
        for c in cfgs:
            hit = [s[1] for s in runs[c][4] if s[2] == nme]
            row.append(fmt(hit[0]) if hit else "-")
        print(f"| {nme} | " + " | ".join(row) + " |")

    for c in cfgs:
        fm = runs[c][2]
        print(f"\n## Top functions, {c}\n")
        print("| function | instructions | % | load | store | alu | mul | branch+jump | xqmac8 | xrequant |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        ranked = sorted(fm.items(), key=lambda kv: -sum(kv[1].values()))[:8]
        for fn, m in ranked:
            t = sum(m.values())
            print(f"| `{fn}` | {fmt(t)} | {100 * t / total[c]:.1f} | {fmt(m['load'])} | {fmt(m['store'])} | "
                  f"{fmt(m['alu'])} | {fmt(m['multiply'])} | {fmt(m['branch'] + m['jump'])} | "
                  f"{fmt(m['xqmac8'])} | {fmt(m['xrequant'])} |")


if __name__ == "__main__":
    main()
