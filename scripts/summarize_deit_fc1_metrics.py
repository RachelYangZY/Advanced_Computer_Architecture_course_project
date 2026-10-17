#!/usr/bin/env python3
"""Summarize B0/B1/X1 Spike traces for the DeiT FC1 tile workload."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

LOAD_RE = re.compile(r"\) (?:c\.)?(?:lb|lbu|lh|lhu|lw|ld|lwu)\b")
STORE_RE = re.compile(r"\) (?:c\.)?(?:sb|sh|sw|sd)\b")
MUL_RE = re.compile(r"\) mulw?\b")
XQ_RE = re.compile(r"\) xqmac8\b|0x56c5850b")
CTRL_RE = re.compile(r"\) (?:c\.)?(?:beq|bne|blt|bge|bltu|bgeu|jal|jalr|j|jr)\b")
COMPRESSED_RE = re.compile(r"\) c\.")
TOHOST_STORE_RE = re.compile(r"\) (?:c\.)?sd\b")


def bound_at_tohost(lines: list[str]) -> list[str]:
    """Keep retired instructions through the first post-pass tohost store.

    Heuristic: after `c.li a5, 1` (pass path), the following `sd` writes tohost.
    Fall back to full retired list if the pattern is not found.
    """
    retired = [ln for ln in lines if ln.startswith("core")]
    for index, line in enumerate(retired):
        if "c.li" in line and ", 1" in line:
            for end in range(index, min(index + 12, len(retired))):
                if TOHOST_STORE_RE.search(retired[end]):
                    return retired[: end + 1]
    # B0 -O0 may materialize the constant differently; cut at first sd after
    # leaving the gemm symbol range is harder here, so keep all retired lines
    # except pure spin loops of the form `c.j pc + 0` / `c.j pc - 2` tails.
    cut = len(retired)
    while cut > 0 and (
        "c.j     pc + 0" in retired[cut - 1]
        or "c.j     pc - 2" in retired[cut - 1]
        or "c.nop" in retired[cut - 1]
    ):
        cut -= 1
    return retired[:cut]


def summarize(label: str, path: Path, macs: int) -> dict[str, float | int | str]:
    lines = path.read_text(errors="replace").splitlines()
    retired = bound_at_tohost(lines)
    loads = sum(bool(LOAD_RE.search(ln)) for ln in retired)
    stores = sum(bool(STORE_RE.search(ln)) for ln in retired)
    muls = sum(bool(MUL_RE.search(ln)) for ln in retired)
    xq = sum(bool(XQ_RE.search(ln)) for ln in retired)
    ctrl = sum(bool(CTRL_RE.search(ln)) for ln in retired)
    compressed = sum(bool(COMPRESSED_RE.search(ln)) for ln in retired)
    count = len(retired)
    return {
        "config": label,
        "retired": count,
        "macs": macs,
        "mulw": muls,
        "xqmac8": xq,
        "loads": loads,
        "stores": stores,
        "control_flow": ctrl,
        "compressed": compressed,
        "macs_per_instr": (macs / count) if count else 0.0,
        "loads_per_mac": (loads / macs) if macs else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--macs", type=int, default=8 * 192 * 8)
    args = parser.parse_args()

    rows = [
        summarize("B0", Path("build/B0_deit_fc1/trace.log"), args.macs),
        summarize("B1", Path("build/B1_deit_fc1/trace.log"), args.macs),
        summarize("X1", Path("build/X1_deit_fc1/trace.log"), args.macs),
    ]
    headers = [
        "config",
        "retired",
        "macs",
        "mulw",
        "xqmac8",
        "loads",
        "stores",
        "control_flow",
        "compressed",
        "macs_per_instr",
        "loads_per_mac",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print(
            "| "
            + " | ".join(
                (
                    f"{row[h]:.6f}"
                    if isinstance(row[h], float)
                    else str(row[h])
                )
                for h in headers
            )
            + " |"
        )

    b0, b1, x1 = rows
    if b0["retired"] and b1["retired"]:
        print(
            f"B1 vs B0 dynamic instruction reduction: "
            f"{100.0 * (1 - b1['retired'] / b0['retired']):.2f}%"
        )
    if b1["retired"] and x1["retired"]:
        print(
            f"X1 vs B1 dynamic instruction reduction: "
            f"{100.0 * (1 - x1['retired'] / b1['retired']):.2f}%"
        )
    if b0["retired"] and x1["retired"]:
        print(
            f"X1 vs B0 dynamic instruction reduction: "
            f"{100.0 * (1 - x1['retired'] / b0['retired']):.2f}%"
        )


if __name__ == "__main__":
    main()
