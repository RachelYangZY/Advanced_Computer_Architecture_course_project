#!/usr/bin/env python3
"""Model block diagram with the top-5 ops of one transformer block (share of B0 instructions).

usage: plot_top_ops.py [results/model/op_mix.json] [OUTDIR]
Writes OUTDIR/top_ops.png.  Ranking stat: share of the block's B0 retired instructions
(= calls x cost per call).  Calls and MACs are printed next to each bar for context.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

JSON = sys.argv[1] if len(sys.argv) > 1 else "results/model/op_mix.json"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/figures")

NAMES = {"FULLY_CONNECTED": "FC (fully connected)", "BATCH_MATMUL": "BMM (batch matmul)",
         "CONV_2D": "CONV (patch embed)", "ADD": "ADD", "MUL": "MUL", "SQUARED_DIFFERENCE": "SQ_DIFF",
         "TRANSPOSE": "TRANSPOSE", "SOFTMAX": "SOFTMAX", "MEAN": "MEAN"}
RED, GRAY, BLUE = "#d62728", "#9aa0a6", "#1f77b4"


def agg(ops: list[dict], b0: list[dict], lo: int, hi: int) -> dict[str, list[int]]:
    a: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # calls, macs, B0 instructions
    for o in b0:
        if lo <= o["op"] <= hi:
            r = a[o["kind"]]
            r[0] += 1
            r[1] += o["macs"]
            r[2] += o["instret"]
    return a


def box(ax, x, y, w, h, text, fc, ec="#345", fs=11, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.5)


def main() -> None:
    d = json.loads(Path(JSON).read_text())
    b0 = d["cfgs"]["B0"]["ops"]
    blk = agg(None, b0, 8, 59)
    stem = agg(None, b0, 0, 7)
    # whole model ~= stem + 12 x block 0 (exact per-kind totals are in results/model/profile.md)
    whole: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for src, mult in ((stem, 1), (blk, 12)):
        for k, v in src.items():
            for j in range(3):
                whole[k][j] += mult * v[j]
    blk_total = sum(v[2] for v in blk.values())
    whole_total = sum(v[2] for v in whole.values())
    top = sorted(blk.items(), key=lambda kv: -kv[1][2])[:5]
    mac_share_whole = sum(v[2] for k, v in whole.items() if v[1] > 0) / whole_total

    fig, ax = plt.subplots(figsize=(17, 8.2))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    ax.set_title("DeiT-Tiny: where the instructions go, and why FC / BMM / CONV are the targets for xqmac8",
                 fontsize=15, fontweight="bold", loc="left")

    # left and right stages
    box(ax, 0.2, 3.3, 1.9, 1.3, "Image\n224x224x3", "#f1f3f4")
    st = whole["CONV_2D"]
    box(ax, 2.6, 3.0, 2.5, 1.9, "STEM\nPatch embed\n(CONV 16x16)\n+ CLS + pos", "#fdebd0", fs=10)
    ax.text(3.85, 2.55, f"CONV = {100 * st[2] / whole_total:.1f}% of all B0 instr.\n{st[1] / 1e6:.1f} M MACs",
            ha="center", va="top", fontsize=9, color=RED, fontweight="bold")
    box(ax, 15.1, 3.0, 1.8, 1.9, "HEAD\nfinal norm\n+ FC 192->1000", "#fdebd0", fs=10)

    # transformer block panel
    px, py, pw, ph = 5.6, 0.5, 9.0, 7.0
    ax.add_patch(FancyBboxPatch((px, py), pw, ph, boxstyle="round,pad=0.02,rounding_size=0.2", fc="#eef3fb",
                                ec="#456", lw=2, ls="--"))
    ax.text(px + 0.25, py + ph - 0.3, "TRANSFORMER BLOCK  x 12", fontsize=13, fontweight="bold", va="center")
    ax.text(px + 0.25, py + ph - 0.75, "Top 5 ops in one block, ranked by share of B0 (baseline) instructions",
            fontsize=10, va="center", color="#333")

    # bars
    bx, bw_max = px + 3.1, 4.3
    row_h, y0 = 0.85, py + ph - 1.8
    ax.text(px + 0.25, y0 + 0.62, "op  (calls per block)", fontsize=9, color="#555")
    ax.text(bx, y0 + 0.62, "share of block's B0 instructions", fontsize=9, color="#555")
    ax.text(px + pw - 0.2, y0 + 0.62, "MACs / block", fontsize=9, color="#555", ha="right")
    for i, (kind, (calls, macs, ins)) in enumerate(top):
        y = y0 - i * row_h
        share = ins / blk_total
        col = RED if macs else GRAY
        ax.text(px + 0.25, y + 0.28, f"{NAMES.get(kind, kind)}", fontsize=11, fontweight="bold", va="center")
        ax.text(px + 0.25, y - 0.02, f"{calls} calls   ({ins / calls / 1e6:,.1f} M instr/call)", fontsize=8.5,
                va="center", color="#444")
        w = max(bw_max * share, 0.03)
        ax.add_patch(plt.Rectangle((bx, y + 0.02), w, 0.5, fc=col, ec="none"))
        ax.text(bx + w + 0.1, y + 0.27, f"{100 * share:.1f}%", fontsize=11, fontweight="bold", va="center")
        ax.text(px + pw - 0.2, y + 0.27, f"{macs / 1e6:,.1f} M" if macs else "0", fontsize=11, va="center",
                ha="right", color=RED if macs else "#666", fontweight="bold" if macs else "normal")
    top_mac = sum(v[2] for k, v in top if v[1])
    ys = y0 - 5 * row_h + 0.45
    ax.add_patch(plt.Rectangle((px + 0.25, ys - 0.02), 0.25, 0.2, fc=RED))
    ax.text(px + 0.6, ys + 0.08, "has MACs (xqmac8 target)", fontsize=9, va="center")
    ax.add_patch(plt.Rectangle((px + 3.3, ys - 0.02), 0.25, 0.2, fc=GRAY))
    ax.text(px + 3.65, ys + 0.08, "no MACs (element-wise / other)", fontsize=9, va="center")
    ax.text(px + 0.25, py + 0.95,
            f"FC + BMM = {100 * top_mac / blk_total:.1f}% of this block.\n"
            f"Whole model: FC + BMM + CONV = {100 * mac_share_whole:.1f}% of B0 instructions.",
            fontsize=10.5, fontweight="bold", va="center", color=RED, linespacing=1.5)
    # arrows
    for (a, b) in ((2.1, 2.6), (5.1, px), (px + pw, 15.1)):
        ax.add_patch(FancyArrowPatch((a, 3.95), (b, 3.95), arrowstyle="-|>", mutation_scale=18, color="#345", lw=1.6))
    fig.text(0.01, 0.01,
             "Measured on Spike (minstret between op boundaries, B0 = naive baseline).  "
             f"Block total {blk_total / 1e6:,.0f} M B0 instructions; whole model ~{whole_total / 1e6:,.0f} M (stem + 12 x block; exact 12,599 M in profile.md).",
             fontsize=8.5, color="#444")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "top_ops.png", dpi=130, bbox_inches="tight")
    print("wrote", OUT / "top_ops.png")
    for k, v in top:
        print(k, v, f"{100 * v[2] / blk_total:.1f}%")
    print("whole-model (stem + 12 x block) mac-op share", f"{100 * mac_share_whole:.1f}%")


if __name__ == "__main__":
    main()
