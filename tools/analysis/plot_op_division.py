#!/usr/bin/env python3
"""Instructions per op kind for B0, B1 and X1: absolute (left) and share of each config's total (right).

usage: plot_op_division.py [results/model/profile.md] [OUTDIR] [CFG ...]     (default cfgs: B0 B1 X1)
Writes OUTDIR/op_division.png. Counts are retired instructions (minstret between op boundaries),
summed over all 648 ops, read from the "By op kind" table of profile.md.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROFILE = sys.argv[1] if len(sys.argv) > 1 else "results/model/profile.md"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/figures")
CFGS = sys.argv[3:] or ["B0", "B1", "X1"]

# stacking order (bottom to top) and colors: reds = ops with MACs (xqmac8 targets), others = no MACs
GROUPS = [
    ("FULLY_CONNECTED", "Fully connected (FC)", "#b2182b"),
    ("BATCH_MATMUL", "Batch matmul (BMM)", "#ef6548"),
    ("CONV_2D", "Convolution (patch embed)", "#fdb27a"),
    ("ADD", "ADD", "#2166ac"),
    ("SOFTMAX", "SOFTMAX", "#4393c3"),
    ("MUL", "MUL", "#92c5de"),
    ("SQUARED_DIFFERENCE", "SQUARED_DIFFERENCE", "#7fbf7b"),
    ("OTHER", "Other (transpose, mean, slice, GELU, ...)", "#b8b8b8"),
]
KEYS = {g[0] for g in GROUPS}


def main() -> None:
    # whole-model per-kind totals come from the "By op kind" table of results/model/profile.md
    # (op_mix.json only covers the stem and block 0)
    start = next(i for i, ln in enumerate(Path(PROFILE).read_text().splitlines()) if ln.startswith("## By op kind"))
    lines = Path(PROFILE).read_text().splitlines()[start + 2:]
    header = [h.strip() for h in lines[0].strip().strip("|").split("|")]
    data: dict[str, dict[str, int]] = {c: defaultdict(int) for c in CFGS}
    for ln in lines[2:]:
        if not ln.startswith("|"):
            break
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        kind = cells[0]
        for cfg in CFGS:
            data[cfg][kind if kind in KEYS else "OTHER"] += int(cells[header.index(f"{cfg} instr")].replace(",", ""))
    totals = {c: sum(v.values()) for c, v in data.items()}

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6.6), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    x = range(len(CFGS))

    bottoms = [0.0] * len(CFGS)
    for key, label, col in GROUPS:
        vals = [data[c][key] / 1e9 for c in CFGS]
        ax1.bar(x, vals, bottom=bottoms, width=0.6, color=col, label=label, edgecolor="white", linewidth=0.6)
        for i, v in enumerate(vals):
            if v > 0.45:
                ax1.text(i, bottoms[i] + v / 2, f"{v:.2f} G", ha="center", va="center", fontsize=9,
                         color="white" if col in ("#b2182b", "#ef6548", "#2166ac") else "black")
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    base = totals[CFGS[0]]
    for i, c in enumerate(CFGS):
        t = totals[c] / 1e9
        sp = base / totals[c]
        ax1.text(i, t + 0.15, f"{t:.2f} G" + ("" if i == 0 else f"\n{sp:.2f}x vs {CFGS[0]}"), ha="center",
                 va="bottom", fontsize=11, fontweight="bold")
    ax1.set_xticks(list(x), CFGS, fontsize=12, fontweight="bold")
    ax1.set_ylabel("Retired instructions per image (G = 10^9)")
    ax1.set_ylim(0, max(totals.values()) / 1e9 * 1.18)
    ax1.set_title("Instructions per op kind", fontsize=13, fontweight="bold", loc="left")
    ax1.spines[["top", "right"]].set_visible(False)

    bottoms = [0.0] * len(CFGS)
    for key, label, col in GROUPS:
        vals = [100 * data[c][key] / totals[c] for c in CFGS]
        ax2.bar(x, vals, bottom=bottoms, width=0.6, color=col, edgecolor="white", linewidth=0.6)
        for i, v in enumerate(vals):
            if v > 3.5:
                ax2.text(i, bottoms[i] + v / 2, f"{v:.1f}%", ha="center", va="center", fontsize=9,
                         color="white" if col in ("#b2182b", "#ef6548", "#2166ac") else "black")
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    mac = {c: 100 * sum(data[c][k] for k in ("FULLY_CONNECTED", "BATCH_MATMUL", "CONV_2D")) / totals[c] for c in CFGS}
    for i, c in enumerate(CFGS):
        ax2.text(i, 101.5, f"MAC ops {mac[c]:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
                 color="#b2182b")
    ax2.set_xticks(list(x), CFGS, fontsize=12, fontweight="bold")
    ax2.set_ylabel("Share of the config's instructions (%)")
    ax2.set_ylim(0, 108)
    ax2.set_title("Share of each config's total", fontsize=13, fontweight="bold", loc="left")
    ax2.spines[["top", "right"]].set_visible(False)

    # third panel: MAC ops vs everything else, absolute
    mac_g = [sum(data[c][k] for k in ("FULLY_CONNECTED", "BATCH_MATMUL", "CONV_2D")) / 1e9 for c in CFGS]
    oth_g = [(totals[c] / 1e9) - m for c, m in zip(CFGS, mac_g)]
    w = 0.36
    ax3.bar([i - w / 2 for i in x], mac_g, width=w, color="#b2182b", label="Ops with MACs (FC, BMM, CONV)")
    ax3.bar([i + w / 2 for i in x], oth_g, width=w, color="#4393c3", label="All other ops")
    for i in x:
        ax3.text(i - w / 2, mac_g[i] + 0.15, f"{mac_g[i]:.2f} G", ha="center", fontsize=10, fontweight="bold",
                 color="#b2182b")
        ax3.text(i + w / 2, oth_g[i] + 0.15, f"{oth_g[i]:.2f} G", ha="center", fontsize=10, fontweight="bold",
                 color="#2166ac")
    ax3.set_xticks(list(x), CFGS, fontsize=12, fontweight="bold")
    ax3.set_ylabel("Retired instructions per image (G)")
    ax3.set_ylim(0, max(mac_g) * 1.18)
    ax3.set_title("MAC ops shrink, the rest stays", fontsize=13, fontweight="bold", loc="left")
    ax3.legend(fontsize=9, frameon=False, loc="upper right")
    ax3.spines[["top", "right"]].set_visible(False)

    h, l = ax1.get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="lower center", fontsize=9.5, frameon=False, ncol=4,
               bbox_to_anchor=(0.5, 0.025))
    fig.suptitle("DeiT-Tiny W8A8: where the instructions go in B0, B1 and X1", fontsize=15, fontweight="bold",
                 x=0.01, ha="left")
    fig.text(0.01, 0.005, "Spike minstret between op boundaries, 648 ops, one image. Reds = ops with MACs "
             "(xqmac8 targets); the other ops use the same code in every config.",
             fontsize=8.5, color="#444")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "op_division.png", dpi=140)
    print("wrote", OUT / "op_division.png")
    for c in CFGS:
        print(c, f"{totals[c]:,}", {k: f"{v / totals[c] * 100:.1f}%" for k, v in data[c].items()})


if __name__ == "__main__":
    main()
