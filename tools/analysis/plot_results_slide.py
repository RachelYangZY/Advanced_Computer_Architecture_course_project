#!/usr/bin/env python3
"""Results slide figure (16:9): total instructions, per-MAC instruction mix, target scorecard.

usage: plot_results_slide.py [OUTDIR]     -> OUTDIR/results_slide.png
Numbers: results/model/profile.md (totals) and results/model/instr_mix.md (class counts, GEMM kernel).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/figures")
MACS = 1_253_683_200
CFG = ["B0", "B1", "X1"]
LABEL = {"B0": "B0\nbaseline", "B1": "B1\noptimized software", "X1": "X1\n+ xqmac8, xrequant"}
TOTAL = {"B0": 12_598_660_515, "B1": 6_478_403_747, "X1": 1_522_226_359}  # profile.md
# instr_mix.md class counts
MIX = {
    "B0": dict(alu=6_724_219_399, load=2_743_229_112, multiply=1_308_455_097, branch=1_766_859_669 + 25_124_377,
               store=33_395_313, custom=0),
    "B1": dict(alu=3_731_091_068, load=1_137_401_796, multiply=1_283_092_751, branch=271_886_232 + 21_559_947,
               store=35_995_929, custom=0),
    "X1": dict(alu=854_832_050, load=310_263_680, multiply=17_026_747, branch=117_928_013 + 21_541_107,
               store=39_599_242, custom=157_682_496 + 5_974_636),
}
CLASSES = [("alu", "ALU (add, shift, address)", "#b0b0b0"), ("load", "load", "#1f77b4"),
           ("multiply", "multiply", "#9467bd"), ("branch", "branch / jump", "#8c564b"),
           ("store", "store", "#17becf"), ("custom", "xqmac8 + xrequant", "#d62728")]
GREEN, RED = "#2e7d32", "#d62728"


def main() -> None:
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    fig.text(0.04, 0.93, "Results: 8.28x fewer instructions, bit-exact with TFLite", fontsize=28,
             fontweight="bold")
    fig.text(0.04, 0.885, "One DeiT-Tiny inference (1.25 G MACs) on RV64IMC, Spike retired-instruction counts",
             fontsize=14, color="#444")

    # (a) total instructions
    ax = fig.add_axes([0.06, 0.42, 0.36, 0.39])
    vals = [TOTAL[c] / 1e9 for c in CFG]
    cols = ["#9aa0a6", "#5b8def", RED]
    bars = ax.bar(range(3), vals, color=cols, width=0.62)
    for i, c in enumerate(CFG):
        ax.text(i, vals[i] + 0.25, f"{vals[i]:.2f} G", ha="center", fontsize=16, fontweight="bold")
        sp = TOTAL["B0"] / TOTAL[c]
        ax.text(i, vals[i] / 2 if vals[i] > 1.2 else vals[i] + 1.2, f"{sp:.2f}x" if i else "1.00x",
                ha="center", va="center", fontsize=18, fontweight="bold",
                color="white" if vals[i] > 1.2 else RED)
    ax.set_xticks(range(3))
    ax.set_xticklabels([LABEL[c] for c in CFG], fontsize=12)
    ax.set_ylim(0, 14.5)
    ax.set_ylabel("instructions per inference (billions)", fontsize=12)
    ax.set_title("(a) Total instructions retired", fontsize=15, fontweight="bold", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(2, 3.4, f"{TOTAL['B1'] / TOTAL['X1']:.2f}x\nvs B1", fontsize=14, fontweight="bold", ha="center",
            color=RED)

    # (b) per-MAC mix
    ax = fig.add_axes([0.52, 0.42, 0.44, 0.39])
    for i, c in enumerate(CFG):
        left = 0.0
        for k, name, col in CLASSES:
            v = MIX[c][k] / MACS
            ax.barh(i, v, left=left, color=col, height=0.6, label=name if i == 0 else None)
            if v > 0.55:
                ax.text(left + v / 2, i, f"{v:.1f}", ha="center", va="center", fontsize=11,
                        color="white" if k != "alu" else "black", fontweight="bold")
            left += v
        ax.text(left + 0.15, i, f"{left:.1f}", va="center", fontsize=16, fontweight="bold")
    ax.set_yticks(range(3))
    ax.set_yticklabels(CFG, fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 11.5)
    ax.set_xlabel("instructions per MAC", fontsize=12)
    ax.set_title("(b) Where each MAC's instructions go", fontsize=15, fontweight="bold", loc="left")
    ax.legend(ncol=3, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False,
              columnspacing=1.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # scorecard
    cards = [("11.0x", "matrix kernel (gemm_nt_rows)", "B1 5.39 G -> X1 0.49 G instr.", "target > 2x"),
             ("8.28x", "end to end vs baseline B0", "4.26x vs optimized software B1", "target > 1.5x"),
             ("1000 / 1000", "output scores identical to TFLite", "bit-exact for B0, B1, X1", "correctness")]
    for i, (big, l1, l2, tag) in enumerate(cards):
        x = 0.04 + i * 0.32
        fig.patches.append(FancyBboxPatch((x, 0.07), 0.30, 0.17, boxstyle="round,pad=0.003,rounding_size=0.012",
                                          transform=fig.transFigure, fc="#f1f8f2", ec=GREEN, lw=2))
        fig.text(x + 0.015, 0.175, big, fontsize=28, fontweight="bold", color=GREEN)
        fig.text(x + 0.015, 0.135, l1, fontsize=13, fontweight="bold")
        fig.text(x + 0.015, 0.105, l2, fontsize=11, color="#444")
        fig.text(x + 0.285, 0.215, "met: " + tag, fontsize=10, color=GREEN, ha="right", fontweight="bold")

    fig.text(0.04, 0.025,
             "Instruction counts, not cycles: Spike retires one instruction per cycle, so real speedup needs the "
             "CV-Wally RTL (next).  xrequant adds only 3.7% (X1MAC 7.97x vs X1 8.28x).  Code size +7.6%.",
             fontsize=10.5, color="#444")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "results_slide.png", dpi=120)
    print("wrote", OUT / "results_slide.png")


if __name__ == "__main__":
    main()
