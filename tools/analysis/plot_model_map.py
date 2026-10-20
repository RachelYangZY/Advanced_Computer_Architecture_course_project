#!/usr/bin/env python3
"""Draw the DeiT-Tiny TFLite graph -> generated C kernels -> dynamic instruction mix, from op_mix.json.

usage: plot_model_map.py [results/model/op_mix.json] [OUTDIR]
Writes OUTDIR/model_map.png (block 0 stage by stage, X1) and OUTDIR/instr_breakdown.png (whole model,
why xrequant is small) plus results/model/op_mix.md (the numbers behind both).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

JSON = sys.argv[1] if len(sys.argv) > 1 else "results/model/op_mix.json"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/figures")

GROUPS = ["xqmac8", "xrequant", "multiply", "load", "store", "alu", "branch/jump", "other"]
COLORS = {"xqmac8": "#d62728", "xrequant": "#ff7f0e", "multiply": "#9467bd", "load": "#1f77b4",
          "store": "#17becf", "alu": "#b0b0b0", "branch/jump": "#8c564b", "other": "#e0e0e0"}
MERGE = {"load": "load", "store": "store", "alu": "alu", "multiply": "multiply", "branch": "branch/jump",
         "jump": "branch/jump", "xqmac8": "xqmac8", "xrequant": "xrequant", "csr": "other", "divide": "other"}

# (stage, op range in block 0 [lo, hi], TFLite ops, generated C function -> inner kernel)
BLOCK = [
    ("LayerNorm 1: mean / variance", 8, 11, "MEAN, SQUARED_DIFFERENCE, MEAN", "op_mean_i8, op_sqdiff_i8"),
    ("LayerNorm 1: 1/sqrt(var+eps)", 12, 15, "ADD, RSQRT, MUL, SUB", "op_add_i8, op_lut_i8, op_mul_i8, op_sub_i8"),
    ("LayerNorm 1: normalize, gamma, beta", 16, 19, "MUL, ADD", "op_mul_i8, op_add_i8"),
    ("QKV projection (197x192 -> 576)", 20, 20, "FULLY_CONNECTED", "op_fc_i8 -> gemm_nt_rows"),
    ("Split heads: transpose, slice q/k/v", 21, 27, "TRANSPOSE, 3x SLICE", "op_transpose_i8, op_slice_i8"),
    ("Scale q, k; transpose k", 28, 32, "MUL, MUL, TRANSPOSE", "op_mul_i8, op_transpose_i8"),
    ("Scores  Q x K^T  (3 heads)", 33, 33, "BATCH_MATMUL", "op_bmm_i8 -> gemm_nt_rows"),
    ("Softmax (197x197, 3 heads)", 34, 36, "SOFTMAX", "op_softmax_i8"),
    ("Context  P x V  (3 heads)", 37, 38, "BATCH_MATMUL", "op_bmm_i8 -> gemm_nt_rows"),
    ("Merge heads: transpose", 39, 41, "TRANSPOSE", "op_transpose_i8"),
    ("Output projection (192 -> 192)", 42, 42, "FULLY_CONNECTED", "op_fc_i8 -> gemm_nt_rows"),
    ("Residual add 1", 43, 43, "ADD", "op_add_i8"),
    ("LayerNorm 2: mean / variance", 44, 47, "MEAN, SQUARED_DIFFERENCE, MEAN", "op_mean_i8, op_sqdiff_i8"),
    ("LayerNorm 2: 1/sqrt(var+eps)", 48, 51, "ADD, RSQRT, MUL, SUB", "op_add_i8, op_lut_i8, op_mul_i8, op_sub_i8"),
    ("LayerNorm 2: normalize, gamma, beta", 52, 55, "MUL, ADD", "op_mul_i8, op_add_i8"),
    ("MLP fc1 (192 -> 768)", 56, 56, "FULLY_CONNECTED", "op_fc_i8 -> gemm_nt_rows"),
    ("GELU", 57, 57, "GELU", "op_lut_i8 (256-entry table)"),
    ("MLP fc2 (768 -> 192)", 58, 58, "FULLY_CONNECTED", "op_fc_i8 -> gemm_nt_rows"),
    ("Residual add 2", 59, 59, "ADD", "op_add_i8"),
]
STEM = [
    ("Patch embed: transpose, conv 16x16/16", 0, 4, "TRANSPOSE, CONV_2D, ...", "op_transpose_i8, op_conv2d_i8 -> gemm_nt_rows"),
    ("CLS token + positional embedding", 5, 7, "QUANTIZE, CONCATENATION, ADD", "op_requant_i8, op_concat_part_i8, op_add_i8"),
]


def fmt(n: float) -> str:
    return f"{n / 1e6:,.2f} M" if n >= 1e5 else f"{n:,.0f}"


def mix_of(op: dict) -> dict[str, int]:
    m = dict.fromkeys(GROUPS, 0)
    for c, v in op["mix"].items():
        m[MERGE[c]] += v
    return m


def stage_rows(data: dict, stages) -> list[dict]:
    rows = []
    for name, lo, hi, tfl, cfn in stages:
        row = {"name": name, "lo": lo, "hi": hi, "tfl": tfl, "cfn": cfn, "cfg": {}}
        for cfg, d in data["cfgs"].items():
            ops = [o for o in d["ops"] if lo <= o["op"] <= hi]
            mix = dict.fromkeys(GROUPS, 0)
            for o in ops:
                for g, v in mix_of(o).items():
                    mix[g] += v
            row["cfg"][cfg] = {"total": sum(o["instret"] for o in ops), "mix": mix,
                               "macs": sum(o["macs"] for o in ops)}
        rows.append(row)
    return rows


def draw_map(data: dict, stem: list[dict], block: list[dict], path: Path) -> None:
    rows = stem + block
    n = len(rows)
    fig, ax = plt.subplots(figsize=(24, 0.62 * n + 4.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(-4.2, n + 2.9)
    ax.axis("off")
    ax.set_title("DeiT-Tiny (W8A8) on RV64IMC + xqmac8/xrequant: TFLite graph -> generated C -> instructions retired (config X1)",
                 fontsize=16, fontweight="bold", loc="left")
    xmax = max(r["cfg"]["X1"]["total"] for r in rows)
    bx0, bw = 42.0, 17.0
    heads = [(1, "TFLite stage (op ids)"), (21.0, "Generated C kernel -> inner loop"),
             (bx0, "X1 instructions (linear)"), (66.0, "X1 mix (%)"),
             (81, "Millions:  B0 | B1 | X1")]
    for x, t in heads:
        ax.text(x, n + 1.1, t, fontsize=11, fontweight="bold", va="bottom")
    ax.plot([0, 100], [n + 1.05, n + 1.05], color="black", lw=1)

    for i, r in enumerate(rows):
        y = n - 1 - i - (0.7 if i >= len(stem) else 0)
        x1 = r["cfg"]["X1"]
        in_stem = i < len(stem)
        if i == len(stem):
            ax.plot([0, 100], [y + 1.2, y + 1.2], color="#555", lw=1.2, ls="--")
            ax.text(0.5, y + 1.25, "Transformer block 0 (repeated 12x; blocks differ by < 0.1%)", ha="left",
                    va="bottom", fontsize=10, style="italic", color="#444")
        ax.add_patch(FancyBboxPatch((0.5, y + 0.12), 19.0, 0.76, boxstyle="round,pad=0.02",
                                    fc="#e8f0fe" if not in_stem else "#fdebd0", ec="#456", lw=0.9))
        ops = f"ops {r['lo']}" if r["lo"] == r["hi"] else f"ops {r['lo']}-{r['hi']}"
        ax.text(0.9, y + 0.62, r["name"], fontsize=8.2, fontweight="bold", va="center")
        ax.text(0.9, y + 0.30, f"{ops}: {r['tfl']}", fontsize=7.0, va="center", color="#234")
        ax.annotate("", xy=(20.9, y + 0.5), xytext=(19.7, y + 0.5), arrowprops=dict(arrowstyle="->", color="#456"))
        ax.add_patch(FancyBboxPatch((21.0, y + 0.12), 19.4, 0.76, boxstyle="round,pad=0.02", fc="#eafaf1",
                                    ec="#385", lw=0.9))
        ax.text(21.3, y + 0.5, r["cfn"], fontsize=8, va="center", family="monospace")
        # absolute stacked bar
        x = bx0
        for g in GROUPS:
            w = bw * x1["mix"][g] / xmax
            if w > 0:
                ax.add_patch(plt.Rectangle((x, y + 0.2), w, 0.6, fc=COLORS[g], ec="none"))
                x += w
        ax.text(x + 0.3, y + 0.5, fmt(x1["total"]), fontsize=8, va="center")
        # normalized bar
        x = 66.0
        for g in GROUPS:
            w = 13.0 * x1["mix"][g] / max(1, x1["total"])
            if w > 0:
                ax.add_patch(plt.Rectangle((x, y + 0.2), w, 0.6, fc=COLORS[g], ec="none"))
                x += w
        b0, b1 = r["cfg"]["B0"]["total"], r["cfg"]["B1"]["total"]
        ax.text(81, y + 0.66, f"{b0 / 1e6:8.2f} | {b1 / 1e6:8.2f} | {x1['total'] / 1e6:7.2f}", fontsize=8,
                va="center", family="monospace")
        ax.text(81, y + 0.30, f"xqmac8 {fmt(x1['mix']['xqmac8']):>9}  xrequant {fmt(x1['mix']['xrequant']):>9}",
                fontsize=7.4, va="center", family="monospace", color="#a33" if x1["mix"]["xqmac8"] else "#555")

    # totals row
    ty = -1.9
    tb = {c: sum(r["cfg"][c]["total"] for r in block) for c in ("B0", "B1", "X1")}
    ts = {c: sum(r["cfg"][c]["total"] for r in stem) for c in ("B0", "B1", "X1")}
    xq = sum(r["cfg"]["X1"]["mix"]["xqmac8"] for r in block)
    xr = sum(r["cfg"]["X1"]["mix"]["xrequant"] for r in block)
    ax.plot([0, 100], [ty + 0.9, ty + 0.9], color="black", lw=1)
    ax.text(1, ty + 0.45, "Block 0 total", fontsize=10, fontweight="bold", va="center")
    ax.text(81, ty + 0.45, f"{tb['B0'] / 1e6:8.2f} | {tb['B1'] / 1e6:8.2f} | {tb['X1'] / 1e6:7.2f}", fontsize=9,
            fontweight="bold", va="center", family="monospace")
    ax.text(1, ty - 0.55, "Whole inference = stem + 12 x block + final norm/head (full numbers: results/model/profile.md).  "
            f"xqmac8 in block 0: {fmt(xq)} ({100 * xq / tb['X1']:.1f}% of its instructions); "
            f"xrequant: {fmt(xr)} ({100 * xr / tb['X1']:.2f}%).  Stem total X1: {fmt(ts['X1'])}.",
            fontsize=9, va="center")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[g]) for g in GROUPS]
    ax.legend(handles, GROUPS, ncol=8, loc="lower center", bbox_to_anchor=(0.45, -0.03), fontsize=10, frameon=False)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def parse_mix_table(path: str) -> dict[str, dict[str, int]]:
    out, cols = {}, None
    for line in Path(path).read_text().splitlines():
        if line.startswith("| config | total"):
            cols = [c.strip() for c in line.strip("|").split("|")][2:]
        elif cols and line.startswith("| ") and not line.startswith("| config") and not line.startswith("|---"):
            p = [c.strip() for c in line.strip("|").split("|")]
            if p[0] in ("B0", "B1", "X1", "X1MAC") and p[0] not in out and "." not in p[2]:
                out[p[0]] = {c: int(v.replace(",", "")) for c, v in zip(cols, p[2:])}
                out[p[0]]["total"] = int(p[1].replace(",", ""))
        elif cols and not line.strip():
            if len(out) >= 4:
                break
    return out


def draw_breakdown(data: dict, block: list[dict], stem: list[dict], path: Path) -> list[str]:
    whole = parse_mix_table("results/model/instr_mix.md")
    fig = plt.figure(figsize=(22, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.38, wspace=0.18)

    # (a) whole-model class totals, exact
    ax = fig.add_subplot(gs[0, 0])
    cfgs = ["B0", "B1", "X1"]
    for j, c in enumerate(cfgs):
        left = 0
        m = dict.fromkeys(GROUPS, 0)
        for k, v in whole[c].items():
            if k != "total":
                m[MERGE[k]] += v
        for g in GROUPS:
            ax.barh(j, m[g] / 1e9, left=left, color=COLORS[g], label=g if j == 0 else None)
            if m[g] / 1e9 > 0.6:
                ax.text(left + m[g] / 2e9, j, f"{100 * m[g] / whole[c]['total']:.0f}%", ha="center", va="center",
                        fontsize=9, color="white" if g not in ("alu", "other") else "black")
            left += m[g] / 1e9
        ax.text(left + 0.1, j, f"{whole[c]['total'] / 1e9:.2f} G", va="center", fontsize=11, fontweight="bold")
    ax.set_yticks(range(3))
    ax.set_yticklabels(cfgs, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("instructions retired per inference (billions), exact PC histogram")
    ax.set_title("(a) Whole model: what the retired instructions are", fontsize=13, fontweight="bold", loc="left")
    ax.legend(ncol=4, fontsize=9, loc="lower right")
    x1 = whole["X1"]
    ax.text(0.99, 0.5, f"X1: xqmac8 {x1['xqmac8'] / 1e6:,.1f} M = {100 * x1['xqmac8'] / x1['total']:.1f}%\n"
            f"X1: xrequant {x1['xrequant'] / 1e6:,.1f} M = {100 * x1['xrequant'] / x1['total']:.2f}%",
            transform=ax.transAxes, ha="right", va="center", fontsize=11, family="monospace",
            bbox=dict(fc="white", ec="#c33"))

    # (b) per kernel kind, X1, block 0 + stem scaled to model
    ax = fig.add_subplot(gs[0, 1])
    kinds: dict[str, dict[str, float]] = {}
    for d in data["cfgs"]["X1"]["ops"]:
        k = d["kind"]
        if k == "RESHAPE":
            continue
        mult = 12 if d["op"] >= 8 else 1
        for g, v in mix_of(d).items():
            kinds.setdefault(k, dict.fromkeys(GROUPS, 0))[g] += v * mult
    order = sorted(kinds, key=lambda k: -sum(kinds[k].values()))
    for j, k in enumerate(order):
        left = 0
        for g in GROUPS:
            ax.barh(j, kinds[k][g] / 1e6, left=left, color=COLORS[g])
            left += kinds[k][g] / 1e6
        ax.text(left + 4, j, f"{left:,.0f} M  ({100 * left * 1e6 / whole['X1']['total']:.1f}%)", va="center", fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("instructions (millions), X1; = stem + 12 x block 0 (final norm and head not included)")
    ax.set_title("(b) X1 by TFLite operator kind (colors as in (a))", fontsize=13, fontweight="bold", loc="left")

    # (c) per FC output: what one requantized output costs
    ax = fig.add_subplot(gs[1, 0])
    fcs = [("qkv\nK=192", 20, 197 * 576), ("proj\nK=192", 42, 197 * 192), ("fc1\nK=192", 56, 197 * 768),
           ("fc2\nK=768", 58, 197 * 192)]
    ops = {o["op"]: o for o in data["cfgs"]["X1"]["ops"]}
    for j, (lab, oi, nout) in enumerate(fcs):
        m = mix_of(ops[oi])
        left = 0
        for g in GROUPS:
            v = m[g] / nout
            ax.bar(j, v, bottom=left, color=COLORS[g])
            if v > 2.5:
                ax.text(j, left + v / 2, f"{g} {v:.1f}", ha="center", va="center", fontsize=9,
                        color="white" if g not in ("alu", "other") else "black")
            left += v
        ax.text(j, left + 1, f"{left:.1f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(fcs)))
    ax.set_xticklabels([f[0] for f in fcs], fontsize=11)
    ax.set_ylabel("instructions per output element (X1)")
    ax.set_ylim(0, 370)
    ax.annotate("xrequant: exactly 1 per output", xy=(0, 25), xytext=(0.45, 190), fontsize=11,
                color="#c60", arrowprops=dict(arrowstyle="->", color="#c60"))
    ax.set_title("(c) One FC output in X1: one xrequant vs the rest of the work",
                 fontsize=13, fontweight="bold", loc="left")

    # (d) per-function breakdown X1 (block 0 + stem scaled)
    ax = fig.add_subplot(gs[1, 1])
    funcs: dict[str, dict[str, float]] = {}
    for d in data["cfgs"]["X1"]["ops"]:
        mult = 12 if d["op"] >= 8 else 1
        for f, mm in d["funcs"].items():
            for c, v in mm.items():
                funcs.setdefault(f, dict.fromkeys(GROUPS, 0))[MERGE[c]] += v * mult
    forder = sorted(funcs, key=lambda f: -sum(funcs[f].values()))[:10]
    for j, f in enumerate(forder):
        left = 0
        for g in GROUPS:
            ax.barh(j, funcs[f][g] / 1e6, left=left, color=COLORS[g])
            left += funcs[f][g] / 1e6
        xr = funcs[f]["xrequant"]
        ax.text(left + 4, j, f"{left:,.0f} M" + (f"   [xrequant {xr / 1e6:.1f} M]" if xr else ""), va="center", fontsize=9)
    ax.set_yticks(range(len(forder)))
    ax.set_yticklabels([f.replace(".constprop.0", "") for f in forder], fontsize=9, family="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("instructions (millions), X1, by C function")
    ax.set_title("(d) X1 by C function: xrequant lives only inside op_fc_i8 / op_bmm_i8", fontsize=13,
                 fontweight="bold", loc="left")
    fig.suptitle("Why xrequant is a small part of the instruction stream", fontsize=17, fontweight="bold", y=0.95)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    est_total = sum(sum(v.values()) for v in kinds.values())
    return [f"(b)/(d) estimate (stem + 12 x block 0): {est_total / 1e6:,.0f} M of {whole['X1']['total'] / 1e6:,.0f} M exact."]


def write_md(stem: list[dict], block: list[dict], path: Path) -> None:
    L = ["# Block-0 instruction counts by stage (Spike, retired instructions)\n",
         "Per-op histogram differencing (`tools/analysis/op_mix.py`). Block 0 repeats 12x per inference.\n",
         "| stage | ops | B0 | B1 | X1 | X1 load | X1 alu | X1 mult | X1 branch/jump | X1 store | xqmac8 | xrequant |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in stem + block:
        x = r["cfg"]["X1"]
        m = x["mix"]
        L.append(f"| {r['name']} | {r['lo']}-{r['hi']} | {r['cfg']['B0']['total']:,} | {r['cfg']['B1']['total']:,} | "
                 f"{x['total']:,} | {m['load']:,} | {m['alu']:,} | {m['multiply']:,} | {m['branch/jump']:,} | "
                 f"{m['store']:,} | {m['xqmac8']:,} | {m['xrequant']:,} |")
    path.write_text("\n".join(L) + "\n")


def main() -> None:
    data = json.loads(Path(JSON).read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    stem, block = stage_rows(data, STEM), stage_rows(data, BLOCK)
    draw_map(data, stem, block, OUT / "model_map.png")
    for line in draw_breakdown(data, block, stem, OUT / "instr_breakdown.png"):
        print(line)
    write_md(stem, block, Path("results/model/op_mix.md"))
    print("wrote", OUT / "model_map.png", OUT / "instr_breakdown.png", "results/model/op_mix.md")


if __name__ == "__main__":
    main()
