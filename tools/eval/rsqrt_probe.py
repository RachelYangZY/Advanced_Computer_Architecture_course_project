#!/usr/bin/env python3
"""Constrain the relative bias of TFLite's int8 RSQRT from observed (input, output) pairs.

    python tools/eval/rsqrt_probe.py <ref dir> [<ref dir> ...]        (~/aca-venv)

For every RSQRT op: rescaled(v) = 1/sqrt(s_in*(v - zp_in)) / s_out. TFLite output k = round(rescaled*(1+b))
+ zp_out. Each observed pair bounds b to [(k-0.5)/r - 1, (k+0.5)/r - 1); report the intersection per
op and globally.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tflite2c"))
import tflite_ir  # noqa: E402
from gen_model import scale_zp  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
m = tflite_ir.load(ROOT / "models/deit_tiny_patch16_224/deit_tiny_w8a8_pilot.tflite")
f32 = np.float32
glo, ghi = -1.0, 1.0
pairs_total = 0
per_op = []
for op in m.ops:
    if op.kind != "RSQRT":
        continue
    ti, to = m.tensors[op.inputs[0]], m.tensors[op.outputs[0]]
    (si, zi), (so, zo) = scale_zp(ti), scale_zp(to)
    seen = {}
    for ref in map(Path, sys.argv[1:]):
        x = np.fromfile(ref / f"t{ti.idx}.bin", dtype=np.int8).astype(int)
        y = np.fromfile(ref / f"t{to.idx}.bin", dtype=np.int8).astype(int)
        for a, b in zip(x, y):
            seen[(int(a), int(b))] = 1
    lo, hi = -1.0, 1.0
    for (a, b) in seen:
        d = float(f32(si)) * (a - zi)
        if d <= 0:
            continue
        r = 1.0 / np.sqrt(d) / float(f32(so))
        k = b - zo
        if b in (-128, 127):
            continue
        lo = max(lo, (k - 0.5) / r - 1.0)
        hi = min(hi, (k + 0.5) / r - 1.0)
    pairs_total += len(seen)
    per_op.append((op.idx, len(seen), lo, hi))
    print(f"op {op.idx:3d} pairs {len(seen):4d} s_in {si:.6g} zp_in {zi} s_out {so:.6g} zp_out {zo}  b in [{lo:.3e}, {hi:.3e})")
    glo, ghi = max(glo, lo), min(ghi, hi)
bad = [p for p in per_op if p[2] >= p[3]]
print(f"{pairs_total} distinct (input, output) pairs over {len(per_op)} RSQRT ops")
print("ops with an empty interval (no single bias fits):", [(p[0], p[2], p[3]) for p in bad])
widths = sorted(per_op, key=lambda p: p[3] - p[2])[:3]
print("tightest per-op intervals:", [(p[0], f"{p[2]:.3e}", f"{p[3]:.3e}") for p in widths])
print(f"global intersection: [{glo:.3e}, {ghi:.3e})  -> {'EMPTY' if glo >= ghi else 'non-empty'}")
