#!/usr/bin/env python3
"""Which formula does TFLite use for the int8 MEAN op?

    python tools/eval/mean_probe.py <ref dir> [<ref dir> ...]        (~/aca-venv)

Candidates: (A) reference_ops::QuantizedMeanOrSum in float32:
    float_mean = float(sum) / float(n); scale = in_scale / out_scale; bias = -in_zp * scale
    q = round_half_away(float_mean * scale + bias) + out_zp
(A2) the same with the multiply-add fused (single rounding); (B) exact rational rounding of
(sum/n - in_zp) * in_scale / out_scale; (C) gemmlowp two-stage as in the generator.
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tflite2c"))
import tflite_ir  # noqa: E402
from gen_model import quantize_multiplier, scale_zp  # noqa: E402

f32 = np.float32
ROOT = Path(__file__).resolve().parents[2]
m = tflite_ir.load(ROOT / "models/deit_tiny_patch16_224/deit_tiny_w8a8_pilot.tflite")


def rha(x):  # round half away from zero
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def gemmlowp(centered, mult, shift):
    left, right = max(shift, 0), max(-shift, 0)
    ab = (centered << left) * mult
    t = ab + np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    h = np.where(t >= 0, t // (1 << 31), -((-t) // (1 << 31)))
    if right:
        mask = (1 << right) - 1
        h = (h >> right) + ((h & mask) > ((mask >> 1) + (h < 0)))
    return h


rows = []
for ref in map(Path, sys.argv[1:]):
    for op in m.ops:
        if op.kind != "MEAN":
            continue
        ti, to = m.tensors[op.inputs[0]], m.tensors[op.outputs[0]]
        axes = tuple(int(a) % len(ti.shape) for a in np.atleast_1d(m.tensors[op.inputs[1]].data))
        x = np.fromfile(ref / f"t{ti.idx}.bin", dtype=np.int8).reshape(ti.shape).astype(np.int64)
        want = np.fromfile(ref / f"t{to.idx}.bin", dtype=np.int8).reshape(-1).astype(np.int64)
        n = int(np.prod([ti.shape[a] for a in axes]))
        (si, zi), (so, zo) = scale_zp(ti), scale_zp(to)
        s = x.sum(axis=axes).reshape(-1)
        scale = f32(f32(si) / f32(so))
        bias = f32(f32(-zi) * scale)
        fmean = (s.astype(f32) / f32(n)).astype(f32)
        a = rha((((fmean * scale).astype(f32)) + bias).astype(f32).astype(np.float64))
        a2 = rha((fmean.astype(np.float64) * float(scale) + float(bias)).astype(f32).astype(np.float64))
        b = np.array([int(np.floor(q + Fraction(1, 2))) if q >= 0 else -int(np.floor(-q + Fraction(1, 2))) for q in
                      [(Fraction(int(v) - zi * n, n) * Fraction(si) / Fraction(so)) for v in s]])
        mu, sh = quantize_multiplier(float(si) / float(so) / n)
        c = gemmlowp(s - zi * n, mu, sh)
        res = []
        for q in (a, a2, b, c):
            got = np.clip(q.astype(np.int64) + zo, -128, 127)
            res.append(int((got != want).sum()))
        rows.append((op.idx, n, float(si) / float(so), zi, zo, *res))
import collections
agg = collections.OrderedDict()
for r in rows:
    k = r[0]
    if k not in agg:
        agg[k] = list(r)
    else:
        for i in range(5, 9):
            agg[k][i] += r[i]
print("op   n    in/out  zi zo | A(f32) A2(fma) B(exact) C(gemmlowp) mismatches summed over images")
for r in agg.values():
    flag = "" if min(r[5:9]) == 0 else "   <-- none exact"
    print(f"{r[0]:4d} {r[1]:4d} {r[2]:7.2f} {r[3]:3d} {r[4]:3d} | {r[5]:5d} {r[6]:5d} {r[7]:5d} {r[8]:5d}{flag}")