#!/usr/bin/env python3
"""Inspect softmax elements that differ from TFLite and test alternative summation models.

    python tools/eval/softmax_probe2.py <ref dir>        (~/aca-venv)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tflite2c"))
import tflite_ir  # noqa: E402

f32 = np.float32
ROOT = Path(__file__).resolve().parents[2]
m = tflite_ir.load(ROOT / "models/deit_tiny_patch16_224/deit_tiny_w8a8_pilot.tflite")
refs = [Path(a) for a in sys.argv[1:]]


def model(x, s_in, s_out, zp, mode, fma=False):
    scale = f32(-f32(s_in))
    tab = np.exp((scale * np.arange(256, dtype=np.int32).astype(f32)).astype(f32).astype(np.float64)).astype(f32)
    rows = x.reshape(-1, x.shape[-1]).astype(np.int32)
    mx = rows.max(axis=1, keepdims=True)
    e = tab[mx - rows]
    if mode == "seq32":
        s = np.cumsum(e, axis=1, dtype=f32)[:, -1:]
    elif mode == "f64":
        s = e.astype(np.float64).sum(axis=1, keepdims=True).astype(f32)
    elif mode == "pairwise32":
        s = e.sum(axis=1, keepdims=True, dtype=f32)
    elif mode == "simd4":   # 4 interleaved float32 accumulators, combined at the end
        n = e.shape[1]
        pad = (-n) % 4
        ep = np.pad(e, ((0, 0), (0, pad)))
        acc = np.zeros((e.shape[0], 4), dtype=f32)
        for k in range(0, ep.shape[1], 4):
            acc = (acc + ep[:, k:k + 4]).astype(f32)
        s = ((acc[:, 0] + acc[:, 1]).astype(f32) + (acc[:, 2] + acc[:, 3]).astype(f32)).astype(f32)[:, None]
    elif mode == "simd8":
        n = e.shape[1]
        pad = (-n) % 8
        ep = np.pad(e, ((0, 0), (0, pad)))
        acc = np.zeros((e.shape[0], 8), dtype=f32)
        for k in range(0, ep.shape[1], 8):
            acc = (acc + ep[:, k:k + 8]).astype(f32)
        t = (acc[:, :4] + acc[:, 4:]).astype(f32)
        s = ((t[:, 0] + t[:, 1]).astype(f32) + (t[:, 2] + t[:, 3]).astype(f32)).astype(f32)[:, None]
    inv = (f32(1.0) / (s * f32(s_out)).astype(f32)).astype(f32)
    if fma:
        p = (e.astype(np.float64) * inv.astype(np.float64) + 0.5).astype(f32)
    else:
        p = ((e * inv).astype(f32) + f32(0.5)).astype(f32)
    return np.clip(p.astype(np.int32) + zp, -128, 127).astype(np.int8).reshape(x.shape), e, s, inv


tot: dict[str, int] = {}
for ref in refs:
    for op in m.ops:
        if op.kind != "SOFTMAX":
            continue
        ti, to = m.tensors[op.inputs[0]], m.tensors[op.outputs[0]]
        x = np.fromfile(ref / f"t{ti.idx}.bin", dtype=np.int8).reshape(ti.shape)
        y = np.fromfile(ref / f"t{to.idx}.bin", dtype=np.int8).reshape(to.shape)
        for mode in ("seq32", "f64", "pairwise32", "simd4", "simd8"):
            for fma in (False, True):
                got = model(x, float(ti.scales[0]), float(to.scales[0]), int(to.zero_points[0]), mode, fma)[0]
                k = f"{mode} fma={fma}"
                tot[k] = tot.get(k, 0) + int((got != y).sum())
for k, v in tot.items():
    print(f"{k:22s} {v}")