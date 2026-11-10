#!/usr/bin/env python3
"""Score RSQRT lookup-table models against pair counts from tools/eval/rsqrt_collect.py.

    python tools/eval/rsqrt_fit.py        (~/aca-venv, needs `tflite` via tflite_ir)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tflite2c"))
import tflite_ir  # noqa: E402
from gen_model import round_half_away, scale_zp  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
f32 = np.float32
m = tflite_ir.load(ROOT / "models/deit_tiny_patch16_224/deit_tiny_w8a8_pilot.tflite")
z = np.load(ROOT / "build/rsqrt_pairs.npz")
counts = z["counts"]
ops = [op for op in m.ops if op.kind == "RSQRT"]
assert len(ops) == len(counts)
vals = np.arange(-128, 128, dtype=np.int32)


def lut(op, how: str, b: float = 0.0) -> np.ndarray:
    ti, to = m.tensors[op.inputs[0]], m.tensors[op.outputs[0]]
    (si, zi), (so, zo) = scale_zp(ti), scale_zp(to)
    with np.errstate(all="ignore"):
        if how == "f32":
            deq = f32(si) * (vals - zi).astype(f32)
            tr = (f32(1.0) / np.sqrt(deq)).astype(f32) * f32(1.0 + b)
            resc = round_half_away(tr * (f32(1.0) / f32(so)))
        elif how == "f32_div":
            deq = f32(si) * (vals - zi).astype(f32)
            tr = (f32(1.0) / np.sqrt(deq)).astype(f32) * f32(1.0 + b)
            resc = round_half_away((tr / f32(so)).astype(f32))
        elif how == "f64":
            deq = float(si) * (vals - zi).astype(np.float64)
            tr = (1.0 / np.sqrt(deq)) * (1.0 + b)
            resc = round_half_away(tr / float(so))
    resc = np.where(np.isfinite(resc), resc, -(2.0 ** 31)).astype(np.int64)
    return np.clip(resc + zo, -128, 127).astype(np.int8)


def score(how: str, b: float = 0.0, per_op: bool = False):
    bad = 0
    tot = 0
    rows = []
    for k, op in enumerate(ops):
        t = lut(op, how, b).astype(int) + 128
        c = counts[k]
        ok = c[np.arange(256), t].sum()
        n = c.sum()
        bad += n - ok
        tot += n
        rows.append(int(n - ok))
    return (int(bad), int(tot), rows) if per_op else (int(bad), int(tot))


for how in ("f32", "f32_div", "f64"):
    print(how, "b=0:", score(how))
print("f32 b=1e-5 (current):", score("f32", 1e-5))
best = min(((score("f32", b)[0], b) for b in np.linspace(-5e-5, 5e-5, 101)))
print("best constant bias in [-5e-5, 5e-5]:", best)
print("per-op mismatches at b=1e-5:", score("f32", 1e-5, True)[2])
