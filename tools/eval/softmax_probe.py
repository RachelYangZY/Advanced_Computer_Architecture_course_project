#!/usr/bin/env python3
"""Check candidate float32 models of TFLite's int8 softmax against dumped TFLite tensors.

    python tools/eval/softmax_probe.py build/ref_img        (run with ~/aca-venv: needs `tflite`)

TFLite (reference SoftmaxInt8LUT): table[255-v] = expf(-in_scale*beta*v); per row
sum = sequential float32 sum of table_offset[x]; inv = 1/(sum*out_scale);
q = int(table_offset[x]*inv + 0.5f) + zp.
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
ref = Path(sys.argv[1])


def table(s_in: float, how: str) -> np.ndarray:
    scale = f32(-f32(s_in))  # beta = 1
    v = np.arange(256, dtype=np.int32).astype(f32)
    arg = (scale * v).astype(f32)
    if how == "f32":
        e = np.exp(arg).astype(f32)
    else:  # double exp, rounded to float32
        e = np.exp(arg.astype(np.float64)).astype(f32)
    return e  # e[d] = expf(scale*d)


def run(x: np.ndarray, s_in: float, s_out: float, zp: int, how: str, fma: bool) -> np.ndarray:
    tab = table(s_in, how)
    rows = x.reshape(-1, x.shape[-1]).astype(np.int32)
    mx = rows.max(axis=1, keepdims=True)
    e = tab[mx - rows]
    ssum = np.cumsum(e, axis=1, dtype=f32)[:, -1:]
    inv = (f32(1.0) / (ssum * f32(s_out)).astype(f32)).astype(f32)
    if fma:
        p = (e.astype(np.float64) * inv.astype(np.float64) + 0.5).astype(f32)
    else:
        p = ((e * inv).astype(f32) + f32(0.5)).astype(f32)
    q = p.astype(np.int32) + zp
    return np.clip(q, -128, 127).astype(np.int8).reshape(x.shape)


tot = {}
for op in m.ops:
    if op.kind != "SOFTMAX":
        continue
    ti, to = m.tensors[op.inputs[0]], m.tensors[op.outputs[0]]
    x = np.fromfile(ref / f"t{ti.idx}.bin", dtype=np.int8).reshape(ti.shape)
    y = np.fromfile(ref / f"t{to.idx}.bin", dtype=np.int8).reshape(to.shape)
    for how in ("f32", "f64"):
        for fma in (False, True):
            got = run(x, float(ti.scales[0]), float(to.scales[0]), int(to.zero_points[0]), how, fma)
            tot[(how, fma)] = tot.get((how, fma), 0) + int((got != y).sum())
print("total mismatching elements over all softmax ops:")
for k, v in tot.items():
    print(f"  table={k[0]} fma={k[1]}: {v}")
