#!/usr/bin/env python3
"""Which real-multiplier formula does TFLite use for BATCH_MATMUL?

    python tools/eval/bmm_probe.py <ref dir> [<ref dir> ...]        (~/aca-venv)

For every BATCH_MATMUL op and every TFLite tensor dump given (tools/eval/bisect_image.py), requantize
the int32 accumulators with each candidate real multiplier (gemmlowp two-stage rounding) and count
mismatches against TFLite's int8 output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tflite2c"))
import tflite_ir  # noqa: E402
from gen_model import quantize_multiplier, scale_zp  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
m = tflite_ir.load(ROOT / "models/deit_tiny_patch16_224/deit_tiny_w8a8_pilot.tflite")
f32 = np.float32


def requant(acc, mult, shift, zo):
    left, right = max(shift, 0), max(-shift, 0)
    ab = (acc << left) * mult
    t = ab + np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    h = np.where(t >= 0, t // (1 << 31), -((-t) // (1 << 31)))   # srdhm, truncate toward zero
    if right:
        mask = (1 << right) - 1
        h = (h >> right) + ((h & mask) > ((mask >> 1) + (h < 0)))
    return np.clip(h + zo, -128, 127)


tot: dict[str, int] = {}
elems = 0
for ref in map(Path, sys.argv[1:]):
    for op in m.ops:
        if op.kind != "BATCH_MATMUL":
            continue
        ta, tb, to = (m.tensors[i] for i in (op.inputs[0], op.inputs[1], op.outputs[0]))
        ld = lambda t: (t.data if t.is_const else
                        np.fromfile(ref / f"t{t.idx}.bin", dtype=np.int8).reshape(t.shape)).astype(np.int64)
        A, B = ld(ta), ld(tb)
        if op.options.get("adj_x"):
            A = A.transpose(0, 2, 1)
        if op.options.get("adj_y"):
            B = B.transpose(0, 2, 1)
        (sa, za), (sb, zb), (so, zo) = scale_zp(ta), scale_zp(tb), scale_zp(to)
        acc = np.einsum("bik,bkj->bij", A - za, B - zb)
        want = np.fromfile(ref / f"t{to.idx}.bin", dtype=np.int8).reshape(to.shape).astype(np.int64)
        variants = {
            "double(sa)*double(sb)/double(so)": float(sa) * float(sb) / float(so),
            "float32(sa*sb) / float32(so) (all float)": float(f32(f32(sa) * f32(sb)) / f32(so)),
            "float32(sa*sb), double divide": float(f32(sa) * f32(sb)) / float(so),
        }
        for name, real in variants.items():
            mu, sh = quantize_multiplier(real)
            tot[name] = tot.get(name, 0) + int((requant(acc, mu, sh, zo) != want).sum())
        elems += want.size
print(f"{elems} BMM output elements checked")
for k, v in tot.items():
    print(f"  {k:45s} mismatches: {v}")
