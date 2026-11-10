#!/usr/bin/env python3
"""Collect (input, output) pair counts of every RSQRT op over many validation images.

    python tools/eval/rsqrt_collect.py [N=300]        (.venv, needs ai_edge_litert)

Writes build/rsqrt_pairs.npz: counts[k] is a 256x256 matrix (input+128, output+128) for RSQRT op k.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ai_edge_litert import interpreter as tfl
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/deit_tiny_patch16_224/deit_tiny_w8a8_pilot.tflite"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RSQRT_OPCODE = None


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    sys.path.insert(0, str(ROOT / "tools" / "tflite2c"))
    # op list from the flatbuffer parser is only available with the `tflite` package; fall back to
    # the interpreter's op details.
    it = tfl.Interpreter(str(MODEL), experimental_op_resolver_type=tfl.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
                         experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    ops = [d for d in it._get_ops_details() if d["op_name"] == "RSQRT"]
    inp = it.get_input_details()[0]
    q = inp["quantization_parameters"]
    counts = np.zeros((len(ops), 256, 256), dtype=np.int64)
    for i in range(n):
        img = Image.open(ROOT / f"data/imagenet_val_subset/val_{i:05d}.png").convert("RGB")
        arr = ((np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        x = np.clip(np.round(arr / float(q["scales"][0])) + int(q["zero_points"][0]), -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], x)
        it.invoke()
        for k, d in enumerate(ops):
            a = it.get_tensor(d["inputs"][0]).reshape(-1).astype(int) + 128
            b = it.get_tensor(d["outputs"][0]).reshape(-1).astype(int) + 128
            np.add.at(counts[k], (a, b), 1)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{n}", flush=True)
    np.savez(ROOT / "build/rsqrt_pairs.npz", counts=counts,
             in_idx=[d["inputs"][0] for d in ops], out_idx=[d["outputs"][0] for d in ops])
    print("ops:", len(ops))


if __name__ == "__main__":
    main()
