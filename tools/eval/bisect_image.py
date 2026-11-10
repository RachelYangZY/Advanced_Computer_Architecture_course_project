#!/usr/bin/env python3
"""Dump every TFLite int8 tensor for one validation image (to find the first op that diverges).

    python tools/eval/bisect_image.py <image index> [outdir=build/ref_img]

Then (WSL): generate the no-reuse model and run the host harness against that directory:
    python tools/tflite2c/gen_model.py --ops 0:648 --out build/gen_full_nr --no-reuse
    gcc -O1 -Isrc/gen -Ibuild/gen_full_nr -o build/gen_full_nr/host_test build/gen_full_nr/{harness,model,weights}.c src/gen/ops_ref.c
    build/gen_full_nr/host_test build/ref_img
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


def main() -> None:
    idx = int(sys.argv[1])
    out = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "build/ref_img")
    out.mkdir(parents=True, exist_ok=True)
    it = tfl.Interpreter(
        str(MODEL),
        experimental_op_resolver_type=tfl.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        experimental_preserve_all_tensors=True,
    )
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    img = Image.open(ROOT / f"data/imagenet_val_subset/val_{idx:05d}.png").convert("RGB")
    arr = ((np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
    q = inp["quantization_parameters"]
    x = np.clip(np.round(arr / float(q["scales"][0])) + int(q["zero_points"][0]), -128, 127).astype(np.int8)
    it.set_tensor(inp["index"], x)
    it.invoke()
    n = 0
    for t in it.get_tensor_details():
        try:
            a = it.get_tensor(t["index"])
        except Exception:
            continue
        if a.dtype == np.int8:
            (out / f"t{t['index']}.bin").write_bytes(np.ascontiguousarray(a).tobytes())
            n += 1
    print(f"dumped {n} int8 tensors to {out}")


if __name__ == "__main__":
    main()
