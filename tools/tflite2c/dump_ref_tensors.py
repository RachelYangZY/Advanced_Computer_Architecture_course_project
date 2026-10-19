#!/usr/bin/env python3
"""Run sample.png through the TFLite model and dump every readable tensor as raw bytes.

Output: <outdir>/t<idx>.bin for each tensor (host reference for the generated C).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ai_edge_litert import interpreter as tfl
from PIL import Image

MODEL = Path("models/deit_tiny_patch16_224/deit_tiny_w8a8_pilot.tflite")
SAMPLE = Path("models/deit_tiny_patch16_224/sample.png")


def preprocess(input_detail: dict) -> np.ndarray:
    image = Image.open(SAMPLE).convert("RGB").resize((224, 224), Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = ((arr - mean) / std).transpose(2, 0, 1)[None]
    q = input_detail["quantization_parameters"]
    scale, zp = float(q["scales"][0]), int(q["zero_points"][0])
    return np.clip(np.round(arr / scale) + zp, -128, 127).astype(np.int8)


def main() -> None:
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "build/ref")
    outdir.mkdir(parents=True, exist_ok=True)
    interp = tfl.Interpreter(
        str(MODEL),
        experimental_op_resolver_type=tfl.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        experimental_preserve_all_tensors=True,
    )
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    interp.set_tensor(inp["index"], preprocess(inp))
    interp.invoke()

    n = 0
    for t in interp.get_tensor_details():
        try:
            arr = interp.get_tensor(t["index"])
        except Exception:
            continue
        if arr.dtype != np.int8:
            continue
        (outdir / f"t{t['index']}.bin").write_bytes(np.ascontiguousarray(arr).tobytes())
        n += 1
    print(f"dumped {n} int8 tensors to {outdir}")


if __name__ == "__main__":
    main()
