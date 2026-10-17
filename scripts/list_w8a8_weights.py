#!/usr/bin/env python3
"""List INT8 2D weight-like tensors from a W8A8 TFLite model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ai_edge_litert import interpreter as tfl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    interp = tfl.Interpreter(str(args.model))
    interp.allocate_tensors()
    details = interp.get_tensor_details()

    candidates = []
    for tensor in details:
        shape = tensor.get("shape")
        dtype = tensor.get("dtype")
        if shape is None or len(shape) != 2 or dtype != np.int8:
            continue
        if int(np.prod(shape)) < 1000:
            continue
        quant = tensor.get("quantization_parameters", {})
        scales = quant.get("scales", [])
        zero_points = quant.get("zero_points", [])
        candidates.append(
            {
                "index": tensor["index"],
                "name": tensor.get("name", ""),
                "shape": list(shape),
                "scale": float(scales[0]) if len(scales) else None,
                "zero_point": int(zero_points[0]) if len(zero_points) else None,
            }
        )

    print(f"total_tensors={len(details)}")
    print(f"candidate_weight_tensors={len(candidates)}")
    for row in candidates[: args.limit]:
        print(
            f"index={row['index']} shape={row['shape']} "
            f"scale={row['scale']} zp={row['zero_point']} name={row['name']}"
        )


if __name__ == "__main__":
    main()
