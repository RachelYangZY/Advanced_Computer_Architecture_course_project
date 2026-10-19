#!/usr/bin/env python3
"""Probe Block0 MLP fc1 related quantization parameters from the W8A8 pilot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ai_edge_litert import interpreter as tfl


def main() -> None:
    model = Path("models/deit_tiny_patch16_224/deit_tiny_w8a8_pilot.tflite")
    interp = tfl.Interpreter(str(model))
    interp.allocate_tensors()
    details = interp.get_tensor_details()

    rows = []
    for tensor in details:
        name = tensor.get("name", "")
        if "Block_0" not in name or "fc1" not in name:
            continue
        quant = tensor.get("quantization_parameters", {})
        scales = [float(x) for x in quant.get("scales", [])]
        zps = [int(x) for x in quant.get("zero_points", [])]
        rows.append(
            {
                "index": int(tensor["index"]),
                "name": name,
                "shape": [int(x) for x in tensor.get("shape", [])],
                "dtype": str(tensor.get("dtype")),
                "scales": scales[:8],
                "zero_points": zps[:8],
            }
        )

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
