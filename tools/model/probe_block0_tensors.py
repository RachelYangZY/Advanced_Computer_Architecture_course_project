#!/usr/bin/env python3
"""List Block 0 tensors (shape/dtype/quant) matching a name substring."""

from __future__ import annotations

import argparse
from pathlib import Path

from ai_edge_litert import interpreter as tfl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("needle", help="substring, e.g. Linear_qkv or norm1")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/deit_tiny_patch16_224/deit_tiny_w8a8_pilot.tflite"),
    )
    args = parser.parse_args()

    interp = tfl.Interpreter(str(args.model))
    interp.allocate_tensors()
    for tensor in interp.get_tensor_details():
        name = tensor.get("name", "")
        if "Block_0/" not in name or args.needle not in name:
            continue
        quant = tensor.get("quantization_parameters", {})
        scales = [float(x) for x in quant.get("scales", [])][:3]
        zps = [int(x) for x in quant.get("zero_points", [])][:3]
        short = name.split("/")[-1][-60:]
        print(
            f"index={tensor['index']} shape={[int(x) for x in tensor['shape']]} "
            f"dtype={tensor['dtype'].__name__} scales={scales} zps={zps} ...{short}"
        )


if __name__ == "__main__":
    main()
