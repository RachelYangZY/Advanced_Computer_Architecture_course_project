#!/usr/bin/env python3
"""Print the op graph around Block 0's MLP (fc1 -> GELU -> fc2) in the W8A8 pilot."""

from __future__ import annotations

from pathlib import Path

from ai_edge_litert import interpreter as tfl

import sys

MODEL = Path("models/deit_tiny_patch16_224/deit_tiny_w8a8_pilot.tflite")
# Name substrings to match (default: Block 0 MLP); pass e.g. Linear_qkv on argv.
NEEDLES = tuple(sys.argv[1:]) or ("Mlp_mlp", "norm2", "Linear_proj")


def main() -> None:
    interp = tfl.Interpreter(
        str(MODEL),
        experimental_op_resolver_type=tfl.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        experimental_preserve_all_tensors=True,
    )
    interp.allocate_tensors()
    details = {t["index"]: t for t in interp.get_tensor_details()}

    def describe(idx: int) -> str:
        t = details[idx]
        q = t["quantization_parameters"]
        scales = [float(x) for x in q["scales"]][:1]
        zps = [int(x) for x in q["zero_points"]][:1]
        return f"{idx}:{t['dtype'].__name__}{[int(x) for x in t['shape']]} s={scales} zp={zps}"

    for op in interp._get_ops_details():
        outs = op["outputs"]
        names = " ".join(details[i]["name"] for i in list(op["inputs"]) + list(outs) if i >= 0)
        if "Block_0/" not in names:
            continue
        if not any(k in names for k in NEEDLES):
            continue
        print(f"op#{op['index']} {op['op_name']}")
        print("   in : " + " | ".join(describe(i) for i in op["inputs"] if i >= 0))
        print("   out: " + " | ".join(describe(i) for i in outs))


if __name__ == "__main__":
    main()
