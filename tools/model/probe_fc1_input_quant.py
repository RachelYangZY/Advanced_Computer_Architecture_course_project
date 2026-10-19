#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
from ai_edge_litert import interpreter as tfl

model = Path("models/deit_tiny_patch16_224/deit_tiny_w8a8_pilot.tflite")
interp = tfl.Interpreter(str(model))
interp.allocate_tensors()
rows = []
for tensor in interp.get_tensor_details():
    name = tensor.get("name", "")
    if "Block_0" not in name:
        continue
    if "norm2" not in name and "fc1" not in name:
        continue
    quant = tensor.get("quantization_parameters", {})
    rows.append(
        {
            "index": int(tensor["index"]),
            "shape": [int(x) for x in tensor.get("shape", [])],
            "dtype": str(tensor.get("dtype")),
            "scales": [float(x) for x in quant.get("scales", [])][:4],
            "zero_points": [int(x) for x in quant.get("zero_points", [])][:4],
            "name": name,
        }
    )
print(json.dumps(rows, indent=2))
