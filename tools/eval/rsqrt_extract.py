#!/usr/bin/env python3
"""Extract TFLite's exact int8 RSQRT lookup tables by running each RSQRT op in isolation.

    python tools/eval/rsqrt_extract.py        (needs ai_edge_litert and flatbuffers)

For each RSQRT op: cut a one-op model out of the flatbuffer, feed all 256 int8 inputs through the
LiteRT interpreter and store the 256 outputs. Output: build/rsqrt_luts.npz (op index -> int8[256],
indexed by input+128) plus a comparison against the float32 formula.
"""

from __future__ import annotations

import sys
from pathlib import Path

import json

import flatbuffers
import numpy as np
from ai_edge_litert import interpreter as tfl
from ai_edge_litert import schema_py_generated as tflite

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/deit_tiny_patch16_224/deit_tiny_w8a8_pilot.tflite"
f32 = np.float32


def one_op_model(op_index: int) -> bytes:
    buf = MODEL.read_bytes()
    mt = tflite.ModelT.InitFromObj(tflite.Model.GetRootAsModel(buf, 0))
    sg = mt.subgraphs[0]
    op = sg.operators[op_index]
    in_idx, out_idx = int(op.inputs[0]), int(op.outputs[0])
    sg.operators = [op]
    sg.inputs = [in_idx]
    sg.outputs = [out_idx]
    for t in sg.tensors:       # constants live in external buffers of the original file; drop them
        t.buffer = 0
    mt.buffers = mt.buffers[:1]
    b = flatbuffers.Builder(4096)
    b.Finish(mt.Pack(b), b"TFL3")
    return bytes(b.Output())


def main() -> None:
    buf = MODEL.read_bytes()
    mt = tflite.ModelT.InitFromObj(tflite.Model.GetRootAsModel(buf, 0))
    codes = {i: c.builtinCode for i, c in enumerate(mt.operatorCodes)}
    ops = [i for i, o in enumerate(mt.subgraphs[0].operators)
           if codes[o.opcodeIndex] == tflite.BuiltinOperator.RSQRT]
    print("RSQRT ops:", len(ops))
    luts = {}
    tables_json = {}
    for k in ops:
        it = tfl.Interpreter(model_content=one_op_model(k),
                             experimental_op_resolver_type=tfl.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        it.allocate_tensors()
        inp, outp = it.get_input_details()[0], it.get_output_details()[0]
        n = int(np.prod(inp["shape"]))
        table = np.zeros(256, dtype=np.int8)
        vals = np.arange(-128, 128, dtype=np.int8)
        for start in range(0, 256, n):
            chunk = vals[start:start + n]
            x = np.zeros(n, dtype=np.int8)
            x[:len(chunk)] = chunk
            it.set_tensor(inp["index"], x.reshape(inp["shape"]))
            it.invoke()
            y = it.get_tensor(outp["index"]).reshape(-1)
            table[start:start + len(chunk)] = y[:len(chunk)]
        luts[str(k)] = table
        tables_json[str(k)] = None   # filled below once the quantization parameters are known
        si = float(inp["quantization_parameters"]["scales"][0])
        zi = int(inp["quantization_parameters"]["zero_points"][0])
        so = float(outp["quantization_parameters"]["scales"][0])
        zo = int(outp["quantization_parameters"]["zero_points"][0])
        tables_json[str(k)] = {"in_scale": si, "in_zp": zi, "out_scale": so, "out_zp": zo,
                               "lut": [int(v) for v in table]}
        with np.errstate(all="ignore"):
            deq = f32(si) * (vals.astype(np.int32) - zi).astype(f32)
            tr = (f32(1.0) / np.sqrt(deq)).astype(f32)
            resc = np.sign(tr * (f32(1.0) / f32(so))) * np.floor(np.abs(tr * (f32(1.0) / f32(so))) + f32(0.5))
        resc = np.where(np.isfinite(resc), resc, -(2.0 ** 31)).astype(np.int64)
        mine = np.clip(resc + zo, -128, 127).astype(np.int8)
        diff = np.nonzero(mine != table)[0]
        print(f"op {k}: float32 formula differs at {len(diff)} of 256 entries",
              [(int(vals[i]), int(mine[i]), int(table[i])) for i in diff[:6]])
    np.savez(ROOT / "build/rsqrt_luts.npz", **luts)
    out = ROOT / "tools/tflite2c/data/deit_tiny_rsqrt_luts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tables_json))
    print("wrote", out)


if __name__ == "__main__":
    main()
