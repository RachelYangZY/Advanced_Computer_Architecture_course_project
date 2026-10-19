#!/usr/bin/env python3
"""Parse a .tflite flatbuffer into a small, self-contained IR (no TFLite runtime).

Used by summarize_model.py and the C generator (gen_model.py). Deterministic:
ops are in graph order, tensors indexed as in the flatbuffer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tflite
from tflite import BuiltinOperator as BO

_OP_NAMES = {v: k for k, v in vars(BO).items() if not k.startswith("_") and isinstance(v, int)}
_TYPE_NAMES = {v: k for k, v in vars(tflite.TensorType).items() if not k.startswith("_")}
_NP = {"INT8": np.int8, "UINT8": np.uint8, "INT32": np.int32, "INT16": np.int16,
       "FLOAT32": np.float32, "INT64": np.int64, "BOOL": np.bool_}


@dataclass
class Tensor:
    idx: int
    name: str
    dtype: str
    shape: tuple[int, ...]
    scales: np.ndarray            # empty if not quantized
    zero_points: np.ndarray
    qdim: int                     # quantized dimension (per-channel)
    data: np.ndarray | None       # constant data, else None

    @property
    def is_const(self) -> bool:
        return self.data is not None

    @property
    def nbytes(self) -> int:
        n = int(np.prod(self.shape)) if self.shape else 1
        return n * np.dtype(_NP.get(self.dtype, np.uint8)).itemsize


@dataclass
class Op:
    idx: int
    kind: str                     # e.g. FULLY_CONNECTED
    inputs: list[int]             # -1 = absent optional input
    outputs: list[int]
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class Model:
    path: str
    sha256: str
    tensors: list[Tensor]
    ops: list[Op]
    inputs: list[int]
    outputs: list[int]


def _options(op, kind: str, buf_opts) -> dict[str, Any]:
    """Extract the few option fields the generator needs, by op kind."""
    out: dict[str, Any] = {}
    if op.BuiltinOptionsType() == 0:
        return out
    table = op.BuiltinOptions()
    if table is None:
        return out
    mod_by_kind = {
        "FULLY_CONNECTED": tflite.FullyConnectedOptions,
        "BATCH_MATMUL": tflite.BatchMatMulOptions,
        "SOFTMAX": tflite.SoftmaxOptions,
        "ADD": tflite.AddOptions,
        "SUB": tflite.SubOptions,
        "MUL": tflite.MulOptions,
        "MEAN": tflite.ReducerOptions,
        "CONV_2D": tflite.Conv2DOptions,
        "RESHAPE": tflite.ReshapeOptions,
        "CONCATENATION": tflite.ConcatenationOptions,
        "GELU": tflite.GeluOptions,
    }
    cls = mod_by_kind.get(kind)
    if cls is None:
        return out
    o = cls()
    o.Init(table.Bytes, table.Pos)
    if kind == "FULLY_CONNECTED":
        out["fused_activation"] = o.FusedActivationFunction()
        out["keep_num_dims"] = bool(o.KeepNumDims())
    elif kind == "BATCH_MATMUL":
        out["adj_x"] = bool(o.AdjX())
        out["adj_y"] = bool(o.AdjY())
    elif kind == "SOFTMAX":
        out["beta"] = o.Beta()
    elif kind in ("ADD", "SUB", "MUL"):
        out["fused_activation"] = o.FusedActivationFunction()
    elif kind == "MEAN":
        out["keep_dims"] = bool(o.KeepDims())
    elif kind == "CONV_2D":
        out.update(stride_h=o.StrideH(), stride_w=o.StrideW(), padding=o.Padding(),
                   fused_activation=o.FusedActivationFunction(),
                   dilation_h=o.DilationHFactor(), dilation_w=o.DilationWFactor())
    elif kind == "GELU":
        out["approximate"] = bool(o.Approximate())
    elif kind == "CONCATENATION":
        out["axis"] = o.Axis()
    return out


def load(path: str | Path) -> Model:
    path = Path(path)
    raw = path.read_bytes()
    m = tflite.Model.GetRootAsModel(raw, 0)
    g = m.Subgraphs(0)

    tensors: list[Tensor] = []
    for i in range(g.TensorsLength()):
        t = g.Tensors(i)
        dtype = _TYPE_NAMES[t.Type()]
        shape = tuple(int(x) for x in t.ShapeAsNumpy()) if t.ShapeLength() else ()
        q = t.Quantization()
        if q is not None and q.ScaleLength():
            scales = np.array(q.ScaleAsNumpy(), dtype=np.float32)
            zps = np.array(q.ZeroPointAsNumpy(), dtype=np.int64)
            qdim = int(q.QuantizedDimension())
        else:
            scales, zps, qdim = np.zeros(0, np.float32), np.zeros(0, np.int64), 0
        buf = m.Buffers(t.Buffer())
        data = None
        if buf.DataLength() > 0:
            data = np.frombuffer(buf.DataAsNumpy().tobytes(), dtype=_NP[dtype]).reshape(shape).copy()
        elif buf.Size() > 0:
            # Large constants live outside the Buffer table: bytes [Offset, Offset+Size) of the file.
            blob = raw[buf.Offset():buf.Offset() + buf.Size()]
            data = np.frombuffer(blob, dtype=_NP[dtype]).reshape(shape).copy()
        tensors.append(Tensor(i, t.Name().decode(), dtype, shape, scales, zps, qdim, data))

    codes = [m.OperatorCodes(i) for i in range(m.OperatorCodesLength())]
    ops: list[Op] = []
    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        c = codes[op.OpcodeIndex()]
        code = max(c.BuiltinCode(), c.DeprecatedBuiltinCode())
        kind = _OP_NAMES.get(code, f"UNKNOWN_{code}")
        n_in = sum(1 for x in op.InputsAsNumpy() if x >= 0)
        if kind == "ADD" and n_in == 1:
            # This export stores the input-quantize op under opcode slot (0,0); the TFLite
            # runtime reports it as QUANTIZE (one input, one output, requantize).
            kind = "QUANTIZE"
        ops.append(Op(
            idx=i, kind=kind,
            inputs=[int(x) for x in op.InputsAsNumpy()],
            outputs=[int(x) for x in op.OutputsAsNumpy()],
            options=_options(op, kind, None),
        ))

    return Model(
        path=str(path), sha256=hashlib.sha256(raw).hexdigest(), tensors=tensors, ops=ops,
        inputs=[int(x) for x in g.InputsAsNumpy()], outputs=[int(x) for x in g.OutputsAsNumpy()],
    )


if __name__ == "__main__":
    import sys
    mdl = load(sys.argv[1])
    print(mdl.sha256, len(mdl.tensors), "tensors", len(mdl.ops), "ops")
