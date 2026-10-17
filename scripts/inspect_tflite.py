#!/usr/bin/env python3
"""Print structural and quantization metadata from a TFLite FlatBuffer."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import tflite


def enum_name(enum_type: object, value: int) -> str:
    for name, enum_value in vars(enum_type).items():
        if not name.startswith("_") and enum_value == value:
            return name
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()

    data = args.model.read_bytes()
    model = tflite.Model.GetRootAsModel(data, 0)
    graph = model.Subgraphs(0)
    tensors = [graph.Tensors(i) for i in range(graph.TensorsLength())]
    buffers = [model.Buffers(i) for i in range(model.BuffersLength())]

    print(f"file_bytes={len(data)}")
    print(f"schema_version={model.Version()}")
    print(f"subgraphs={model.SubgraphsLength()}")
    print(f"tensor_count={len(tensors)}")
    print("tensor_type_counts:")
    type_counts = Counter(tensor.Type() for tensor in tensors)
    for tensor_type, count in sorted(type_counts.items()):
        print(f"  {enum_name(tflite.TensorType, tensor_type)}: {count}")

    def describe_tensor(index: int) -> None:
        tensor = tensors[index]
        shape = [tensor.Shape(i) for i in range(tensor.ShapeLength())]
        quant = tensor.Quantization()
        scales = [] if quant is None else [quant.Scale(i) for i in range(quant.ScaleLength())]
        zero_points = [] if quant is None else [quant.ZeroPoint(i) for i in range(quant.ZeroPointLength())]
        print(
            f"  index={index} name={tensor.Name().decode(errors='replace')} "
            f"type={enum_name(tflite.TensorType, tensor.Type())} shape={shape} "
            f"buffer={tensor.Buffer()} scale_count={len(scales)} "
            f"zero_point_count={len(zero_points)} scales={scales[:8]} "
            f"zero_points={zero_points[:8]}"
        )

    print("inputs:")
    for i in range(graph.InputsLength()):
        describe_tensor(graph.Inputs(i))
    print("outputs:")
    for i in range(graph.OutputsLength()):
        describe_tensor(graph.Outputs(i))

    print(f"operator_count={graph.OperatorsLength()}")
    operator_counts: Counter[str] = Counter()
    for i in range(graph.OperatorsLength()):
        op = graph.Operators(i)
        opcode = model.OperatorCodes(op.OpcodeIndex())
        builtin_code = max(opcode.BuiltinCode(), opcode.DeprecatedBuiltinCode())
        operator_counts[enum_name(tflite.BuiltinOperator, builtin_code)] += 1
    print("operator_counts:")
    for name, count in sorted(operator_counts.items()):
        print(f"  {name}: {count}")

    print("constant_weight_storage_by_type:")
    storage_counts: Counter[str] = Counter()
    storage_bytes: Counter[str] = Counter()
    for tensor in tensors:
        if tensor.Buffer() == 0:
            continue
        size = buffers[tensor.Buffer()].DataLength()
        if size:
            name = enum_name(tflite.TensorType, tensor.Type())
            storage_counts[name] += 1
            storage_bytes[name] += size
    for name in sorted(storage_bytes):
        print(f"  {name}: tensors={storage_counts[name]} bytes={storage_bytes[name]}")


if __name__ == "__main__":
    main()
