#!/usr/bin/env python3
"""Extract a DeiT W8A8 linear-operator tile into one generated C header.

The header contains everything the bare-metal B0/B1/X1 programs need:
scalar INT8 operands, packed uint64 operands (for Xqmac8), INT32 golden,
per-channel requant multipliers, and INT8 golden outputs.

Usage: extract_deit_op.py --op qkv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ai_edge_litert import interpreter as tfl

# Tensor indices come from the W8A8 pilot model (see probe_block0_tensors.py).
OPS = {
    "fc1": dict(
        needle="Block_0/timm.layers.mlp.Mlp_mlp/torch.nn.modules.linear.Linear_fc1",
        weight_idx=69, in_idx=184, out_idx=185, weight_shape=(768, 192),
        description="Block 0 MLP fc1 (192 -> 768)",
    ),
    "fc2": dict(
        needle="Block_0/timm.layers.mlp.Mlp_mlp/torch.nn.modules.linear.Linear_fc2",
        weight_idx=97, in_idx=186, out_idx=187, weight_shape=(192, 768),
        description="Block 0 MLP fc2 (768 -> 192), input is GELU output",
    ),
    "qkv": dict(
        needle="Block_0/timm.layers.attention.Attention_attn/torch.nn.modules.linear.Linear_qkv",
        weight_idx=72, in_idx=148, out_idx=149, weight_shape=(576, 192),
        description="Block 0 attention QKV projection (192 -> 576)",
    ),
}


def pack_rows_u64(matrix: np.ndarray) -> np.ndarray:
    rows, inner = matrix.shape
    words = inner // 8
    packed = np.zeros((rows, words), dtype=np.uint64)
    lanes = matrix.astype(np.int8).reshape(rows, words, 8)
    for lane in range(8):
        packed |= (lanes[:, :, lane].astype(np.uint64) & np.uint64(0xFF)) << np.uint64(8 * lane)
    return packed


def pack_columns_u64(matrix_kn: np.ndarray) -> np.ndarray:
    return pack_rows_u64(matrix_kn.T.copy())


def float_to_multiplier_shift(scale: float) -> tuple[int, int]:
    shift = 0
    scaled = float(scale)
    while scaled < (1 << 30) and shift < 31:
        scaled *= 2.0
        shift += 1
    return int(round(scaled)), shift


def requant_ref(acc: np.ndarray, multipliers: list[int], shift: int, zp: int) -> np.ndarray:
    out = np.empty(acc.shape, dtype=np.int8)
    for r in range(acc.shape[0]):
        for c in range(acc.shape[1]):
            product = int(acc[r, c]) * int(multipliers[c])
            if shift > 0:
                product = (product + (1 << (shift - 1))) >> shift
            out[r, c] = np.int8(max(-128, min(127, product + zp)))
    return out


def c_array(name: str, ctype: str, values: np.ndarray, guard: str | None = None) -> str:
    flat = values.reshape(-1)
    lines = []
    if guard:
        lines.append(f"#ifdef {guard}")
    lines.append(f"static const {ctype} {name}[] = {{")
    per_line = 8 if ctype == "uint64_t" else 16
    for i in range(0, flat.size, per_line):
        chunk = flat[i : i + per_line]
        if ctype == "uint64_t":
            body = ", ".join(f"UINT64_C({int(v)})" for v in chunk)
        else:
            body = ", ".join(str(int(v)) for v in chunk)
        lines.append(f"    {body},")
    lines.append("};")
    if guard:
        lines.append("#endif")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=sorted(OPS), required=True)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/deit_tiny_patch16_224/deit_tiny_w8a8_pilot.tflite"),
    )
    args = parser.parse_args()
    cfg = OPS[args.op]
    inner = cfg["weight_shape"][1]
    assert inner % 8 == 0

    interp = tfl.Interpreter(str(args.model))
    interp.allocate_tensors()
    details = {t["index"]: t for t in interp.get_tensor_details()}

    weight = details[cfg["weight_idx"]]
    if cfg["needle"] not in weight["name"] or tuple(weight["shape"]) != cfg["weight_shape"]:
        raise SystemExit(f"tensor {cfg['weight_idx']} does not match op {args.op}")

    weights_nk = interp.get_tensor(cfg["weight_idx"]).astype(np.int8)
    weights_kn = weights_nk[: args.columns, :].T.copy()  # [K, N]

    rng = np.random.default_rng(20260929)
    activations = rng.integers(-8, 9, size=(args.rows, inner), dtype=np.int8)
    golden_i32 = activations.astype(np.int32) @ weights_kn.astype(np.int32)

    act_q = details[cfg["in_idx"]]["quantization_parameters"]
    out_q = details[cfg["out_idx"]]["quantization_parameters"]
    w_scales = [float(x) for x in weight["quantization_parameters"]["scales"][: args.columns]]
    act_scale = float(act_q["scales"][0])
    out_scale = float(out_q["scales"][0])
    out_zp = int(out_q["zero_points"][0])

    pairs = [float_to_multiplier_shift(act_scale * ws / out_scale) for ws in w_scales]
    shift = max(s for _, s in pairs)
    multipliers = [m << (shift - s) for m, s in pairs]
    golden_i8 = requant_ref(golden_i32, multipliers, shift, out_zp)

    packed_a = pack_rows_u64(activations)
    packed_b = pack_columns_u64(weights_kn)
    macs = args.rows * inner * args.columns
    input_id = f"deit-block0-{args.op}-tile-v1"

    header = "\n".join(
        [
            "/* Auto-generated by scripts/extract_deit_op.py. Do not edit. */",
            f"/* Operator: {cfg['description']}; input id: {input_id} */",
            "#ifndef DEIT_OP_DATA_GENERATED_H",
            "#define DEIT_OP_DATA_GENERATED_H",
            "",
            "#include <stdint.h>",
            "",
            f"#define DEIT_ROWS {args.rows}",
            f"#define DEIT_INNER {inner}",
            f"#define DEIT_COLUMNS {args.columns}",
            f"#define DEIT_MACS {macs}",
            f"#define DEIT_REQUANT_SHIFT {shift}",
            f"#define DEIT_REQUANT_ZERO_POINT {out_zp}",
            "",
            c_array("deit_activations", "int8_t", activations, "DEIT_SCALAR_DATA"),
            c_array("deit_weights", "int8_t", weights_kn, "DEIT_SCALAR_DATA"),
            c_array("deit_packed_activations", "uint64_t", packed_a, "DEIT_PACKED_DATA"),
            c_array("deit_packed_weights", "uint64_t", packed_b, "DEIT_PACKED_DATA"),
            c_array("deit_golden_i32", "int32_t", golden_i32),
            c_array("deit_requant_multipliers", "int32_t", np.array(multipliers, dtype=np.int64).astype(np.int32)),
            c_array("deit_golden_i8", "int8_t", golden_i8),
            "",
            "#endif",
            "",
        ]
    )
    header_path = Path(f"src/deit_{args.op}_data.h")
    header_path.write_text(header, encoding="utf-8")

    meta = {
        "input_id": input_id,
        "operator": cfg["description"],
        "model": str(args.model).replace("\\", "/"),
        "weight_tensor_index": cfg["weight_idx"],
        "act_tensor_index": cfg["in_idx"],
        "out_tensor_index": cfg["out_idx"],
        "tile": {"rows": args.rows, "inner": inner, "columns": args.columns, "macs": macs},
        "activation_source": "deterministic rng seed=20260929 in [-8, 8]",
        "act_scale": act_scale,
        "weight_scales_tile": w_scales,
        "out_scale": out_scale,
        "out_zero_point": out_zp,
        "requant_shift": shift,
        "requant_multipliers": multipliers,
        "header": str(header_path).replace("\\", "/"),
        "golden_i32_corners": {
            "first": int(golden_i32[0, 0]),
            "last": int(golden_i32[-1, -1]),
        },
        "golden_i8_corners": {
            "first": int(golden_i8[0, 0]),
            "last": int(golden_i8[-1, -1]),
        },
        "notes": "Raw INT32 GEMM + per-channel requant; no activation zero-point correction or bias.",
    }
    meta_path = Path(f"models/deit_tiny_patch16_224/deit_{args.op}_tile_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
