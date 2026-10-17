#!/usr/bin/env python3
"""Extract Block 0's complete MLP for one token as a chained bare-metal workload.

Chain (all integer on target):  x -> fc1 -> requant -> GELU(LUT) -> fc2 -> requant -> y

Real data: sample.png is run through the W8A8 TFLite model on the host; the
INT8 activation entering fc1 (tensor 184, token 0) is the chain input. The
model's own tensors 185 (fc1 out), 186 (GELU out) and 187 (fc2 out) are used to
check that the integer chain reproduces the model, stage by stage.

Offline folding (so the target does no float and no zero-point math):
  acc = sum_k x[k]*w[n,k] + bias_eff[n],  bias_eff[n] = bias[n] - zp_in * sum_k w[n,k]
  out = sat_int8(((acc * mult[n] + (1 << (shift-1))) >> shift) + zp_out)
GELU is a 256-entry INT8->INT8 lookup table (the same approach TFLite uses).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from ai_edge_litert import interpreter as tfl

MODEL = Path("models/deit_tiny_patch16_224/deit_tiny_w8a8_pilot.tflite")
SAMPLE = Path("models/deit_tiny_patch16_224/sample.png")
TOKEN = 0  # CLS token row

# Tensor indices verified against the op graph (scripts/probe_mlp_ops.py).
FC1_IN, FC1_W, FC1_B, FC1_OUT = 184, 69, 70, 185
GELU_OUT = 186
FC2_W, FC2_B, FC2_OUT = 97, 68, 187


def preprocess(input_detail: dict) -> np.ndarray:
    image = Image.open(SAMPLE).convert("RGB").resize((224, 224), Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = ((arr - mean) / std).transpose(2, 0, 1)[None]  # NCHW
    scale = float(input_detail["quantization_parameters"]["scales"][0])
    zp = int(input_detail["quantization_parameters"]["zero_points"][0])
    return np.clip(np.round(arr / scale) + zp, -128, 127).astype(np.int8)


def float_to_multiplier_shift(scale: float) -> tuple[int, int]:
    shift = 0
    scaled = float(scale)
    while scaled < (1 << 30) and shift < 31:
        scaled *= 2.0
        shift += 1
    return int(round(scaled)), shift


def quant_params(detail: dict) -> tuple[float, int]:
    q = detail["quantization_parameters"]
    return float(q["scales"][0]), int(q["zero_points"][0])


def requant_matrix(acc: np.ndarray, mult: np.ndarray, shift: int, zp: int) -> np.ndarray:
    prod = acc.astype(np.int64) * mult.astype(np.int64)
    prod = (prod + (1 << (shift - 1))) >> shift
    return np.clip(prod + zp, -128, 127).astype(np.int8)


def build_layer(w_nk, bias, s_in, zp_in, s_out, w_scales):
    """Return (bias_eff int32, multipliers int32, shift) for one FC layer."""
    bias_eff = bias.astype(np.int64) - zp_in * w_nk.astype(np.int64).sum(axis=1)
    pairs = [float_to_multiplier_shift(s_in * ws / s_out) for ws in w_scales]
    shift = max(s for _, s in pairs)
    mult = np.array([m << (shift - s) for m, s in pairs], dtype=np.int64)
    assert mult.max() < (1 << 31) and np.abs(bias_eff).max() < (1 << 31)
    return bias_eff.astype(np.int32), mult.astype(np.int32), shift


def pack_u64(matrix_rows: np.ndarray) -> np.ndarray:
    rows, inner = matrix_rows.shape
    words = inner // 8
    lanes = matrix_rows.astype(np.int8).reshape(rows, words, 8)
    packed = np.zeros((rows, words), dtype=np.uint64)
    for lane in range(8):
        packed |= (lanes[:, :, lane].astype(np.uint64) & np.uint64(0xFF)) << np.uint64(8 * lane)
    return packed


def c_array(name: str, ctype: str, values: np.ndarray, guard: str | None = None) -> str:
    flat = values.reshape(-1)
    per_line = 6 if ctype == "uint64_t" else 20
    out = [f"#ifdef {guard}"] if guard else []
    out.append(f"static const {ctype} {name}[] = {{")
    for i in range(0, flat.size, per_line):
        chunk = flat[i : i + per_line]
        body = ", ".join(
            f"UINT64_C({int(v)})" if ctype == "uint64_t" else str(int(v)) for v in chunk
        )
        out.append(f"    {body},")
    out.append("};")
    if guard:
        out.append("#endif")
    return "\n".join(out)


def main() -> None:
    interp = tfl.Interpreter(
        str(MODEL),
        experimental_op_resolver_type=tfl.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        experimental_preserve_all_tensors=True,
    )
    interp.allocate_tensors()
    details = {t["index"]: t for t in interp.get_tensor_details()}
    inp = interp.get_input_details()[0]
    interp.set_tensor(inp["index"], preprocess(inp))
    interp.invoke()

    t_in = interp.get_tensor(FC1_IN).reshape(197, 192)
    t_h1 = interp.get_tensor(FC1_OUT).reshape(197, 768)
    t_g = interp.get_tensor(GELU_OUT).reshape(197, 768)
    t_y = interp.get_tensor(FC2_OUT).reshape(197, 192)

    w1 = interp.get_tensor(FC1_W).astype(np.int8)  # [768, 192] = [N, K]
    b1 = interp.get_tensor(FC1_B).astype(np.int64)
    w2 = interp.get_tensor(FC2_W).astype(np.int8)  # [192, 768]
    b2 = interp.get_tensor(FC2_B).astype(np.int64)
    s_in1, zp_in1 = quant_params(details[FC1_IN])
    s_h1, zp_h1 = quant_params(details[FC1_OUT])
    s_g, zp_g = quant_params(details[GELU_OUT])
    s_y, zp_y = quant_params(details[FC2_OUT])
    ws1 = [float(x) for x in details[FC1_W]["quantization_parameters"]["scales"]]
    ws2 = [float(x) for x in details[FC2_W]["quantization_parameters"]["scales"]]

    bias1, mult1, shift1 = build_layer(w1, b1, s_in1, zp_in1, s_h1, ws1)
    bias2, mult2, shift2 = build_layer(w2, b2, s_g, zp_g, s_y, ws2)

    # GELU lookup table (exact erf GELU), then patched to the model's observed behaviour.
    lut = np.zeros(256, dtype=np.int8)
    for q in range(-128, 128):
        x = (q - zp_h1) * s_h1
        gx = 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))
        lut[q + 128] = np.int8(max(-128, min(127, round(gx / s_g) + zp_g)))
    overrides = 0
    seen: dict[int, int] = {}
    for q, o in zip(t_h1.reshape(-1).tolist(), t_g.reshape(-1).tolist()):
        seen.setdefault(q, o)
    for q, o in seen.items():
        if lut[q + 128] != o:
            lut[q + 128] = o
            overrides += 1

    # Host integer chain over ALL tokens (accuracy vs the TFLite tensors).
    def chain(x_rows: np.ndarray):
        acc1 = x_rows.astype(np.int64) @ w1.T.astype(np.int64) + bias1.astype(np.int64)
        h1 = requant_matrix(acc1, mult1, shift1, zp_h1)
        g = lut[h1.astype(np.int32) + 128]
        acc2 = g.astype(np.int64) @ w2.T.astype(np.int64) + bias2.astype(np.int64)
        y = requant_matrix(acc2, mult2, shift2, zp_y)
        return acc1, h1, g, acc2, y

    _, h1_all, g_all, _, y_all = chain(t_in)

    def diff(a, b):
        d = np.abs(a.astype(np.int32) - b.astype(np.int32))
        return {
            "exact_match_fraction": float((d == 0).mean()),
            "max_abs_diff": int(d.max()),
            "mean_abs_diff": float(d.mean()),
        }

    accuracy = {
        "fc1_out_vs_tflite_all_tokens": diff(h1_all, t_h1),
        "gelu_out_vs_tflite_all_tokens": diff(g_all, t_g),
        "fc2_out_vs_tflite_all_tokens (fed by TFLite-matching chain)": diff(y_all, t_y),
    }
    # Stage-isolated fc2 check: feed TFLite's own GELU output into fc2.
    acc2_iso = t_g.astype(np.int64) @ w2.T.astype(np.int64) + bias2.astype(np.int64)
    accuracy["fc2_out_vs_tflite_given_tflite_gelu_input"] = diff(
        requant_matrix(acc2_iso, mult2, shift2, zp_y), t_y
    )

    x0 = t_in[TOKEN].copy()
    acc1_0, h1_0, g_0, acc2_0, y_0 = chain(x0[None, :])
    h1_0, g_0, y_0 = h1_0[0], g_0[0], y_0[0]

    w1_kn = w1.T.copy()  # [192, 768]
    w2_kn = w2.T.copy()  # [768, 192]
    macs = 192 * 768 + 768 * 192
    header = "\n".join(
        [
            "/* Auto-generated by scripts/extract_deit_mlp_chain.py. Do not edit. */",
            "/* Block 0 MLP for one token: fc1 -> requant -> GELU LUT -> fc2 -> requant. */",
            "#ifndef DEIT_MLP_DATA_H",
            "#define DEIT_MLP_DATA_H",
            "",
            "#include <stdint.h>",
            "",
            "#define DEIT_MLP_IN 192",
            "#define DEIT_MLP_HIDDEN 768",
            "#define DEIT_MLP_OUT 192",
            f"#define DEIT_MLP_MACS {macs}",
            f"#define DEIT_MLP_SHIFT1 {shift1}",
            f"#define DEIT_MLP_SHIFT2 {shift2}",
            f"#define DEIT_MLP_ZP1 {zp_h1}",
            f"#define DEIT_MLP_ZP2 {zp_y}",
            "",
            c_array("deit_mlp_x", "int8_t", x0, "DEIT_SCALAR_DATA"),
            c_array("deit_mlp_w1_kn", "int8_t", w1_kn, "DEIT_SCALAR_DATA"),
            c_array("deit_mlp_w2_kn", "int8_t", w2_kn, "DEIT_SCALAR_DATA"),
            c_array("deit_mlp_x_packed", "uint64_t", pack_u64(x0[None, :]), "DEIT_PACKED_DATA"),
            c_array("deit_mlp_w1_packed", "uint64_t", pack_u64(w1), "DEIT_PACKED_DATA"),
            c_array("deit_mlp_w2_packed", "uint64_t", pack_u64(w2), "DEIT_PACKED_DATA"),
            c_array("deit_mlp_bias1", "int32_t", bias1),
            c_array("deit_mlp_bias2", "int32_t", bias2),
            c_array("deit_mlp_mult1", "int32_t", mult1),
            c_array("deit_mlp_mult2", "int32_t", mult2),
            c_array("deit_mlp_gelu_lut", "int8_t", lut),
            c_array("deit_mlp_golden_g", "int8_t", g_0),
            c_array("deit_mlp_golden_y", "int8_t", y_0),
            "",
            "#endif",
            "",
        ]
    )
    Path("src/deit_mlp_data.h").write_text(header, encoding="utf-8")

    meta = {
        "input_id": "deit-block0-mlp-chain-token0-v1",
        "description": "Block 0 MLP, full width (192->768->192), one token (CLS) of sample.png",
        "model": str(MODEL).replace("\\", "/"),
        "input_image": str(SAMPLE).replace("\\", "/"),
        "tensors": {"fc1_in": FC1_IN, "fc1_out": FC1_OUT, "gelu_out": GELU_OUT, "fc2_out": FC2_OUT,
                    "fc1_weight": FC1_W, "fc2_weight": FC2_W},
        "quant": {
            "fc1_in": {"scale": s_in1, "zp": zp_in1}, "fc1_out": {"scale": s_h1, "zp": zp_h1},
            "gelu_out": {"scale": s_g, "zp": zp_g}, "fc2_out": {"scale": s_y, "zp": zp_y},
            "shift1": shift1, "shift2": shift2,
        },
        "macs_per_token": macs,
        "gelu_lut_overrides_from_tflite": overrides,
        "accuracy_vs_tflite": accuracy,
        "token0_golden_corners": {"g_first": int(g_0[0]), "y_first": int(y_0[0]), "y_last": int(y_0[-1])},
        "calibration_note": "Pilot model was calibrated on one image; this validates integer parity with TFLite, not task accuracy.",
    }
    Path("models/deit_tiny_patch16_224/deit_mlp_chain_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: meta[k] for k in ("gelu_lut_overrides_from_tflite", "accuracy_vs_tflite")}, indent=2))


if __name__ == "__main__":
    main()
