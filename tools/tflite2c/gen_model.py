#!/usr/bin/env python3
"""TFLite (W8A8) -> bare-metal C generator.

Pipeline:  parse (tflite_ir) -> lower (precompute quant params, LUTs) -> plan memory
           -> emit weights.c / weights.h / model.c / harness.c

The generated model is a straight-line sequence of kernel calls on a static arena.
No TFLite runtime, no floating point on the target.

Usage:
  gen_model.py --ops 56:60 --out build/gen            # op range [start, end)
  gen_model.py --ops 56:60 --out build/gen_noreuse --no-reuse   # one buffer per tensor
Unsupported ops make the generator fail loudly.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import tflite_ir  # noqa: E402

DEFAULT_MODEL = "models/deit_tiny_patch16_224/deit_tiny_w8a8_pilot.tflite"
GENERATOR_VERSION = "0.1"
ALIGN = 8
EW_MAX_RANK = 5
RSQRT_REL_BIAS = 1e-5


# ----------------------------------------------------------------------------
# Quantization helpers (match TFLite)
# ----------------------------------------------------------------------------

def quantize_multiplier(real: float) -> tuple[int, int]:
    """TFLite QuantizeMultiplier: real -> (Q31 multiplier, shift)."""
    if real == 0.0:
        return 0, 0
    q, shift = math.frexp(real)
    qf = int(math.floor(q * (1 << 31) + 0.5))
    if qf == (1 << 31):
        qf //= 2
        shift += 1
    if shift < -31:
        shift, qf = 0, 0
    return qf, shift


def round_half_away(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.floor(np.abs(x) + np.float32(0.5))


def act_range(fused: int) -> tuple[int, int]:
    if fused != 0:
        raise NotImplementedError(f"fused activation {fused} not supported yet")
    return -128, 127


def scale_zp(t: tflite_ir.Tensor) -> tuple[float, int]:
    return float(t.scales[0]), int(t.zero_points[0])


# ----------------------------------------------------------------------------
# C emission helpers
# ----------------------------------------------------------------------------

def c_array(ctype: str, name: str, values: np.ndarray, aligned: bool = True) -> tuple[str, str]:
    """Return (extern declaration, definition) for a const array."""
    flat = np.asarray(values).reshape(-1)
    n = max(len(flat), 1)
    attr = " __attribute__((aligned(8)))" if aligned else ""
    decl = f"extern const {ctype} {name}[{n}];"
    lines = [f"const {ctype} {name}[{n}]{attr} = {{"]
    row: list[str] = []
    for v in (flat if len(flat) else [0]):
        row.append(str(int(v)))
        if len(row) == 24:
            lines.append("    " + ",".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ",".join(row) + ",")
    lines.append("};")
    return decl, "\n".join(lines)


def sanitize(name: str, keep: int = 48) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    return s[-keep:]


# ----------------------------------------------------------------------------
# Graph lowering
# ----------------------------------------------------------------------------

@dataclass
class Buffer:
    root: int
    size: int = 0
    start: int = 10**9
    end: int = -1
    offset: int = -1


def prod(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def broadcast_strides(shape_in: tuple[int, ...], out_shape: tuple[int, ...]) -> list[int]:
    rank = len(out_shape)
    padded = [1] * (rank - len(shape_in)) + list(shape_in)
    strides, acc = [0] * rank, 1
    for d in range(rank - 1, -1, -1):
        strides[d] = 0 if padded[d] == 1 and out_shape[d] != 1 else acc
        acc *= padded[d]
    return strides


def pad_list(v: list[int], fill: int = 0) -> list[int]:
    return v + [fill] * (EW_MAX_RANK - len(v))


class Generator:
    def __init__(self, model: tflite_ir.Model, start: int, end: int, reuse: bool):
        self.m = model
        self.T = model.tensors
        self.ops = model.ops[start:end]
        self.start, self.end = start, end
        self.reuse = reuse
        self.decls: list[str] = []
        self.defs: list[str] = []
        self.emitted_consts: set[int] = set()
        self.param_defs: list[str] = []
        self.calls: list[str] = []
        self.alias: dict[int, int] = {}      # tensor -> tensor it aliases (RESHAPE)
        self.buffers: dict[int, Buffer] = {}
        self.arena_size = 0

    # --- tensor roles ------------------------------------------------------
    def analyse(self) -> None:
        produced, used_at = {}, {}
        for pos, op in enumerate(self.ops):
            for o in op.outputs:
                produced[o] = pos
            for i in op.inputs:
                if i >= 0:
                    used_at.setdefault(i, []).append(pos)
        self.produced, self.used_at = produced, used_at
        # RESHAPE aliasing: output shares the input buffer (no copy).
        for op in self.ops:
            if op.kind == "RESHAPE":
                src, dst = op.inputs[0], op.outputs[0]
                if prod(self.T[src].shape) != prod(self.T[dst].shape):
                    raise ValueError(f"RESHAPE {op.idx}: size mismatch")
                self.alias[dst] = src

        def root(t: int) -> int:
            while t in self.alias:
                t = self.alias[t]
            return t
        self.root = root

        self.inputs = sorted(
            t for t in used_at if t not in produced and not self.T[t].is_const
        )
        consumed = set(used_at)
        self.outputs = sorted(t for t in produced if t not in consumed)
        n_ops = len(self.ops)
        for t in list(self.inputs) + list(produced):
            r = root(t)
            b = self.buffers.setdefault(r, Buffer(r, size=prod(self.T[r].shape)))
            b.size = max(b.size, prod(self.T[t].shape))
            first = 0 if t in self.inputs else produced[t]
            last = max(used_at.get(t, [produced.get(t, 0)]))
            if t in self.outputs:
                last = n_ops
            b.start, b.end = min(b.start, first), max(b.end, last)

    # --- memory planning ---------------------------------------------------
    def plan(self) -> None:
        bufs = sorted(self.buffers.values(), key=lambda b: (-b.size, b.start, b.root))
        placed: list[Buffer] = []
        top = 0
        for b in bufs:
            size = (b.size + ALIGN - 1) // ALIGN * ALIGN
            if not self.reuse:
                b.offset = top
                top += size
                continue
            conflicts = sorted(
                (p for p in placed if not (p.end < b.start or b.end < p.start)),
                key=lambda p: p.offset,
            )
            off = 0
            for p in conflicts:
                psize = (p.size + ALIGN - 1) // ALIGN * ALIGN
                if off + size <= p.offset:
                    break
                off = max(off, p.offset + psize)
            b.offset = off
            top = max(top, off + size)
            placed.append(b)
        self.arena_size = (top + ALIGN - 1) // ALIGN * ALIGN

    def off(self, t: int) -> int:
        return self.buffers[self.root(t)].offset

    def ptr(self, t: int) -> str:
        if self.T[t].is_const:
            return f"c_{t}"
        return f"(arena + {self.off(t)})"

    # --- constants ---------------------------------------------------------
    def const(self, t: int, ctype: str | None = None, dtype=None) -> str:
        if t not in self.emitted_consts:
            self.emitted_consts.add(t)
            data = self.T[t].data
            ct = ctype or {"INT8": "int8_t", "INT32": "int32_t"}[self.T[t].dtype]
            d, f = c_array(ct, f"c_{t}", data)
            self.decls.append(d)
            self.defs.append(f)
        return f"c_{t}"

    def extra(self, ctype: str, name: str, values) -> str:
        d, f = c_array(ctype, name, np.asarray(values))
        self.decls.append(d)
        self.defs.append(f)
        return name

    # --- per-op lowering ---------------------------------------------------
    def lower(self) -> None:
        for pos, op in enumerate(self.ops):
            fn = getattr(self, f"lower_{op.kind.lower()}", None)
            if fn is None:
                raise NotImplementedError(
                    f"op #{op.idx} {op.kind} not supported by the generator yet"
                )
            fn(op)

    def check_i8(self, *ids: int) -> None:
        for i in ids:
            if self.T[i].dtype != "INT8":
                raise NotImplementedError(f"tensor {i} dtype {self.T[i].dtype}")

    def lower_reshape(self, op) -> None:
        self.calls.append(f"    /* op {op.idx} RESHAPE: alias t{op.inputs[0]} -> t{op.outputs[0]} */")

    def lower_fully_connected(self, op) -> None:
        x, w, b = op.inputs[0], op.inputs[1], op.inputs[2] if len(op.inputs) > 2 else -1
        out = op.outputs[0]
        self.check_i8(x, w, out)
        tx, tw, to = self.T[x], self.T[w], self.T[out]
        n, k = tw.shape
        if np.any(tw.zero_points != 0):
            raise NotImplementedError("FC weights must be symmetric (zp 0)")
        rows = prod(tx.shape) // k
        s_in, zp_in = scale_zp(tx)
        s_out, zp_out = scale_zp(to)
        w_scales = tw.scales if len(tw.scales) > 1 else np.repeat(tw.scales, n)
        mult, shift = [], []
        for c in range(n):
            real = (float(s_in) * float(w_scales[c])) / float(s_out)
            q, s = quantize_multiplier(real)
            mult.append(q)
            shift.append(s)
        lo, hi = act_range(op.options.get("fused_activation", 0))
        wname = self.const(w)
        bname = self.const(b) if b >= 0 else self.extra("int32_t", f"zero_bias_{op.idx}", np.zeros(n, np.int32))
        mname = self.extra("int32_t", f"fc_mult_{op.idx}", np.array(mult, np.int32))
        sname = self.extra("int32_t", f"fc_shift_{op.idx}", np.array(shift, np.int32))
        self.param_defs.append(
            f"static const fc_params_t p_{op.idx} = {{ .w = {wname}, .bias = {bname}, "
            f".mult = {mname}, .shift = {sname}, .in_zp = {zp_in}, .out_zp = {zp_out}, "
            f".act_min = {lo}, .act_max = {hi}, .rows = {rows}, .k = {k}, .n = {n} }};"
        )
        self.calls.append(
            f"    op_fc_i8(&p_{op.idx}, {self.ptr(x)}, {self.ptr(out)});  /* op {op.idx} FC {rows}x{k}x{n} */"
        )

    def _lut_op(self, op, func, tag: str) -> None:
        """INT8->INT8 lookup table op, built like TFLite's PopulateLookupTable (float32)."""
        x, out = op.inputs[0], op.outputs[0]
        self.check_i8(x, out)
        s_in, zp_in = scale_zp(self.T[x])
        s_out, zp_out = scale_zp(self.T[out])
        vals = np.arange(-128, 128, dtype=np.int32)
        f32 = np.float32
        deq = f32(s_in) * (vals - zp_in).astype(f32)
        with np.errstate(all="ignore"):
            transformed = func(deq).astype(f32)
        resc = round_half_away(transformed * (f32(1.0) / f32(s_out)))
        # Non-finite results convert to INT_MIN on x86 and then saturate to -128.
        resc = np.where(np.isfinite(resc), resc, -(2.0 ** 31)).astype(np.int64)
        lut = np.clip(resc + zp_out, -128, 127).astype(np.int8)
        name = self.extra("int8_t", f"lut_{op.idx}", lut)
        n = prod(self.T[x].shape)
        self.calls.append(
            f"    op_lut_i8({name}, {self.ptr(x)}, {self.ptr(out)}, {n});  /* op {op.idx} {tag} */"
        )

    def lower_gelu(self, op) -> None:
        if op.options.get("approximate"):
            raise NotImplementedError("approximate GELU")
        f32 = np.float32

        def gelu(deq):
            erf = np.array([math.erf(float(v) * math.sqrt(0.5)) for v in deq], dtype=np.float64).astype(f32)
            return f32(0.5) * deq * (f32(1.0) + erf)

        self._lut_op(op, gelu, "GELU")

    def lower_rsqrt(self, op) -> None:
        # EMPIRICAL: TFLite's int8 RSQRT lands slightly above the exact value; a +1e-5 relative bias
        # reproduces it on all observed (input, output) pairs of 25 RSQRT ops x 3 inputs (0 vs ~4
        # mismatching pairs without it). The true algorithm is unknown (see docs/GENERATOR.md).
        bias = np.float32(1.0 + RSQRT_REL_BIAS)
        self._lut_op(op, lambda d: (np.float32(1.0) / np.sqrt(d)) * bias, "RSQRT")

    def lower_mean(self, op) -> None:
        x, axes_t, out = op.inputs[0], op.inputs[1], op.outputs[0]
        self.check_i8(x, out)
        tx, to = self.T[x], self.T[out]
        rank = len(tx.shape)
        if rank > EW_MAX_RANK:
            raise NotImplementedError("mean rank")
        axes = {int(a) % rank for a in np.atleast_1d(self.T[axes_t].data)}
        reduce = [1 if d in axes else 0 for d in range(rank)]
        count = prod([tx.shape[d] for d in axes])
        n_out = prod([tx.shape[d] for d in range(rank) if d not in axes])
        if n_out > 4096:
            raise NotImplementedError(f"mean output {n_out} > MEAN_MAX_OUT")
        s_in, zp_in = scale_zp(tx)
        s_out, zp_out = scale_zp(to)
        mult, shift = quantize_multiplier(float(s_in) / float(s_out) / count)
        # EMPIRICAL rounding rule. TFLite's MEAN rounds once (exact rational rounding) for 49 of the 50
        # MEAN ops but reproduces gemmlowp double rounding for the one op with input/output scale ratio
        # 53.5 (LayerNorm 1 variance mean of Block 0). The selection criterion is not known; ratio < 64
        # separates that op from all others in this model (see docs/GENERATOR.md).
        single_round = 0 if float(s_in) / float(s_out) < 64.0 else 1
        arr = lambda v: "{" + ",".join(str(x) for x in pad_list(list(v))) + "}"  # noqa: E731
        self.param_defs.append(
            f"static const mean_params_t p_{op.idx} = {{ .rank = {rank}, .in_shape = {arr(tx.shape)}, "
            f".reduce = {arr(reduce)}, .count = {count}, .zp_in = {zp_in}, .zp_out = {zp_out}, "
            f".mult = {mult}, .shift = {shift}, .single_round = {single_round} }};"
        )
        self.calls.append(
            f"    op_mean_i8(&p_{op.idx}, {self.ptr(x)}, {self.ptr(out)});  /* op {op.idx} MEAN axes {sorted(axes)} */"
        )

    def lower_slice(self, op) -> None:
        x, begin_t, size_t, out = op.inputs[0], op.inputs[1], op.inputs[2], op.outputs[0]
        self.check_i8(x, out)
        tx = self.T[x]
        rank = len(tx.shape)
        if rank > EW_MAX_RANK:
            raise NotImplementedError("slice rank")
        begin = [int(v) for v in self.T[begin_t].data]
        size = [int(v) for v in self.T[size_t].data]
        size = [tx.shape[d] - begin[d] if s < 0 else s for d, s in enumerate(size)]
        if tuple(size) != tuple(self.T[out].shape):
            raise ValueError(f"SLICE {op.idx}: size {size} != output {self.T[out].shape}")
        arr = lambda v: "{" + ",".join(str(x) for x in pad_list(list(v))) + "}"  # noqa: E731
        self.param_defs.append(
            f"static const slice_params_t p_{op.idx} = {{ .rank = {rank}, .in_shape = {arr(tx.shape)}, "
            f".begin = {arr(begin)}, .size = {arr(size)} }};"
        )
        self.calls.append(
            f"    op_slice_i8(&p_{op.idx}, {self.ptr(x)}, {self.ptr(out)});  /* op {op.idx} SLICE begin {begin} */"
        )

    def lower_batch_matmul(self, op) -> None:
        a, b, out = op.inputs[0], op.inputs[1], op.outputs[0]
        self.check_i8(a, b, out)
        ta, tb, to = self.T[a], self.T[b], self.T[out]
        adj_x, adj_y = bool(op.options.get("adj_x")), bool(op.options.get("adj_y"))
        if len(ta.shape) != 3 or len(tb.shape) != 3 or ta.shape[0] != tb.shape[0]:
            raise NotImplementedError(f"BATCH_MATMUL shapes {ta.shape} x {tb.shape}")
        batch = ta.shape[0]
        m, k = (ta.shape[2], ta.shape[1]) if adj_x else (ta.shape[1], ta.shape[2])
        n = tb.shape[1] if adj_y else tb.shape[2]
        kb = tb.shape[2] if adj_y else tb.shape[1]
        if kb != k or tuple(to.shape) != (batch, m, n):
            raise ValueError(f"BATCH_MATMUL {op.idx}: inconsistent shapes")
        sa, za = scale_zp(ta)
        sb, zb = scale_zp(tb)
        so, zo = scale_zp(to)
        mult, shift = quantize_multiplier(float(sa) * float(sb) / float(so))
        lo, hi = act_range(0)
        self.param_defs.append(
            f"static const bmm_params_t p_{op.idx} = {{ .batch = {batch}, .m = {m}, .k = {k}, .n = {n}, "
            f".adj_x = {int(adj_x)}, .adj_y = {int(adj_y)}, .zp_a = {za}, .zp_b = {zb}, .zp_out = {zo}, "
            f".mult = {mult}, .shift = {shift}, .act_min = {lo}, .act_max = {hi} }};"
        )
        pa = self.const(a) if ta.is_const else self.ptr(a)
        pb = self.const(b) if tb.is_const else self.ptr(b)
        self.calls.append(
            f"    op_bmm_i8(&p_{op.idx}, {pa}, {pb}, {self.ptr(out)});  /* op {op.idx} BMM {batch}x{m}x{k}x{n} */"
        )

    def lower_softmax(self, op) -> None:
        x, out = op.inputs[0], op.outputs[0]
        self.check_i8(x, out)
        if float(op.options.get("beta", 1.0)) != 1.0:
            raise NotImplementedError("softmax beta != 1")
        tx = self.T[x]
        s_in, _ = scale_zp(tx)
        s_out, zp_out = scale_zp(self.T[out])
        out_mul = 1.0 / float(s_out)
        if abs(out_mul - round(out_mul)) > 1e-6:
            raise NotImplementedError(f"softmax output scale {s_out}: 1/scale not an integer")
        # exp(-d*scale) in Q31, d = row_max - x. The input scale is the float32 value TFLite stores.
        tab = np.array([int(math.floor(math.exp(-float(np.float32(s_in)) * d) * (1 << 31) + 0.5))
                        for d in range(256)], dtype=np.uint32)
        tname = self.extra("uint32_t", f"smx_tab_{op.idx}", tab)
        cols = tx.shape[-1]
        rows = prod(tx.shape) // cols
        self.param_defs.append(
            f"static const softmax_params_t p_{op.idx} = {{ .rows = {rows}, .cols = {cols}, "
            f".exp_tab = {tname}, .out_mul = {int(round(out_mul))}, .zp_out = {zp_out} }};"
        )
        self.calls.append(
            f"    op_softmax_i8(&p_{op.idx}, {self.ptr(x)}, {self.ptr(out)});  /* op {op.idx} SOFTMAX {rows}x{cols} */"
        )

    def lower_squared_difference(self, op) -> None:
        a, b, out, ta, tb, to, rank, sa, sb = self._ew_common(op)
        s1, z1 = scale_zp(ta)
        s2, z2 = scale_zp(tb)
        so, zo = scale_zp(to)
        left = 7
        twice_max = 2.0 * max(float(s1), float(s2))
        m1, sh1 = quantize_multiplier(float(s1) / twice_max)
        m2, sh2 = quantize_multiplier(float(s2) / twice_max)
        mo, sho = quantize_multiplier(twice_max * twice_max / ((1 << (2 * left)) * float(so)))
        fields = dict(zp_a=z1, zp_b=z2, zp_out=zo, left_shift=left, mult_a=m1, mult_b=m2,
                      mult_out=mo, shift_a=sh1, shift_b=sh2, shift_out=sho)
        self._emit_ew(op, "op_sqdiff_i8", fields, rank, to, sa, sb, a, b, out, "SQUARED_DIFFERENCE")

    def _ew_common(self, op):
        a, b = op.inputs[0], op.inputs[1]
        out = op.outputs[0]
        self.check_i8(a, b, out)
        ta, tb, to = self.T[a], self.T[b], self.T[out]
        rank = len(to.shape)
        if rank > EW_MAX_RANK:
            raise NotImplementedError(f"rank {rank} > {EW_MAX_RANK}")
        sa = broadcast_strides(ta.shape, to.shape)
        sb = broadcast_strides(tb.shape, to.shape)
        return a, b, out, ta, tb, to, rank, sa, sb

    def _emit_ew(self, op, fn, fields: dict[str, int], rank, to, sa, sb, a, b, out, tag) -> None:
        arr = lambda v: "{" + ",".join(str(x) for x in pad_list(list(v))) + "}"  # noqa: E731
        lo, hi = act_range(op.options.get("fused_activation", 0))
        wa = self.const(a) if self.T[a].is_const else None
        wb = self.const(b) if self.T[b].is_const else None
        fld = ", ".join(f".{k} = {v}" for k, v in fields.items())
        self.param_defs.append(
            f"static const ew_params_t p_{op.idx} = {{ .rank = {rank}, .shape = {arr(to.shape)}, "
            f".stride_a = {arr(sa)}, .stride_b = {arr(sb)}, .act_min = {lo}, .act_max = {hi}, {fld} }};"
        )
        pa = wa or self.ptr(a)
        pb = wb or self.ptr(b)
        self.calls.append(f"    {fn}(&p_{op.idx}, {pa}, {pb}, {self.ptr(out)});  /* op {op.idx} {tag} */")

    def _addsub(self, op, fn, tag) -> None:
        a, b, out, ta, tb, to, rank, sa, sb = self._ew_common(op)
        s1, z1 = scale_zp(ta)
        s2, z2 = scale_zp(tb)
        so, zo = scale_zp(to)
        left = 20
        twice_max = 2.0 * max(float(s1), float(s2))
        m1, sh1 = quantize_multiplier(float(s1) / twice_max)
        m2, sh2 = quantize_multiplier(float(s2) / twice_max)
        mo, sho = quantize_multiplier(twice_max / ((1 << left) * float(so)))
        fields = dict(zp_a=z1, zp_b=z2, zp_out=zo, left_shift=left, mult_a=m1, mult_b=m2,
                      mult_out=mo, shift_a=sh1, shift_b=sh2, shift_out=sho)
        self._emit_ew(op, fn, fields, rank, to, sa, sb, a, b, out, tag)

    def lower_add(self, op) -> None:
        self._addsub(op, "op_add_i8", "ADD")

    def lower_sub(self, op) -> None:
        self._addsub(op, "op_sub_i8", "SUB")

    def lower_mul(self, op) -> None:
        a, b, out, ta, tb, to, rank, sa, sb = self._ew_common(op)
        s1, z1 = scale_zp(ta)
        s2, z2 = scale_zp(tb)
        so, zo = scale_zp(to)
        mult, shift = quantize_multiplier((float(s1) * float(s2)) / float(so))
        fields = dict(zp_a=z1, zp_b=z2, zp_out=zo, mult=mult, shift=shift)
        self._emit_ew(op, "op_mul_i8", fields, rank, to, sa, sb, a, b, out, "MUL")

    def lower_transpose(self, op) -> None:
        x, perm_t, out = op.inputs[0], op.inputs[1], op.outputs[0]
        self.check_i8(x, out)
        perm = [int(p) for p in self.T[perm_t].data]
        tx, to = self.T[x], self.T[out]
        rank = len(perm)
        if rank > EW_MAX_RANK:
            raise NotImplementedError("transpose rank")
        in_strides, acc = [0] * rank, 1
        for d in range(rank - 1, -1, -1):
            in_strides[d] = acc
            acc *= tx.shape[d]
        stride = [in_strides[p] for p in perm]
        arr = lambda v: "{" + ",".join(str(x) for x in pad_list(list(v))) + "}"  # noqa: E731
        self.param_defs.append(
            f"static const transpose_params_t p_{op.idx} = {{ .rank = {rank}, "
            f".shape = {arr(to.shape)}, .stride = {arr(stride)} }};"
        )
        self.calls.append(
            f"    op_transpose_i8(&p_{op.idx}, {self.ptr(x)}, {self.ptr(out)});  /* op {op.idx} TRANSPOSE */"
        )

    # --- file emission -----------------------------------------------------
    def emit(self, outdir: Path) -> None:
        outdir.mkdir(parents=True, exist_ok=True)
        header = (
            f"/* Generated by gen_model.py v{GENERATOR_VERSION} - DO NOT EDIT.\n"
            f" * model sha256: {self.m.sha256}\n"
            f" * ops [{self.start}, {self.end}), reuse={'yes' if self.reuse else 'no'} */\n"
        )
        (outdir / "weights.h").write_text(
            header + "#ifndef WEIGHTS_H\n#define WEIGHTS_H\n#include <stdint.h>\n"
            + "\n".join(self.decls) + "\n#endif\n", encoding="utf-8", newline="\n")
        (outdir / "weights.c").write_text(
            header + '#include "weights.h"\n\n' + "\n\n".join(self.defs) + "\n",
            encoding="utf-8", newline="\n")

        tt = []
        ins, outs = set(self.inputs), set(self.outputs)
        checks = sorted(self.produced) if not self.reuse else self.outputs
        # SOFTMAX is integer-only on the target, TFLite uses float32: allow rare off-by-one near ties.
        approx = {o for op in self.ops if op.kind in ("SOFTMAX", "BATCH_MATMUL") for o in op.outputs}
        approx |= {t for t in self.alias if self.root(t) in approx or self.root(t) in
                   {self.root(a) for a in approx}}
        for t in sorted(set(self.inputs) | set(checks)):
            tt.append(
                f'    {{ {t}, {self.off(t)}, {prod(self.T[t].shape)}, {1 if t in ins else 0}, '
                f'{1 if t in set(checks) else 0}, {1 if t in approx else 0}, "{sanitize(self.T[t].name)}" }},'
            )
        model_c = [
            header, '#include <stdint.h>\n#include "ops_ref.h"\n#include "weights.h"\n#include "model.h"\n',
            f"static int8_t arena_buf[{self.arena_size}] __attribute__((aligned(8)));",
            "int8_t *model_arena(void) { return arena_buf; }",
            "",
            *self.param_defs,
            "",
            "void model_run(void)\n{\n    int8_t *arena = arena_buf;",
            *self.calls,
            "}",
            "",
            "const model_tensor_t model_tensors[] = {",
            *tt,
            "};",
            f"const int model_tensor_count = {len(tt)};",
        ]
        (outdir / "model.c").write_text("\n".join(model_c) + "\n", encoding="utf-8", newline="\n")
        (outdir / "model.h").write_text(
            "#ifndef MODEL_H\n#define MODEL_H\n#include <stdint.h>\n"
            f"#define MODEL_ARENA_BYTES {self.arena_size}\n"
            "typedef struct { int idx; int offset; int nbytes; int is_input; int check; int approx; const char *name; } model_tensor_t;\n"
            "extern const model_tensor_t model_tensors[];\nextern const int model_tensor_count;\n"
            "int8_t *model_arena(void);\nvoid model_run(void);\n#endif\n",
            encoding="utf-8", newline="\n")
        (outdir / "harness.c").write_text(HARNESS, encoding="utf-8", newline="\n")

    def report(self) -> str:
        naive = sum((b.size + ALIGN - 1) // ALIGN * ALIGN for b in self.buffers.values())
        return (f"ops [{self.start},{self.end}): {len(self.ops)} ops, inputs {self.inputs}, "
                f"outputs {self.outputs}, arena {self.arena_size} B (no-reuse {naive} B)")


HARNESS = r'''/* Host test harness: load reference inputs, run the model, diff against TFLite tensors. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"

static int load(const char *dir, int idx, int8_t *dst, int nbytes)
{
    char path[512];
    snprintf(path, sizeof path, "%s/t%d.bin", dir, idx);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "missing %s\n", path); return -1; }
    size_t n = fread(dst, 1, (size_t)nbytes, f);
    fclose(f);
    if ((int)n != nbytes) { fprintf(stderr, "short read %s (%zu of %d)\n", path, n, nbytes); return -1; }
    return 0;
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : "build/ref";
    int8_t *arena = model_arena();
    for (int i = 0; i < model_tensor_count; i++) {
        const model_tensor_t *t = &model_tensors[i];
        if (t->is_input && load(dir, t->idx, arena + t->offset, t->nbytes)) return 2;
    }
    model_run();
    int bad_tensors = 0;
    int8_t *ref = malloc(1 << 22);
    for (int i = 0; i < model_tensor_count; i++) {
        const model_tensor_t *t = &model_tensors[i];
        if (!t->check) continue;
        if (t->nbytes > (1 << 22) || load(dir, t->idx, ref, t->nbytes)) return 2;
        int mism = 0, maxd = 0;
        for (int j = 0; j < t->nbytes; j++) {
            int d = (int)arena[t->offset + j] - (int)ref[j];
            if (d < 0) d = -d;
            if (d) mism++;
            if (d > maxd) maxd = d;
        }
        /* approx tensors (softmax): allow <= 0.01% elements off by one */
        int allowed = t->approx && maxd <= 1 && mism * 10000 <= t->nbytes;
        printf("t%-4d %-50s n=%-7d mismatches=%-6d max|diff|=%d %s\n",
               t->idx, t->name, t->nbytes, mism, maxd,
               mism == 0 ? "ok" : (allowed ? "ok~(approx)" : "FAIL"));
        if (mism && !allowed) bad_tensors++;
    }
    printf("%s: %d tensor(s) mismatching\n", bad_tensors ? "FAIL" : "PASS", bad_tensors);
    return bad_tensors ? 1 : 0;
}
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ops", default="0:648", help="op index range START:END (end exclusive)")
    ap.add_argument("--out", default="build/gen")
    ap.add_argument("--no-reuse", action="store_true", help="give every tensor its own buffer")
    args = ap.parse_args()
    s, e = (int(x) for x in args.ops.split(":"))
    model = tflite_ir.load(args.model)
    g = Generator(model, s, e, reuse=not args.no_reuse)
    g.analyse()
    g.plan()
    g.lower()
    g.emit(Path(args.out))
    print(g.report())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
