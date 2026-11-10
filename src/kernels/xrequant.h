#ifndef XREQUANT_H
#define XREQUANT_H

#include <stdint.h>

#include "xrequant_encoding.h"

/* One xrequant custom instruction: rd = requant(rs1 = accumulator, rs2 = packed operand).
 * rd is write-only. Encoding for rd=a0, rs1=a1, rs2=a2: 0x56c5950b. */
static inline int8_t xrequant_hw(int32_t accumulator, uint64_t packed_rs2)
{
    register int64_t result asm("a0");
    register int64_t source_acc asm("a1") = accumulator;
    register uint64_t source_rs2 asm("a2") = packed_rs2;

    asm volatile(".word 0x56c5950b"
                 : "=r"(result)
                 : "r"(source_acc), "r"(source_rs2));
    return (int8_t)result;
}

/* Legacy single-rounding requantization (mode 0, right shift 0..63). */
static inline int8_t xrequant_i32(
    int32_t accumulator,
    int32_t multiplier,
    int shift,
    int32_t zero_point)
{
    return xrequant_hw(accumulator, xrequant_pack_rs2(multiplier, xrequant_pack_ctrl(shift, zero_point)));
}

/* TFLite-exact requantization. mode 1: half-up right shift (FC/CONV), mode 2: half-away (BMM, ...).
 * shift is TFLite's signed shift (positive = left shift of the accumulator, handled in the instruction). */
static inline int8_t xrequant_tfl(
    int32_t accumulator,
    int32_t multiplier,
    int shift,
    int32_t zero_point,
    int mode)
{
    return xrequant_hw(accumulator, xrequant_pack(multiplier, shift, zero_point, mode));
}

#endif
