#ifndef XREQUANT_H
#define XREQUANT_H

#include <stdint.h>

#include "xrequant_encoding.h"

/* Bare-metal helper: one xrequant custom instruction. */
static inline int8_t xrequant_i32(
    int32_t accumulator,
    int32_t multiplier,
    int shift,
    int32_t zero_point)
{
    register int64_t result asm("a0") =
        (int64_t)xrequant_pack_ctrl(shift, zero_point);
    register int64_t source_acc asm("a1") = accumulator;
    register int64_t source_mult asm("a2") = multiplier;

    /* Encoding for rd=a0, rs1=a1, rs2=a2: 0x56c5950b */
    asm volatile(".word 0x56c5950b"
                 : "+r"(result)
                 : "r"(source_acc), "r"(source_mult));
    return (int8_t)result;
}

/* TFLite-exact requantization of an already left-shifted accumulator.
 * mode 1: half-up right shift (FC/CONV), mode 2: half-away (BMM, ADD, ...). right_shift in 0..31. */
static inline int8_t xrequant_tfl(
    int32_t accumulator,
    int32_t multiplier,
    int right_shift,
    int32_t zero_point,
    int mode)
{
    register int64_t result asm("a0") =
        (int64_t)xrequant_pack_ctrl_mode(right_shift, zero_point, mode);
    register int64_t source_acc asm("a1") = accumulator;
    register int64_t source_mult asm("a2") = multiplier;

    asm volatile(".word 0x56c5950b"
                 : "+r"(result)
                 : "r"(source_acc), "r"(source_mult));
    return (int8_t)result;
}

#endif
