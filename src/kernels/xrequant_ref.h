#ifndef XREQUANT_REF_H
#define XREQUANT_REF_H

/*
 * Software oracle for the xrequant instruction (pure C, host and target).
 *
 *   xrequant rd, rs1, rs2        rd is written only (never read)
 *
 *   acc  = sext32(rs1)
 *   rs2[31:0]   multiplier (signed)
 *   rs2[39:32]  zero point (signed)
 *   rs2[47:40]  shift field
 *   rs2[49:48]  mode
 *
 *   mode 1  TFLite, round-half-up right shift      (FULLY_CONNECTED, CONV_2D)
 *   mode 2  TFLite/gemmlowp, round-half-away right shift (BATCH_MATMUL, ADD/MUL...)
 *     The shift field is TFLite's signed shift (int8): sh > 0 shifts the accumulator left first
 *     (32-bit wrap), sh <= 0 shifts the high-multiply result right by -sh. |sh| must be <= 31.
 *       x = (int32)((uint32)acc << max(sh, 0))
 *       v = srdhm(x, mult) = (x*mult + nudge) / 2^31, truncated toward zero,
 *           nudge = 2^30 if x*mult >= 0 else 1 - 2^30; INT32_MIN*INT32_MIN gives INT32_MAX
 *       r = max(-sh, 0)
 *       mode 1: t = r ? (v + 2^(r-1)) >> r : v          (arithmetic shift)
 *       mode 2: t = RoundingDivideByPOT(v, r)
 *   mode 0  legacy single rounding, shift field is an unsigned right shift s in 0..63:
 *             t = s ? (acc*mult + 2^(s-1)) >> s : acc*mult
 *           kept so the earlier tile experiments stay reproducible; not TFLite-exact.
 *   mode 3  reserved (illegal instruction)
 *
 *   rd = sext8(clamp(t + zp, -128, 127))
 */

#include <stdint.h>

static inline int64_t xrequant_ref_srdhm(int32_t acc, int32_t mult)
{
    if (acc == INT32_MIN && mult == INT32_MIN) {
        return INT32_MAX;
    }
    int64_t ab = (int64_t)acc * (int64_t)mult;
    int64_t nudge = ab >= 0 ? ((int64_t)1 << 30) : (1 - ((int64_t)1 << 30));
    int64_t n = ab + nudge;
    return n >= 0 ? (n >> 31) : -((-n) >> 31);   /* truncate toward zero */
}

/* Returns 1 if the operand combination is legal (otherwise the instruction traps). */
static inline int xrequant_legal(uint64_t rs2)
{
    int mode = (int)((rs2 >> 48) & 0x3);
    int sh = (int)(int8_t)(rs2 >> 40);
    return mode != 3 && (mode == 0 || (sh >= -31 && sh <= 31));
}

/* Returns the signed int8 result; the caller guarantees xrequant_legal(rs2). */
static inline int8_t xrequant_reference(int32_t acc, uint64_t rs2)
{
    int32_t mult = (int32_t)(uint32_t)rs2;
    int32_t zp = (int8_t)(rs2 >> 32);
    int mode = (int)((rs2 >> 48) & 0x3);
    int64_t t;
    if (mode == 0) {
        int s = (int)((rs2 >> 40) & 0xff);
        int64_t p = (int64_t)acc * (int64_t)mult;
        t = s > 0 ? (p + ((int64_t)1 << (s - 1))) >> s : p;
    } else {
        int sh = (int)(int8_t)(rs2 >> 40);
        int left = sh > 0 ? sh : 0, r = sh > 0 ? 0 : -sh;
        int32_t x = (int32_t)((uint32_t)acc << left);
        int64_t v = xrequant_ref_srdhm(x, mult);
        if (r == 0) {
            t = v;
        } else if (mode == 1) {
            t = (v + ((int64_t)1 << (r - 1))) >> r;
        } else {
            int64_t mask = ((int64_t)1 << r) - 1;
            int64_t rem = v & mask;
            int64_t thr = (mask >> 1) + (v < 0 ? 1 : 0);
            t = (v >> r) + (rem > thr ? 1 : 0);
        }
    }
    t += zp;
    if (t > 127) t = 127;
    if (t < -128) t = -128;
    return (int8_t)t;
}

#endif
