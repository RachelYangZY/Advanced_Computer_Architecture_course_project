#ifndef XREQUANT_REF_H
#define XREQUANT_REF_H

/*
 * Software oracle for the xrequant instruction (pure C, host and target).
 *
 *   ctrl = rd (before execution)
 *     ctrl[7:0]   zero point (signed)
 *     ctrl[15:8]  shift s (unsigned)
 *     ctrl[17:16] mode
 *   acc = sext32(rs1), mult = sext32(rs2)
 *
 *   mode 1  TFLite, round-half-up right shift      (FULLY_CONNECTED, CONV_2D; s in 0..31)
 *     v = srdhm(acc, mult) = (acc*mult + nudge) / 2^31, truncated toward zero,
 *         nudge = 2^30 if acc*mult >= 0 else 1 - 2^30; INT32_MIN*INT32_MIN gives INT32_MAX
 *     t = s ? (v + 2^(s-1)) >> s : v                  (arithmetic shift)
 *   mode 2  TFLite/gemmlowp, round-half-away-from-zero right shift (ADD/MUL/BATCH_MATMUL...; s in 0..31)
 *     t = RoundingDivideByPOT(v, s)
 *   mode 0  legacy single rounding: t = s ? (acc*mult + 2^(s-1)) >> s : acc*mult   (s in 0..63)
 *           kept so the earlier tile experiments stay reproducible; not TFLite-exact.
 *   mode 3  reserved (illegal instruction)
 *
 *   rd = sext8(clamp(t + zp, -128, 127))
 *
 * A TFLite multiplier with a positive (left) shift is applied by the caller: acc <<= left, s = 0.
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

/* Returns the signed int8 result; the caller guarantees mode in 0..2 and s in range. */
static inline int8_t xrequant_reference(int32_t acc, int32_t mult, uint32_t ctrl)
{
    int32_t zp = (int8_t)(ctrl & 0xff);
    int s = (int)((ctrl >> 8) & 0xff);
    int mode = (int)((ctrl >> 16) & 0x3);
    int64_t t;
    if (mode == 0) {
        int64_t p = (int64_t)acc * (int64_t)mult;
        t = s > 0 ? (p + ((int64_t)1 << (s - 1))) >> s : p;
    } else {
        int64_t v = xrequant_ref_srdhm(acc, mult);
        if (s == 0) {
            t = v;
        } else if (mode == 1) {
            t = (v + ((int64_t)1 << (s - 1))) >> s;
        } else {
            int64_t mask = ((int64_t)1 << s) - 1;
            int64_t rem = v & mask;
            int64_t thr = (mask >> 1) + (v < 0 ? 1 : 0);
            t = (v >> s) + (rem > thr ? 1 : 0);
        }
    }
    t += zp;
    if (t > 127) t = 127;
    if (t < -128) t = -128;
    return (int8_t)t;
}

#endif
