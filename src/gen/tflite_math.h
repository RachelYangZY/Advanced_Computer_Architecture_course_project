#ifndef TFLITE_MATH_H
#define TFLITE_MATH_H

/*
 * Fixed-point helpers that reproduce TFLite/gemmlowp integer arithmetic bit for bit.
 * Used by the generated model's reference kernels so host results can be compared
 * exactly against the TFLite interpreter.
 */

#include <stdint.h>

/* gemmlowp SaturatingRoundingDoublingHighMul */
static inline int32_t tfl_srdhm(int32_t a, int32_t b)
{
    if (a == INT32_MIN && b == INT32_MIN) {
        return INT32_MAX;
    }
    int64_t ab = (int64_t)a * (int64_t)b;
    int32_t nudge = ab >= 0 ? (1 << 30) : (1 - (1 << 30));
    return (int32_t)((ab + nudge) / (1LL << 31));
}

/* gemmlowp RoundingDivideByPOT (round half away from zero) */
static inline int32_t tfl_rdivpot(int32_t x, int exponent)
{
    int32_t mask = (int32_t)((1LL << exponent) - 1);
    int32_t remainder = x & mask;
    int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> exponent) + (remainder > threshold ? 1 : 0);
}

/* TFLite MultiplyByQuantizedMultiplier (shift > 0: left shift, shift <= 0: right shift) */
static inline int32_t tfl_mul_qm(int32_t x, int32_t multiplier, int shift)
{
    int left = shift > 0 ? shift : 0;
    int right = shift > 0 ? 0 : -shift;
    return tfl_rdivpot(tfl_srdhm((int32_t)((uint32_t)x << left), multiplier), right);
}

/*
 * Variant used by TFLite's optimized (ruy) per-channel FULLY_CONNECTED path: the final
 * right shift rounds half up (add half, arithmetic shift) instead of half away from zero.
 * Verified bit-exact against the interpreter for Block 0 fc1 (see EXPERIMENTS.md).
 */
static inline int32_t tfl_mul_qm_up(int32_t x, int32_t multiplier, int shift)
{
    int left = shift > 0 ? shift : 0;
    int right = shift > 0 ? 0 : -shift;
    int32_t v = tfl_srdhm((int32_t)((uint32_t)x << left), multiplier);
    if (right == 0) {
        return v;
    }
    return (int32_t)(((int64_t)v + (1LL << (right - 1))) >> right);
}

static inline int8_t tfl_clamp_i8(int32_t v, int32_t lo, int32_t hi)
{
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return (int8_t)v;
}

#endif
