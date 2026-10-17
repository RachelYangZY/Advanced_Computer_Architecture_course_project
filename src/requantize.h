#ifndef REQUANTIZE_H
#define REQUANTIZE_H

#include <stdint.h>

/* Integer requantization matching the common gemmlowp/TFLite form:
 *
 *   rounded = (acc * multiplier + rounding) >> shift
 *   out     = sat_int8(rounded + zero_point)
 *
 * multiplier is a signed 32-bit fixed-point factor.
 * shift is a non-negative right shift.
 */
int8_t requantize_i32_to_i8(
    int32_t accumulator,
    int32_t multiplier,
    int shift,
    int32_t zero_point);

void requantize_i32_to_i8_vector(
    const int32_t *accumulators,
    int8_t *output,
    int count,
    int32_t multiplier,
    int shift,
    int32_t zero_point);

#endif
