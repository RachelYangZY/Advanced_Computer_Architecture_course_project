#include "requantize.h"

int8_t requantize_i32_to_i8(
    int32_t accumulator,
    int32_t multiplier,
    int shift,
    int32_t zero_point)
{
    int64_t product = (int64_t)accumulator * (int64_t)multiplier;

    if (shift > 0) {
        const int64_t rounding = (int64_t)1 << (shift - 1);
        product = (product + rounding) >> shift;
    }

    product += zero_point;
    if (product > 127) {
        return 127;
    }
    if (product < -128) {
        return -128;
    }
    return (int8_t)product;
}

void requantize_i32_to_i8_vector(
    const int32_t *accumulators,
    int8_t *output,
    int count,
    int32_t multiplier,
    int shift,
    int32_t zero_point)
{
    for (int index = 0; index < count; ++index) {
        output[index] = requantize_i32_to_i8(
            accumulators[index], multiplier, shift, zero_point);
    }
}
