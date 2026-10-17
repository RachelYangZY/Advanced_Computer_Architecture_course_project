#include "xqmac8.h"

int64_t xqmac8_reference(int64_t accumulator, uint64_t packed_a, uint64_t packed_b)
{
    for (int lane = 0; lane < 8; ++lane) {
        int shift = lane * 8;
        int8_t a = (int8_t)(packed_a >> shift);
        int8_t b = (int8_t)(packed_b >> shift);
        accumulator += (int64_t)a * (int64_t)b;
    }
    return accumulator;
}