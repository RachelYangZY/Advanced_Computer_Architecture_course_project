#include "xqmac8.h"

int64_t xqmac8_reference(uint64_t packed_a, uint64_t packed_b)
{
    int64_t sum = 0;
    for (int lane = 0; lane < 8; ++lane) {
        int shift = lane * 8;
        int8_t a = (int8_t)(packed_a >> shift);
        int8_t b = (int8_t)(packed_b >> shift);
        sum += (int64_t)a * (int64_t)b;
    }
    return sum;
}
