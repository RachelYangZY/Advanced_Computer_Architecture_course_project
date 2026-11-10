#include <stdint.h>
#include <stdio.h>

#include "xqmac8.h"

int main(void)
{
    const uint64_t packed_a = UINT64_C(0x807f0102fe030405);
    const uint64_t packed_b = UINT64_C(0x0203fd04f5060708);
    const int64_t expected_dot_product =
        (int64_t)5 * 8 + (int64_t)4 * 7 + (int64_t)3 * 6 +
        (int64_t)-2 * -11 + (int64_t)2 * 4 + (int64_t)1 * -3 +
        (int64_t)127 * 3 + (int64_t)-128 * 2;
    /* rd is write-only: the result is the dot product alone (no accumulator input). */
    const int64_t result = xqmac8_reference(packed_a, packed_b);

    if (result != expected_dot_product) {
        fprintf(stderr, "Xqmac8 mismatch: got %lld, expected %lld\n",
                (long long)result,
                (long long)expected_dot_product);
        return 1;
    }

    printf("Xqmac8 reference test passed: %lld\n", (long long)result);
    return 0;
}