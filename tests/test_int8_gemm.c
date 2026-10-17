#include <stdint.h>
#include <stdio.h>

#include "int8_gemm.h"

int main(void)
{
    const int8_t activations[] = {
        1, 2, 3,
        -4, 5, -6,
    };
    const int8_t weights[] = {
        7, -8,
        9, 10,
        -11, 12,
    };
    const int32_t expected[] = {
        -8, 48,
        83, 10,
    };
    int32_t output[4] = {0};

    int8_gemm_reference(activations, weights, output, 2, 3, 2);

    for (int index = 0; index < 4; ++index) {
        if (output[index] != expected[index]) {
            fprintf(stderr, "Mismatch at %d: got %ld, expected %ld\n",
                    index, (long)output[index], (long)expected[index]);
            return 1;
        }
    }

    printf("INT8 GEMM reference test passed\n");
    return 0;
}