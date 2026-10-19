#include <stdint.h>

#include "int8_gemm.h"

#define ACTIVATION_ROW 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
#define WEIGHT_ROW 1, -2, 3, -4, 5, -6, 7, -8

static const int8_t activations[] = {
    ACTIVATION_ROW,
    ACTIVATION_ROW,
    ACTIVATION_ROW,
    ACTIVATION_ROW,
    ACTIVATION_ROW,
    ACTIVATION_ROW,
    ACTIVATION_ROW,
    ACTIVATION_ROW,
};

static const int8_t weights[] = {
    WEIGHT_ROW,
    WEIGHT_ROW,
    WEIGHT_ROW,
    WEIGHT_ROW,
    WEIGHT_ROW,
    WEIGHT_ROW,
    WEIGHT_ROW,
    WEIGHT_ROW,
    WEIGHT_ROW,
    WEIGHT_ROW,
    WEIGHT_ROW,
    WEIGHT_ROW,
    WEIGHT_ROW,
    WEIGHT_ROW,
    WEIGHT_ROW,
    WEIGHT_ROW,
};

static int32_t output[64];

volatile uint64_t tohost __attribute__((section(".tohost"), aligned(64)));
volatile uint64_t fromhost __attribute__((section(".tohost"), aligned(64)));

int main(void)
{
    int8_gemm_reference(activations, weights, output, 8, 16, 8);
    uint64_t passed = output[0] == 136 && output[1] == -272 &&
                      output[7] == -1088 && output[63] == -1088;
    tohost = passed ? 1 : 3;
    for (;;) {
    }
}
