#include <stdint.h>

#include "deit_fc1_tile_data.h"
#include "int8_gemm.h"

static int32_t output[DEIT_FC1_TILE_ROWS * DEIT_FC1_TILE_COLUMNS];

volatile uint64_t tohost __attribute__((section(".tohost"), aligned(64)));
volatile uint64_t fromhost __attribute__((section(".tohost"), aligned(64)));

int main(void)
{
    int8_gemm_reference(
        deit_fc1_activations,
        deit_fc1_weights,
        output,
        DEIT_FC1_TILE_ROWS,
        DEIT_FC1_TILE_INNER,
        DEIT_FC1_TILE_COLUMNS);

    uint64_t passed = 1;
    for (int index = 0; index < DEIT_FC1_TILE_ROWS * DEIT_FC1_TILE_COLUMNS;
         ++index) {
        if (output[index] != deit_fc1_golden[index]) {
            passed = 0;
            break;
        }
    }

    tohost = passed ? 1 : 3;
    for (;;) {
    }
}
