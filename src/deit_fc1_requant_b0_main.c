#include <stdint.h>

#include "deit_fc1_requant_data.h"
#include "int8_gemm.h"
#include "requantize.h"

static int32_t accumulators[DEIT_FC1_TILE_ROWS * DEIT_FC1_TILE_COLUMNS];
static int8_t output[DEIT_FC1_TILE_ROWS * DEIT_FC1_TILE_COLUMNS];

volatile uint64_t tohost __attribute__((section(".tohost"), aligned(64)));
volatile uint64_t fromhost __attribute__((section(".tohost"), aligned(64)));

int main(void)
{
    int8_gemm_reference(
        deit_fc1_activations,
        deit_fc1_weights,
        accumulators,
        DEIT_FC1_TILE_ROWS,
        DEIT_FC1_TILE_INNER,
        DEIT_FC1_TILE_COLUMNS);

    for (int row = 0; row < DEIT_FC1_TILE_ROWS; ++row) {
        for (int column = 0; column < DEIT_FC1_TILE_COLUMNS; ++column) {
            int index = row * DEIT_FC1_TILE_COLUMNS + column;
            output[index] = requantize_i32_to_i8(
                accumulators[index],
                deit_fc1_requant_multipliers[column],
                DEIT_FC1_REQUANT_SHIFT,
                DEIT_FC1_REQUANT_ZERO_POINT);
        }
    }

    uint64_t passed = 1;
    for (int index = 0; index < DEIT_FC1_TILE_ROWS * DEIT_FC1_TILE_COLUMNS;
         ++index) {
        if (output[index] != deit_fc1_requant_golden[index]) {
            passed = 0;
            break;
        }
    }

    tohost = passed ? 1 : 3;
    for (;;) {
    }
}
