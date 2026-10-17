/* Generic DeiT linear-operator tile: Xqmac8 GEMM + xrequant (X1).
 * Build with -DDEIT_OP_HEADER=\"deit_<op>_data.h\" -DDEIT_PACKED_DATA. */
#include <stdint.h>

#include "int8_gemm_x1.h"
#include "xrequant.h"

#define DEIT_PACKED_DATA
#include DEIT_OP_HEADER

static int32_t accumulators[DEIT_ROWS * DEIT_COLUMNS];
static int8_t output[DEIT_ROWS * DEIT_COLUMNS];

volatile uint64_t tohost __attribute__((section(".tohost"), aligned(64)));
volatile uint64_t fromhost __attribute__((section(".tohost"), aligned(64)));

int main(void)
{
    int8_gemm_x1(deit_packed_activations, deit_packed_weights, accumulators,
                 DEIT_ROWS, DEIT_INNER, DEIT_COLUMNS);

    for (int row = 0; row < DEIT_ROWS; ++row) {
        for (int column = 0; column < DEIT_COLUMNS; ++column) {
            int index = row * DEIT_COLUMNS + column;
            output[index] = xrequant_i32(
                accumulators[index], deit_requant_multipliers[column],
                DEIT_REQUANT_SHIFT, DEIT_REQUANT_ZERO_POINT);
        }
    }

    uint64_t passed = 1;
    for (int index = 0; index < DEIT_ROWS * DEIT_COLUMNS; ++index) {
        if (accumulators[index] != deit_golden_i32[index] ||
            output[index] != deit_golden_i8[index]) {
            passed = 0;
            break;
        }
    }

    tohost = passed ? 1 : 3;
    for (;;) {
    }
}
