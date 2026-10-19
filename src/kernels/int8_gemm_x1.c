#include "int8_gemm_x1.h"

static int32_t xqmac8_dot8(
    int32_t accumulator,
    uint64_t packed_a,
    uint64_t packed_b)
{
    register int64_t result asm("a0") = accumulator;
    register uint64_t source_a asm("a1") = packed_a;
    register uint64_t source_b asm("a2") = packed_b;

    asm volatile(".word 0x56c5850b"
                 : "+r"(result)
                 : "r"(source_a), "r"(source_b));
    return (int32_t)result;
}

void int8_gemm_x1(
    const uint64_t *packed_activations,
    const uint64_t *packed_weights,
    int32_t *output,
    int rows,
    int inner,
    int columns)
{
    const int packed_words = inner / 8;

    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            const uint64_t *activation_row =
                &packed_activations[row * packed_words];
            const uint64_t *weight_column =
                &packed_weights[column * packed_words];
            int32_t accumulator = 0;

            for (int word = 0; word < packed_words; ++word) {
                accumulator = xqmac8_dot8(
                    accumulator, activation_row[word], weight_column[word]);
            }
            output[row * columns + column] = accumulator;
        }
    }
}
