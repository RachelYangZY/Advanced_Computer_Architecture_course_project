#include "int8_gemm.h"

void int8_gemm_reference(
    const int8_t *activations,
    const int8_t *weights,
    int32_t *output,
    int rows,
    int inner,
    int columns)
{
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            int32_t accumulator = 0;
            for (int index = 0; index < inner; ++index) {
                int activation_index = row * inner + index;
                int weight_index = index * columns + column;
                accumulator += (int32_t)activations[activation_index] *
                               (int32_t)weights[weight_index];
            }
            output[row * columns + column] = accumulator;
        }
    }
}