#ifndef INT8_GEMM_X1_H
#define INT8_GEMM_X1_H

#include <stdint.h>

void int8_gemm_x1(
    const uint64_t *packed_activations,
    const uint64_t *packed_weights,
    int32_t *output,
    int rows,
    int inner,
    int columns);

#endif
