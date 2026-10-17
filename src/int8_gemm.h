#ifndef INT8_GEMM_H
#define INT8_GEMM_H

#include <stdint.h>

void int8_gemm_reference(
    const int8_t *activations,
    const int8_t *weights,
    int32_t *output,
    int rows,
    int inner,
    int columns);

#endif