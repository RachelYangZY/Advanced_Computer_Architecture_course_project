#include <stdint.h>

#include "int8_gemm_x1.h"

static const uint64_t packed_activations[8][2] = {
    {UINT64_C(0x0807060504030201), UINT64_C(0x100f0e0d0c0b0a09)},
    {UINT64_C(0x0807060504030201), UINT64_C(0x100f0e0d0c0b0a09)},
    {UINT64_C(0x0807060504030201), UINT64_C(0x100f0e0d0c0b0a09)},
    {UINT64_C(0x0807060504030201), UINT64_C(0x100f0e0d0c0b0a09)},
    {UINT64_C(0x0807060504030201), UINT64_C(0x100f0e0d0c0b0a09)},
    {UINT64_C(0x0807060504030201), UINT64_C(0x100f0e0d0c0b0a09)},
    {UINT64_C(0x0807060504030201), UINT64_C(0x100f0e0d0c0b0a09)},
    {UINT64_C(0x0807060504030201), UINT64_C(0x100f0e0d0c0b0a09)},
};

static const uint64_t packed_weights[8][2] = {
    {UINT64_C(0x0101010101010101), UINT64_C(0x0101010101010101)},
    {UINT64_C(0xfefefefefefefefe), UINT64_C(0xfefefefefefefefe)},
    {UINT64_C(0x0303030303030303), UINT64_C(0x0303030303030303)},
    {UINT64_C(0xfcfcfcfcfcfcfcfc), UINT64_C(0xfcfcfcfcfcfcfcfc)},
    {UINT64_C(0x0505050505050505), UINT64_C(0x0505050505050505)},
    {UINT64_C(0xfafafafafafafafa), UINT64_C(0xfafafafafafafafa)},
    {UINT64_C(0x0707070707070707), UINT64_C(0x0707070707070707)},
    {UINT64_C(0xf8f8f8f8f8f8f8f8), UINT64_C(0xf8f8f8f8f8f8f8f8)},
};

static int32_t output[64];

volatile uint64_t tohost __attribute__((section(".tohost"), aligned(64)));
volatile uint64_t fromhost __attribute__((section(".tohost"), aligned(64)));

int main(void)
{
    int8_gemm_x1(&packed_activations[0][0], &packed_weights[0][0],
                 output, 8, 16, 8);
    uint64_t passed = output[0] == 136 && output[1] == -272 &&
                      output[7] == -1088 && output[63] == -1088;
    tohost = passed ? 1 : 3;
    for (;;) {
    }
}