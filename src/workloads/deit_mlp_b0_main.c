/* Block 0 MLP for one token (B0/B1): fc1 -> requant -> GELU LUT -> fc2 -> requant.
 * Scalar GEMM and scalar requant; no floating point. */
#include <stdint.h>

#include "int8_gemm.h"
#include "requantize.h"

#define DEIT_SCALAR_DATA
#include "deit_mlp_data.h"

static int32_t acc1[DEIT_MLP_HIDDEN];
static int32_t acc2[DEIT_MLP_OUT];
static int8_t hidden[DEIT_MLP_HIDDEN];   /* fc1 output (INT8) */
static int8_t gelu_out[DEIT_MLP_HIDDEN]; /* GELU output = fc2 input */
static int8_t result[DEIT_MLP_OUT];

volatile uint64_t tohost __attribute__((section(".tohost"), aligned(64)));
volatile uint64_t fromhost __attribute__((section(".tohost"), aligned(64)));

__attribute__((noinline)) static void stage_requant_fc1(void)
{
    for (int n = 0; n < DEIT_MLP_HIDDEN; ++n) {
        hidden[n] = requantize_i32_to_i8(acc1[n] + deit_mlp_bias1[n],
                                         deit_mlp_mult1[n], DEIT_MLP_SHIFT1,
                                         DEIT_MLP_ZP1);
    }
}

__attribute__((noinline)) static void stage_gelu(void)
{
    for (int n = 0; n < DEIT_MLP_HIDDEN; ++n) {
        gelu_out[n] = deit_mlp_gelu_lut[(uint8_t)(hidden[n] + 128)];
    }
}

__attribute__((noinline)) static void stage_requant_fc2(void)
{
    for (int n = 0; n < DEIT_MLP_OUT; ++n) {
        result[n] = requantize_i32_to_i8(acc2[n] + deit_mlp_bias2[n],
                                         deit_mlp_mult2[n], DEIT_MLP_SHIFT2,
                                         DEIT_MLP_ZP2);
    }
}

int main(void)
{
    int8_gemm_reference(deit_mlp_x, deit_mlp_w1_kn, acc1, 1, DEIT_MLP_IN,
                        DEIT_MLP_HIDDEN);
    stage_requant_fc1();
    stage_gelu();
    int8_gemm_reference(gelu_out, deit_mlp_w2_kn, acc2, 1, DEIT_MLP_HIDDEN,
                        DEIT_MLP_OUT);
    stage_requant_fc2();

    uint64_t passed = 1;
    for (int n = 0; n < DEIT_MLP_HIDDEN; ++n) {
        if (gelu_out[n] != deit_mlp_golden_g[n]) {
            passed = 0;
            break;
        }
    }
    for (int n = 0; n < DEIT_MLP_OUT; ++n) {
        if (result[n] != deit_mlp_golden_y[n]) {
            passed = 0;
            break;
        }
    }

    tohost = passed ? 1 : 3;
    for (;;) {
    }
}
