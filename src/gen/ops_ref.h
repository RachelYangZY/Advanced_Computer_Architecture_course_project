#ifndef OPS_REF_H
#define OPS_REF_H

/*
 * Reference INT8 kernels used by the generated model (gen_model.py).
 * Semantics follow TFLite exactly. All parameters are precomputed on the host
 * and passed as const structs, so the device never does floating point.
 */

#include <stdint.h>

#define EW_MAX_RANK 5

/* FULLY_CONNECTED, per-channel weights [n][k], TFLite semantics. */
typedef struct {
    const int8_t *w;        /* [n][k] */
    const int32_t *bias;    /* [n] (zeros if the op has no bias) */
    const int32_t *mult;    /* [n] Q31 multipliers */
    const int32_t *shift;   /* [n] */
    int32_t in_zp, out_zp, act_min, act_max;
    int32_t rows, k, n;
} fc_params_t;

/* Binary elementwise op with broadcasting (ADD, SUB, MUL). */
typedef struct {
    int32_t rank;
    int32_t shape[EW_MAX_RANK];      /* output shape */
    int32_t stride_a[EW_MAX_RANK];   /* input strides, 0 where broadcast */
    int32_t stride_b[EW_MAX_RANK];
    int32_t zp_a, zp_b, zp_out, act_min, act_max;
    /* ADD/SUB */
    int32_t left_shift;
    int32_t mult_a, mult_b, mult_out;
    int32_t shift_a, shift_b, shift_out;
    /* MUL */
    int32_t mult, shift;
} ew_params_t;

/* TRANSPOSE with arbitrary rank <= EW_MAX_RANK. */
typedef struct {
    int32_t rank;
    int32_t shape[EW_MAX_RANK];      /* output shape */
    int32_t stride[EW_MAX_RANK];     /* input stride for each output dim */
} transpose_params_t;

void op_fc_i8(const fc_params_t *p, const int8_t *in, int8_t *out);
void op_lut_i8(const int8_t *lut256, const int8_t *in, int8_t *out, int32_t n);
void op_add_i8(const ew_params_t *p, const int8_t *a, const int8_t *b, int8_t *out);
void op_sub_i8(const ew_params_t *p, const int8_t *a, const int8_t *b, int8_t *out);
void op_mul_i8(const ew_params_t *p, const int8_t *a, const int8_t *b, int8_t *out);
void op_transpose_i8(const transpose_params_t *p, const int8_t *in, int8_t *out);

#endif
