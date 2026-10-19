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

/* MEAN over a set of axes (quantized, input/output scales differ). */
#define MEAN_MAX_OUT 4096
typedef struct {
    int32_t rank;
    int32_t in_shape[EW_MAX_RANK];
    int32_t reduce[EW_MAX_RANK];     /* 1 = axis is reduced */
    int32_t count;                   /* number of elements averaged per output */
    int32_t zp_in, zp_out, mult, shift;
    int32_t single_round;            /* 1: exact single rounding of sum*mult; 0: gemmlowp double rounding */
} mean_params_t;

/* SLICE: copy a [begin, begin+size) box, rank <= EW_MAX_RANK. */
typedef struct {
    int32_t rank;
    int32_t in_shape[EW_MAX_RANK];
    int32_t begin[EW_MAX_RANK];
    int32_t size[EW_MAX_RANK];       /* == output shape */
} slice_params_t;

/* BATCH_MATMUL on int8: out[b][i][j] = requant(sum_k (a - zp_a) * (b - zp_b)). */
typedef struct {
    int32_t batch, m, k, n;
    int32_t adj_x, adj_y;            /* a is [k][m] if adj_x; b is [n][k] if adj_y */
    int32_t zp_a, zp_b, zp_out, mult, shift, act_min, act_max;
} bmm_params_t;

/* SOFTMAX over the last axis (beta == 1, output scale 1/out_mul), table-driven integer math. */
typedef struct {
    int32_t rows, cols;
    const uint32_t *exp_tab;         /* exp(-d * in_scale) * 2^31, d = row_max - x in [0,255] */
    int32_t out_mul;                 /* 1 / out_scale (integer, e.g. 256) */
    int32_t zp_out;
} softmax_params_t;

/* CONV_2D, NHWC input (batch 1), weights [oc][kh][kw][ic] per-channel, explicit top/left padding. */
typedef struct {
    const int8_t *w;
    const int32_t *bias;             /* [oc] */
    const int32_t *mult;             /* [oc] Q31 multipliers */
    const int32_t *shift;            /* [oc] */
    int32_t in_h, in_w, in_c, out_h, out_w, out_c, k_h, k_w;
    int32_t stride_h, stride_w, pad_top, pad_left;
    int32_t in_zp, out_zp, act_min, act_max;
} conv_params_t;

/* CONCATENATION: each input contributes a chunk of in_chunk bytes per outer row. */
typedef struct {
    int32_t outer, out_chunk;
} concat_params_t;

void op_conv2d_i8(const conv_params_t *p, const int8_t *in, int8_t *out);
void op_concat_part_i8(const concat_params_t *p, const int8_t *in, int8_t *out,
                       int32_t in_chunk, int32_t offset);
/* QUANTIZE int8 -> int8: out = (x - zp_in) * real + zp_out (mult 0 means plain copy). */
void op_requant_i8(const int8_t *in, int8_t *out, int32_t n, int32_t zp_in, int32_t zp_out,
                   int32_t mult, int32_t shift);
void op_slice_i8(const slice_params_t *p, const int8_t *in, int8_t *out);
void op_bmm_i8(const bmm_params_t *p, const int8_t *a, const int8_t *b, int8_t *out);
void op_softmax_i8(const softmax_params_t *p, const int8_t *in, int8_t *out);
void op_mean_i8(const mean_params_t *p, const int8_t *in, int8_t *out);
void op_sqdiff_i8(const ew_params_t *p, const int8_t *a, const int8_t *b, int8_t *out);
void op_fc_i8(const fc_params_t *p, const int8_t *in, int8_t *out);
void op_lut_i8(const int8_t *lut256, const int8_t *in, int8_t *out, int32_t n);
void op_add_i8(const ew_params_t *p, const int8_t *a, const int8_t *b, int8_t *out);
void op_sub_i8(const ew_params_t *p, const int8_t *a, const int8_t *b, int8_t *out);
void op_mul_i8(const ew_params_t *p, const int8_t *a, const int8_t *b, int8_t *out);
void op_transpose_i8(const transpose_params_t *p, const int8_t *in, int8_t *out);

#endif
