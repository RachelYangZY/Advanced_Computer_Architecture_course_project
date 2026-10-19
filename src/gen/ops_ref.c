#include "ops_ref.h"

#include "tflite_math.h"

void op_fc_i8(const fc_params_t *p, const int8_t *in, int8_t *out)
{
    for (int32_t r = 0; r < p->rows; r++) {
        const int8_t *x = in + (int64_t)r * p->k;
        for (int32_t c = 0; c < p->n; c++) {
            const int8_t *w = p->w + (int64_t)c * p->k;
            int32_t acc = 0;
            for (int32_t i = 0; i < p->k; i++) {
                acc += ((int32_t)x[i] - p->in_zp) * (int32_t)w[i];
            }
            acc += p->bias[c];
            acc = tfl_mul_qm_up(acc, p->mult[c], p->shift[c]);
            acc += p->out_zp;
            out[(int64_t)r * p->n + c] = tfl_clamp_i8(acc, p->act_min, p->act_max);
        }
    }
}

void op_lut_i8(const int8_t *lut256, const int8_t *in, int8_t *out, int32_t n)
{
    for (int32_t i = 0; i < n; i++) {
        out[i] = lut256[(int32_t)in[i] + 128];
    }
}

typedef int8_t (*ew_fn)(const ew_params_t *p, int8_t a, int8_t b);

static int8_t add_one(const ew_params_t *p, int8_t a, int8_t b)
{
    int32_t sa = ((int32_t)a - p->zp_a) * (1 << p->left_shift);
    int32_t sb = ((int32_t)b - p->zp_b) * (1 << p->left_shift);
    sa = tfl_mul_qm(sa, p->mult_a, p->shift_a);
    sb = tfl_mul_qm(sb, p->mult_b, p->shift_b);
    int32_t raw = tfl_mul_qm(sa + sb, p->mult_out, p->shift_out) + p->zp_out;
    return tfl_clamp_i8(raw, p->act_min, p->act_max);
}

static int8_t sub_one(const ew_params_t *p, int8_t a, int8_t b)
{
    int32_t sa = ((int32_t)a - p->zp_a) * (1 << p->left_shift);
    int32_t sb = ((int32_t)b - p->zp_b) * (1 << p->left_shift);
    sa = tfl_mul_qm(sa, p->mult_a, p->shift_a);
    sb = tfl_mul_qm(sb, p->mult_b, p->shift_b);
    int32_t raw = tfl_mul_qm(sa - sb, p->mult_out, p->shift_out) + p->zp_out;
    return tfl_clamp_i8(raw, p->act_min, p->act_max);
}

static int8_t mul_one(const ew_params_t *p, int8_t a, int8_t b)
{
    int32_t prod = ((int32_t)a - p->zp_a) * ((int32_t)b - p->zp_b);
    int32_t raw = tfl_mul_qm(prod, p->mult, p->shift) + p->zp_out;
    return tfl_clamp_i8(raw, p->act_min, p->act_max);
}

/* TFLite SQUARED_DIFFERENCE (int8): inputs scaled by left_shift, diff squared, then requantized. */
static int8_t sqdiff_one(const ew_params_t *p, int8_t a, int8_t b)
{
    int32_t sa = ((int32_t)a - p->zp_a) * (1 << p->left_shift);
    int32_t sb = ((int32_t)b - p->zp_b) * (1 << p->left_shift);
    sa = tfl_mul_qm(sa, p->mult_a, p->shift_a);
    sb = tfl_mul_qm(sb, p->mult_b, p->shift_b);
    int32_t d = sa - sb;
    int32_t raw = tfl_mul_qm(d * d, p->mult_out, p->shift_out) + p->zp_out;
    return tfl_clamp_i8(raw, p->act_min, p->act_max);
}

static void ew_run(const ew_params_t *p, const int8_t *a, const int8_t *b, int8_t *out, ew_fn fn)
{
    int32_t total = 1;
    for (int d = 0; d < p->rank; d++) {
        total *= p->shape[d];
    }
    int32_t idx[EW_MAX_RANK] = {0};
    int64_t oa = 0, ob = 0;
    for (int32_t n = 0; n < total; n++) {
        out[n] = fn(p, a[oa], b[ob]);
        for (int d = p->rank - 1; d >= 0; d--) {
            idx[d]++;
            oa += p->stride_a[d];
            ob += p->stride_b[d];
            if (idx[d] < p->shape[d]) {
                break;
            }
            oa -= (int64_t)p->stride_a[d] * p->shape[d];
            ob -= (int64_t)p->stride_b[d] * p->shape[d];
            idx[d] = 0;
        }
    }
}

void op_sqdiff_i8(const ew_params_t *p, const int8_t *a, const int8_t *b, int8_t *out)
{
    ew_run(p, a, b, out, sqdiff_one);
}

void op_mean_i8(const mean_params_t *p, const int8_t *in, int8_t *out)
{
    static int32_t acc[MEAN_MAX_OUT];
    int32_t ostride[EW_MAX_RANK];
    int32_t n_out = 1, total = 1;
    for (int d = p->rank - 1; d >= 0; d--) {
        total *= p->in_shape[d];
        if (p->reduce[d]) {
            ostride[d] = 0;
        } else {
            ostride[d] = n_out;
            n_out *= p->in_shape[d];
        }
    }
    for (int32_t i = 0; i < n_out; i++) {
        acc[i] = 0;
    }
    int32_t idx[EW_MAX_RANK] = {0};
    int64_t o = 0;
    for (int32_t n = 0; n < total; n++) {
        acc[o] += in[n];
        for (int d = p->rank - 1; d >= 0; d--) {
            idx[d]++;
            o += ostride[d];
            if (idx[d] < p->in_shape[d]) {
                break;
            }
            o -= (int64_t)ostride[d] * p->in_shape[d];
            idx[d] = 0;
        }
    }
    for (int32_t i = 0; i < n_out; i++) {
        int32_t v = tfl_mul_qm(acc[i] - p->zp_in * p->count, p->mult, p->shift) + p->zp_out;
        out[i] = tfl_clamp_i8(v, -128, 127);
    }
}

void op_add_i8(const ew_params_t *p, const int8_t *a, const int8_t *b, int8_t *out)
{
    ew_run(p, a, b, out, add_one);
}

void op_sub_i8(const ew_params_t *p, const int8_t *a, const int8_t *b, int8_t *out)
{
    ew_run(p, a, b, out, sub_one);
}

void op_mul_i8(const ew_params_t *p, const int8_t *a, const int8_t *b, int8_t *out)
{
    ew_run(p, a, b, out, mul_one);
}

void op_transpose_i8(const transpose_params_t *p, const int8_t *in, int8_t *out)
{
    int32_t total = 1;
    for (int d = 0; d < p->rank; d++) {
        total *= p->shape[d];
    }
    int32_t idx[EW_MAX_RANK] = {0};
    int64_t off = 0;
    for (int32_t n = 0; n < total; n++) {
        out[n] = in[off];
        for (int d = p->rank - 1; d >= 0; d--) {
            idx[d]++;
            off += p->stride[d];
            if (idx[d] < p->shape[d]) {
                break;
            }
            off -= (int64_t)p->stride[d] * p->shape[d];
            idx[d] = 0;
        }
    }
}
