/*
 * Optimized FULLY_CONNECTED / BATCH_MATMUL / CONV_2D kernels (link-time overrides of the weak
 * reference kernels in ops_ref.c; all other ops keep the reference implementation).
 *
 *   B1: this file as is (scalar, register-blocked core, zero points folded out of the inner loop)
 *   X1: -DGEMM_XQMAC8 -DUSE_XREQUANT (xqmac8 dot products, xrequant requantization)
 *       -DGEMM_XQMAC8 alone keeps the exact software requantization (X1 MAC-only variant).
 *
 * Shapes the kernels cannot handle (buffers too small, misaligned or k % 8 != 0 operands on X1)
 * fall back to the *_ref kernels.
 */
#include <stdint.h>

#include "gemm_core.h"
#include "ops_ref.h"
#include "tflite_math.h"

#define GEMM_MAX_N 1024
#define PATCH_MAX 1024
#define BMM_MAX_B (48 * 1024)

#ifdef GEMM_XQMAC8
#define KPAD(k) (((k) + 7) & ~7)
#define ALIGNED8(ptr) ((((uintptr_t)(ptr)) & 7) == 0)
#else
#define KPAD(k) (k)
#define ALIGNED8(ptr) 1
#endif

static int32_t acc_buf[2][GEMM_MAX_N];
static int32_t col_sum[GEMM_MAX_N];
static uint64_t bt_words[BMM_MAX_B / 8];
static uint64_t row_words[2][PATCH_MAX / 8];

#ifdef USE_XREQUANT
/* xrequant: (acc * mult + 2^(s-1)) >> s + zp, saturated to int8; ctrl = s << 8 | zp. */
static inline int32_t xrequant(int32_t acc, int32_t mult, int32_t shift, int32_t zp)
{
    int64_t r = (int64_t)(((uint64_t)(31 - shift) & 0xff) << 8 | ((uint64_t)zp & 0xff));
    __asm__(".insn r 0x0b, 1, 0x2b, %0, %1, %2" : "+r"(r) : "r"((int64_t)acc), "r"((int64_t)mult));
    return (int32_t)r;
}
#define RQ_UP(acc, mult, shift, zp, lo, hi) tfl_clamp_i8(xrequant((acc), (mult), (shift), (zp)), (lo), (hi))
#define RQ_GEMM(acc, mult, shift, zp, lo, hi) RQ_UP(acc, mult, shift, zp, lo, hi)
#else
#define RQ_UP(acc, mult, shift, zp, lo, hi) tfl_clamp_i8(tfl_mul_qm_up((acc), (mult), (shift)) + (zp), (lo), (hi))
#define RQ_GEMM(acc, mult, shift, zp, lo, hi) tfl_clamp_i8(tfl_mul_qm((acc), (mult), (shift)) + (zp), (lo), (hi))
#endif

void op_fc_i8(const fc_params_t *p, const int8_t *in, int8_t *out)
{
    const int32_t k = p->k, n = p->n;
    if (n > GEMM_MAX_N || KPAD(k) != k || !ALIGNED8(in) || !ALIGNED8(p->w)) {
        op_fc_i8_ref(p, in, out);
        return;
    }
    for (int32_t r = 0; r < p->rows; r += 2) {
        const int8_t *x0 = in + (int64_t)r * k;
        const int8_t *x1 = r + 1 < p->rows ? x0 + k : 0;
        gemm_nt_rows(x0, x1, p->w, k, n, k, acc_buf[0], acc_buf[1]);
        for (int32_t rr = 0; rr < (x1 ? 2 : 1); rr++) {
            int8_t *o = out + (int64_t)(r + rr) * n;
            const int32_t *acc = acc_buf[rr];
            for (int32_t c = 0; c < n; c++) {
                o[c] = RQ_UP(acc[c] + p->bias_eff[c], p->mult[c], p->shift[c], p->out_zp,
                             p->act_min, p->act_max);
            }
        }
    }
}

/* Copy k bytes into dst and zero the padding up to KPAD(k); returns the sum of the k bytes. */
static int32_t pack_row(int8_t *dst, const int8_t *src, int32_t k, int32_t stride)
{
    int32_t s = 0;
    for (int32_t i = 0; i < k; i++) {
        int8_t v = src[(int64_t)i * stride];
        dst[i] = v;
        s += v;
    }
    for (int32_t i = k; i < KPAD(k); i++) {
        dst[i] = 0;
    }
    return s;
}

void op_bmm_i8(const bmm_params_t *p, const int8_t *a, const int8_t *b, int8_t *out)
{
    const int32_t m = p->m, k = p->k, n = p->n, kp = KPAD(k);
    if (p->adj_x || n > GEMM_MAX_N || kp > PATCH_MAX || (int64_t)n * kp > BMM_MAX_B) {
        op_bmm_i8_ref(p, a, b, out);
        return;
    }
    int8_t *bt = (int8_t *)bt_words;
    const int32_t kzz = k * p->zp_a * p->zp_b;
    for (int32_t bi = 0; bi < p->batch; bi++) {
        const int8_t *A = a + (int64_t)bi * m * k;
        const int8_t *B = b + (int64_t)bi * k * n;
        int8_t *O = out + (int64_t)bi * m * n;
        /* B^T: one contiguous row of kp bytes per output column j */
        for (int32_t j = 0; j < n; j++) {
            const int8_t *src = p->adj_y ? B + (int64_t)j * k : B + j;
            col_sum[j] = pack_row(bt + (int64_t)j * kp, src, k, p->adj_y ? 1 : n);
        }
        for (int32_t i = 0; i < m; i += 2) {
            int32_t nr = i + 1 < m ? 2 : 1;
            const int8_t *x[2];
            int32_t row_sum[2];
            for (int32_t rr = 0; rr < nr; rr++) {
                const int8_t *src = A + (int64_t)(i + rr) * k;
                if (kp == k && ALIGNED8(src)) {
                    int32_t s = 0;
                    for (int32_t t = 0; t < k; t++) s += src[t];
                    x[rr] = src;
                    row_sum[rr] = s;
                } else {
                    x[rr] = (const int8_t *)row_words[rr];
                    row_sum[rr] = pack_row((int8_t *)row_words[rr], src, k, 1);
                }
            }
            gemm_nt_rows(x[0], nr == 2 ? x[1] : 0, bt, kp, n, kp, acc_buf[0], acc_buf[1]);
            for (int32_t rr = 0; rr < nr; rr++) {
                int8_t *o = O + (int64_t)(i + rr) * n;
                const int32_t *acc = acc_buf[rr];
                const int32_t row_term = kzz - p->zp_b * row_sum[rr];
                for (int32_t j = 0; j < n; j++) {
                    int32_t v = acc[j] + row_term - p->zp_a * col_sum[j];
                    o[j] = RQ_GEMM(v, p->mult, p->shift, p->zp_out, p->act_min, p->act_max);
                }
            }
        }
    }
}

/* im2col for one output position; out-of-image taps get in_zp, which the folded bias cancels. */
static void im2col(const conv_params_t *p, const int8_t *in, int32_t oy, int32_t ox, int8_t *dst)
{
    int8_t *d = dst;
    for (int32_t ky = 0; ky < p->k_h; ky++) {
        int32_t iy = oy * p->stride_h - p->pad_top + ky;
        for (int32_t kx = 0; kx < p->k_w; kx++) {
            int32_t ix = ox * p->stride_w - p->pad_left + kx;
            if (iy < 0 || iy >= p->in_h || ix < 0 || ix >= p->in_w) {
                for (int32_t c = 0; c < p->in_c; c++) d[c] = (int8_t)p->in_zp;
            } else {
                const int8_t *s = in + ((int64_t)iy * p->in_w + ix) * p->in_c;
                for (int32_t c = 0; c < p->in_c; c++) d[c] = s[c];
            }
            d += p->in_c;
        }
    }
}

void op_conv2d_i8(const conv_params_t *p, const int8_t *in, int8_t *out)
{
    const int32_t kk = p->k_h * p->k_w * p->in_c, oc = p->out_c;
    if (oc > GEMM_MAX_N || kk > PATCH_MAX || KPAD(kk) != kk || !ALIGNED8(p->w)) {
        op_conv2d_i8_ref(p, in, out);
        return;
    }
    const int32_t npos = p->out_h * p->out_w;
    for (int32_t pos = 0; pos < npos; pos += 2) {
        int32_t nr = pos + 1 < npos ? 2 : 1;
        for (int32_t rr = 0; rr < nr; rr++) {
            im2col(p, in, (pos + rr) / p->out_w, (pos + rr) % p->out_w, (int8_t *)row_words[rr]);
        }
        gemm_nt_rows((const int8_t *)row_words[0], nr == 2 ? (const int8_t *)row_words[1] : 0,
                     p->w, kk, oc, kk, acc_buf[0], acc_buf[1]);
        for (int32_t rr = 0; rr < nr; rr++) {
            int8_t *o = out + (int64_t)(pos + rr) * oc;
            const int32_t *acc = acc_buf[rr];
            for (int32_t c = 0; c < oc; c++) {
                o[c] = RQ_UP(acc[c] + p->bias_eff[c], p->mult[c], p->shift[c], p->out_zp,
                             p->act_min, p->act_max);
            }
        }
    }
}
