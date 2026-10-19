#ifndef GEMM_CORE_H
#define GEMM_CORE_H

/*
 * Int8 "NT" dot-product core shared by the optimized FULLY_CONNECTED, BATCH_MATMUL and CONV_2D
 * kernels: acc[r][c] = sum_i x_r[i] * w_c[i] for up to two rows x and n weight rows w (both
 * contiguous along i). Zero points are handled by the callers (folded bias or row/column sums).
 *
 * B1 (default): scalar RV64IMC, 2 rows x 4 channels register blocking.
 * X1 (GEMM_XQMAC8): xqmac8, eight MACs per instruction; requires k % 8 == 0 and 8-byte aligned rows.
 */

#include <stdint.h>

#ifdef GEMM_XQMAC8
static inline int64_t xqmac8(int64_t acc, uint64_t a, uint64_t b)
{
    __asm__(".insn r 0x0b, 0, 0x2b, %0, %1, %2" : "+r"(acc) : "r"(a), "r"(b));
    return acc;
}
#endif

/* acc0[c] (and acc1[c] if x1 != 0) for c in [0, n); w rows are ldw bytes apart. */
static void gemm_nt_rows(const int8_t *x0, const int8_t *x1, const int8_t *w, int32_t ldw,
                         int32_t n, int32_t k, int32_t *acc0, int32_t *acc1)
{
    int32_t c = 0;
#ifdef GEMM_XQMAC8
    const int32_t kw = k / 8;
    const uint64_t *a0 = (const uint64_t *)x0;
    const uint64_t *a1 = (const uint64_t *)(x1 ? x1 : x0);
    for (; c + 4 <= n; c += 4) {
        const uint64_t *w0 = (const uint64_t *)(w + (int64_t)c * ldw);
        const uint64_t *w1 = (const uint64_t *)((const int8_t *)w0 + ldw);
        const uint64_t *w2 = (const uint64_t *)((const int8_t *)w1 + ldw);
        const uint64_t *w3 = (const uint64_t *)((const int8_t *)w2 + ldw);
        int64_t s00 = 0, s01 = 0, s02 = 0, s03 = 0, s10 = 0, s11 = 0, s12 = 0, s13 = 0;
        for (int32_t i = 0; i < kw; i++) {
            uint64_t v0 = a0[i], v1 = a1[i];
            uint64_t u0 = w0[i], u1 = w1[i], u2 = w2[i], u3 = w3[i];
            s00 = xqmac8(s00, v0, u0); s01 = xqmac8(s01, v0, u1);
            s02 = xqmac8(s02, v0, u2); s03 = xqmac8(s03, v0, u3);
            s10 = xqmac8(s10, v1, u0); s11 = xqmac8(s11, v1, u1);
            s12 = xqmac8(s12, v1, u2); s13 = xqmac8(s13, v1, u3);
        }
        acc0[c] = (int32_t)s00; acc0[c + 1] = (int32_t)s01; acc0[c + 2] = (int32_t)s02; acc0[c + 3] = (int32_t)s03;
        if (x1) {
            acc1[c] = (int32_t)s10; acc1[c + 1] = (int32_t)s11; acc1[c + 2] = (int32_t)s12; acc1[c + 3] = (int32_t)s13;
        }
    }
    for (; c < n; c++) {
        const uint64_t *w0 = (const uint64_t *)(w + (int64_t)c * ldw);
        int64_t s0 = 0, s1 = 0;
        for (int32_t i = 0; i < kw; i++) {
            s0 = xqmac8(s0, a0[i], w0[i]);
            s1 = xqmac8(s1, a1[i], w0[i]);
        }
        acc0[c] = (int32_t)s0;
        if (x1) acc1[c] = (int32_t)s1;
    }
#else
    const int8_t *y1 = x1 ? x1 : x0;
    for (; c + 4 <= n; c += 4) {
        const int8_t *w0 = w + (int64_t)c * ldw, *w1 = w0 + ldw, *w2 = w1 + ldw, *w3 = w2 + ldw;
        int32_t s00 = 0, s01 = 0, s02 = 0, s03 = 0, s10 = 0, s11 = 0, s12 = 0, s13 = 0;
        for (int32_t i = 0; i < k; i++) {
            int32_t v0 = x0[i], v1 = y1[i];
            int32_t u0 = w0[i], u1 = w1[i], u2 = w2[i], u3 = w3[i];
            s00 += v0 * u0; s01 += v0 * u1; s02 += v0 * u2; s03 += v0 * u3;
            s10 += v1 * u0; s11 += v1 * u1; s12 += v1 * u2; s13 += v1 * u3;
        }
        acc0[c] = s00; acc0[c + 1] = s01; acc0[c + 2] = s02; acc0[c + 3] = s03;
        if (x1) {
            acc1[c] = s10; acc1[c + 1] = s11; acc1[c + 2] = s12; acc1[c + 3] = s13;
        }
    }
    for (; c < n; c++) {
        const int8_t *w0 = w + (int64_t)c * ldw;
        int32_t s0 = 0, s1 = 0;
        for (int32_t i = 0; i < k; i++) {
            s0 += (int32_t)x0[i] * w0[i];
            s1 += (int32_t)y1[i] * w0[i];
        }
        acc0[c] = s0;
        if (x1) acc1[c] = s1;
    }
#endif
}

#endif
