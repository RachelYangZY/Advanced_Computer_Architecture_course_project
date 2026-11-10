/*
 * Self-test for the integer-only softmax: compares op_softmax_i8 (fast), op_softmax_i8_slow and a
 * native float32 implementation of TFLite's SoftmaxInt8LUT on random rows and random scales.
 *
 *   gcc -O2 -ffp-contract=off -Isrc/gen tools/eval/softmax_selftest.c src/gen/ops_ref.c -lm -o /tmp/smx
 *   /tmp/smx [rows_per_scale=20000]
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ops_ref.h"

static uint64_t rng = 88172645463325252ULL;
static uint32_t rnd(void)
{
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)(rng >> 16);
}

static void tflite_ref(const float *table, float out_scale, int zp, const int8_t *x, int cols, int8_t *y)
{
    int mx = -128;
    for (int j = 0; j < cols; j++) if (x[j] > mx) mx = x[j];
    const float *off = &table[255 - mx];
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) sum += off[x[j]];
    volatile float inv = 1.0f / (sum * out_scale);
    for (int j = 0; j < cols; j++) {
        volatile float pr = off[x[j]] * inv;
        /* TFLite does not round p + 0.5 to float32 (measured), so add in double */
        int32_t q = (int32_t)((double)pr + 0.5) + zp;
        y[j] = (int8_t)(q > 127 ? 127 : q < -128 ? -128 : q);
    }
}

int main(int argc, char **argv)
{
    int rows_per = argc > 1 ? atoi(argv[1]) : 20000;
    long bad_fast = 0, bad_slow = 0, total = 0;
    const float out_scales[] = {1.0f / 256.0f, 1.0f / 128.0f, 0.00390625f};
    for (int t = 0; t < 40; t++) {
        float s_in = 0.005f + 0.012f * (float)t;           /* 0.005 .. 0.47 */
        float out_scale = out_scales[t % 3];
        int zp = -128;
        float table[256];                                   /* table[255 - v] = expf(-s*v) */
        uint32_t bits[256];
        int denormal = 0;
        for (int v = 0; v < 256; v++) {
            float e = expf(-s_in * (float)v);
            table[255 - v] = e;
            memcpy(&bits[v], &e, 4);                        /* indexed by d = v */
            if (e != 0.0f && e < 1.17549435e-38f) denormal = 1;
        }
        if (denormal) { printf("scale %.3f skipped (denormal entries)\n", s_in); continue; }
        uint32_t sbits;
        memcpy(&sbits, &out_scale, 4);
        for (int cols = 197; cols <= 197 + 1; cols += 1) {
            softmax_params_t p = {.rows = 1, .cols = cols, .exp_f32 = bits, .out_scale_bits = sbits, .zp_out = zp};
            int8_t x[512], y_ref[512], y_fast[512], y_slow[512];
            for (int r = 0; r < rows_per; r++) {
                int spread = 1 + (int)(rnd() % 256);
                int base = (int)(rnd() % 256) - 128;
                for (int j = 0; j < cols; j++) {
                    int v = base + (int)(rnd() % (uint32_t)spread) - spread / 2;
                    x[j] = (int8_t)(v > 127 ? 127 : v < -128 ? -128 : v);
                }
                tflite_ref(table, out_scale, zp, x, cols, y_ref);
                op_softmax_i8(&p, x, y_fast);
                op_softmax_i8_slow(&p, x, y_slow);
                if (memcmp(y_ref, y_fast, (size_t)cols)) bad_fast++;
                if (memcmp(y_ref, y_slow, (size_t)cols)) bad_slow++;
                total++;
            }
        }
    }
    printf("rows tested %ld: fast mismatching rows %ld, slow mismatching rows %ld\n", total, bad_fast, bad_slow);
    return (bad_fast || bad_slow) ? 1 : 0;
}
