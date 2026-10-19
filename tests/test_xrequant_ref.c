/* Host check: the xrequant oracle (xrequant_ref.h) equals TFLite's fixed-point arithmetic
 * (tflite_math.h) for modes 1 and 2 over millions of random and edge operands. */
#include <stdint.h>
#include <stdio.h>

#include "tflite_math.h"
#include "xrequant_ref.h"

static uint64_t s = 88172645463325252ULL;
static uint64_t rnd(void)
{
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

int main(void)
{
    uint64_t bad = 0, n = 0;
    for (int i = 0; i < 20000000; i++) {
        uint64_t r = rnd();
        int32_t acc = (int32_t)(r >> 8) >> (r >> 40 & 31);
        int32_t mult = (r & 1) ? (int32_t)(0x40000000u | ((uint32_t)rnd() & 0x3fffffffu)) : (int32_t)rnd();
        if (r >> 41 & 1) acc = (int32_t)(1u << (r >> 42 & 31));
        int shift = (int)((r >> 48) % 32), zp = (int)(int8_t)(r >> 56), mode = 1 + (int)((r >> 47) & 1);
        uint32_t ctrl = ((uint32_t)mode << 16) | ((uint32_t)shift << 8) | ((uint32_t)zp & 0xff);
        int32_t v = tfl_srdhm(acc, mult);
        int64_t t = mode == 1 ? (shift ? ((int64_t)v + (1LL << (shift - 1))) >> shift : v)
                              : (int64_t)tfl_rdivpot(v, shift);
        t += zp;
        int8_t want = (int8_t)(t > 127 ? 127 : (t < -128 ? -128 : t));
        n++;
        if (xrequant_reference(acc, mult, ctrl) != want && bad++ < 5)
            fprintf(stderr, "mismatch acc=%d mult=%d s=%d zp=%d mode=%d\n", acc, mult, shift, zp, mode);
    }
    printf("xrequant oracle vs tflite_math: %llu cases, %llu mismatches\n",
           (unsigned long long)n, (unsigned long long)bad);
    return bad != 0;
}
