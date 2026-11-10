/* Host check: the xrequant oracle (xrequant_ref.h) equals TFLite's fixed-point arithmetic
 * (tflite_math.h: MultiplyByQuantizedMultiplier, half-up variant for mode 1) for modes 1 and 2,
 * including positive (left) shifts, over millions of random and edge operands. */
#include <stdint.h>
#include <stdio.h>

#include "tflite_math.h"
#include "xrequant_encoding.h"
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
        int shift = (int)((r >> 48) % 63) - 31;   /* TFLite signed shift: -31..31 */
        int zp = (int)(int8_t)(r >> 56), mode = 1 + (int)((r >> 47) & 1);
        uint64_t rs2 = xrequant_pack(mult, shift, zp, mode);
        int32_t q = mode == 1 ? tfl_mul_qm_up(acc, mult, shift) : tfl_mul_qm(acc, mult, shift);
        int64_t t = (int64_t)q + zp;
        int8_t want = (int8_t)(t > 127 ? 127 : (t < -128 ? -128 : t));
        n++;
        if (!xrequant_legal(rs2)) {
            fprintf(stderr, "legal() rejected shift=%d mode=%d\n", shift, mode);
            bad++;
            continue;
        }
        if (xrequant_reference(acc, rs2) != want && bad++ < 5)
            fprintf(stderr, "mismatch acc=%d mult=%d s=%d zp=%d mode=%d\n", acc, mult, shift, zp, mode);
    }
    printf("xrequant oracle vs tflite_math: %llu cases, %llu mismatches\n",
           (unsigned long long)n, (unsigned long long)bad);
    return bad != 0;
}
