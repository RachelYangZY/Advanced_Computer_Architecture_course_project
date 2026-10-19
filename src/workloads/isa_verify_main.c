/*
 * Directed + randomized verification of xqmac8 and xrequant on the target (Spike).
 *
 * Every executed custom instruction is compared with its C oracle (src/kernels/xqmac8.c and
 * src/kernels/xrequant_ref.h). For the two TFLite modes of xrequant, the oracle is also compared with
 * the independent TFLite fixed-point helpers (src/gen/tflite_math.h), so a mistake common to the
 * instruction and its reference cannot hide.
 *
 * Exit code (tohost): 1 = pass, 3 = at least one failure. Prints a summary and the first failures.
 * Seed: ISA_VERIFY_SEED (default 1), random iterations: ISA_VERIFY_N (default 200000 each).
 */
#include <stddef.h>
#include <stdint.h>

#include "htif_print.h"
#include "tflite_math.h"
#include "xqmac8.h"
#include "xrequant_ref.h"

#ifndef ISA_VERIFY_SEED
#define ISA_VERIFY_SEED 1
#endif
#ifndef ISA_VERIFY_N
#define ISA_VERIFY_N 200000
#endif

static uint64_t rng_state = ISA_VERIFY_SEED * 0x9e3779b97f4a7c15ULL + 1;

static uint64_t rnd(void)   /* xorshift64* */
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 0x2545f4914f6cdd1dULL;
}

static inline int64_t hw_xqmac8(int64_t acc, uint64_t a, uint64_t b)
{
    __asm__(".insn r 0x0b, 0, 0x2b, %0, %1, %2" : "+r"(acc) : "r"(a), "r"(b));
    return acc;
}

static inline int8_t hw_xrequant(int32_t acc, int32_t mult, uint32_t ctrl)
{
    int64_t r = ctrl;
    __asm__(".insn r 0x0b, 1, 0x2b, %0, %1, %2" : "+r"(r) : "r"((int64_t)acc), "r"((int64_t)mult));
    return (int8_t)r;
}

static uint64_t n_tests, n_fail;

static void fail_xqmac(int64_t acc, uint64_t a, uint64_t b, int64_t got, int64_t want)
{
    n_fail++;
    if (n_fail <= 10) {
        put_s("FAIL xqmac8 acc="); put_x((uint64_t)acc); put_s(" a="); put_x(a);
        put_s(" b="); put_x(b); put_s(" got="); put_i(got); put_s(" want="); put_i(want);
        flush_line();
    }
}

static void check_xqmac(int64_t acc, uint64_t a, uint64_t b)
{
    n_tests++;
    int64_t got = hw_xqmac8(acc, a, b), want = xqmac8_reference(acc, a, b);
    if (got != want) fail_xqmac(acc, a, b, got, want);
}

static void check_xrequant(int32_t acc, int32_t mult, int s, int zp, int mode)
{
    uint32_t ctrl = ((uint32_t)mode << 16) | ((uint32_t)s << 8) | ((uint32_t)zp & 0xff);
    n_tests++;
    int8_t got = hw_xrequant(acc, mult, ctrl);
    int8_t want = xrequant_reference(acc, mult, ctrl);
    int8_t tfl = want;
    int have_tfl = 0;
    if (mode == 1 || mode == 2) {
        /* independent TFLite arithmetic: srdhm, then right shift with the mode's rounding */
        int32_t v = tfl_srdhm(acc, mult);
        int64_t t = mode == 1 ? (s ? (((int64_t)v + (1LL << (s - 1))) >> s) : v)
                              : (int64_t)tfl_rdivpot(v, s);
        t += zp;
        tfl = (int8_t)(t > 127 ? 127 : (t < -128 ? -128 : t));
        have_tfl = 1;
    }
    if (got != want || (have_tfl && tfl != want)) {
        n_fail++;
        if (n_fail <= 10) {
            put_s("FAIL xrequant acc="); put_i(acc); put_s(" mult="); put_i(mult);
            put_s(" shift="); put_i(s); put_s(" zp="); put_i(zp); put_s(" mode="); put_i(mode);
            put_s(" got="); put_i(got); put_s(" ref="); put_i(want); put_s(" tflite="); put_i(tfl);
            flush_line();
        }
    }
}

static const int32_t edge32[] = {
    0, 1, -1, 2, -2, 127, -128, 128, -129, 255, 256, 32767, -32768, 65536,
    (1 << 30) - 1, 1 << 30, -(1 << 30), INT32_MAX, INT32_MAX - 1, INT32_MIN, INT32_MIN + 1,
};
#define N_EDGE ((int)(sizeof edge32 / sizeof edge32[0]))

static int32_t rand32(void)
{
    uint64_t r = rnd();
    switch (r & 7) {
    case 0: return (int32_t)(r >> 8);                         /* full range */
    case 1: return (int32_t)(r >> 8) >> (r >> 40 & 31);       /* varied magnitude */
    case 2: return edge32[(r >> 8) % N_EDGE];
    case 3: return (int32_t)((r >> 8) % 2001) - 1000;         /* small */
    case 4: return (int32_t)((r >> 8) % 200001) - 100000;     /* typical accumulator */
    case 5: return 1 << ((r >> 8) % 31);                      /* powers of two (exact ties) */
    case 6: return -(1 << ((r >> 8) % 31));
    default: return (int32_t)(r >> 8) | (int32_t)0x40000000;
    }
}

int main(void)
{
    /* ---- xqmac8: directed ---- */
    static const uint64_t pat[] = {
        0x0000000000000000ULL, 0x8080808080808080ULL, 0x7f7f7f7f7f7f7f7fULL, 0x0101010101010101ULL,
        0xffffffffffffffffULL, 0x80007f00ff0180ffULL, 0x0123456789abcdefULL, 0x7f80807f7f80807fULL,
    };
    static const int64_t accs[] = {
        0, 1, -1, INT32_MAX, INT32_MIN, INT64_MAX, INT64_MIN, INT64_MAX - 100000, INT64_MIN + 100000,
    };
    for (unsigned i = 0; i < sizeof pat / sizeof pat[0]; i++)
        for (unsigned j = 0; j < sizeof pat / sizeof pat[0]; j++)
            for (unsigned k = 0; k < sizeof accs / sizeof accs[0]; k++)
                check_xqmac(accs[k], pat[i], pat[j]);
    for (int lane = 0; lane < 8; lane++)   /* one nonzero lane at a time: checks lane order */
        for (int va = -128; va <= 127; va += 51)
            for (int vb = -128; vb <= 127; vb += 51)
                check_xqmac(0, (uint64_t)(uint8_t)va << (8 * lane), (uint64_t)(uint8_t)vb << (8 * lane));
    uint64_t directed_xq = n_tests;

    /* ---- xqmac8: random ---- */
    for (int i = 0; i < ISA_VERIFY_N; i++) {
        uint64_t a = rnd(), b = rnd();
        int64_t acc = (rnd() & 1) ? (int64_t)rnd() : (int32_t)rnd();
        check_xqmac(acc, a, b);
    }
    uint64_t random_xq = n_tests - directed_xq;

    /* ---- xrequant: directed (edges x edges x shifts x modes) ---- */
    static const int shifts[] = { 0, 1, 2, 7, 8, 15, 16, 30, 31 };
    static const int zps[] = { 0, 1, -1, 127, -128, 55, -73 };
    uint64_t before = n_tests;
    for (int m = 1; m <= 2; m++)
        for (int a = 0; a < N_EDGE; a++)
            for (int b = 0; b < N_EDGE; b++)
                for (unsigned s = 0; s < sizeof shifts / sizeof shifts[0]; s++)
                    for (unsigned z = 0; z < sizeof zps / sizeof zps[0]; z++)
                        check_xrequant(edge32[a], edge32[b], shifts[s], zps[z], m);
    /* legacy mode, shift up to 63 */
    for (int a = 0; a < N_EDGE; a++)
        for (int b = 0; b < N_EDGE; b++)
            for (int s = 0; s <= 63; s += 9)
                check_xrequant(edge32[a], edge32[b], s, -5, 0);
    uint64_t directed_rq = n_tests - before;

    /* ---- xrequant: random ---- */
    before = n_tests;
    for (int i = 0; i < ISA_VERIFY_N; i++) {
        uint64_t r = rnd();
        int mode = (int)(r % 3);
        int s = (int)((r >> 8) % (mode == 0 ? 64 : 32));
        int zp = (int)(int8_t)(r >> 16);
        int32_t acc = rand32();
        int32_t mult = (r >> 24 & 3) == 0 ? (int32_t)(1u << (30 - (int)((r >> 28) % 8))) : rand32();
        /* TFLite multipliers are positive Q31 in [2^30, 2^31); bias half of the cases to that range */
        if ((r >> 32 & 1) != 0) mult = (int32_t)(0x40000000u | ((uint32_t)rnd() & 0x3fffffffu));
        check_xrequant(acc, mult, s, zp, mode);
    }
    uint64_t random_rq = n_tests - before;

    put_s("xqmac8:   directed="); put_u(directed_xq); put_s(" random="); put_u(random_xq); flush_line();
    put_s("xrequant: directed="); put_u(directed_rq); put_s(" random="); put_u(random_rq); flush_line();
    put_s("seed="); put_u(ISA_VERIFY_SEED); put_s(" tests="); put_u(n_tests);
    put_s(" failures="); put_u(n_fail); flush_line();
    put_s(n_fail ? "RESULT FAIL" : "RESULT PASS"); flush_line();

    tohost = n_fail ? 3 : 1;
    for (;;) {
    }
}
