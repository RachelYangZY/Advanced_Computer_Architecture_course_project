#include "ops_ref.h"

#include "tflite_math.h"

#define WEAK __attribute__((weak))

WEAK void op_fc_i8(const fc_params_t *p, const int8_t *in, int8_t *out) { op_fc_i8_ref(p, in, out); }
WEAK void op_bmm_i8(const bmm_params_t *p, const int8_t *a, const int8_t *b, int8_t *out) { op_bmm_i8_ref(p, a, b, out); }
WEAK void op_conv2d_i8(const conv_params_t *p, const int8_t *in, int8_t *out) { op_conv2d_i8_ref(p, in, out); }

void op_fc_i8_ref(const fc_params_t *p, const int8_t *in, int8_t *out)
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
        int32_t centered = acc[i] - p->zp_in * p->count;
        int32_t v;
        if (p->single_round) {
            /* round-half-away of centered * mult / 2^(31 - shift), one rounding step */
            int s = 31 - p->shift;
            int64_t prod = (int64_t)centered * p->mult;
            int64_t half = 1LL << (s - 1);
            v = (int32_t)(prod >= 0 ? (prod + half) >> s : -((-prod + half) >> s));
        } else {
            v = tfl_mul_qm(centered, p->mult, p->shift);
        }
        out[i] = tfl_clamp_i8(v + p->zp_out, -128, 127);
    }
}

void op_slice_i8(const slice_params_t *p, const int8_t *in, int8_t *out)
{
    int32_t in_stride[EW_MAX_RANK];
    int64_t acc = 1;
    for (int d = p->rank - 1; d >= 0; d--) {
        in_stride[d] = (int32_t)acc;
        acc *= p->in_shape[d];
    }
    int32_t total = 1;
    int64_t off = 0;
    for (int d = 0; d < p->rank; d++) {
        total *= p->size[d];
        off += (int64_t)p->begin[d] * in_stride[d];
    }
    int32_t idx[EW_MAX_RANK] = {0};
    for (int32_t n = 0; n < total; n++) {
        out[n] = in[off];
        for (int d = p->rank - 1; d >= 0; d--) {
            idx[d]++;
            off += in_stride[d];
            if (idx[d] < p->size[d]) {
                break;
            }
            off -= (int64_t)in_stride[d] * p->size[d];
            idx[d] = 0;
        }
    }
}

void op_bmm_i8_ref(const bmm_params_t *p, const int8_t *a, const int8_t *b, int8_t *out)
{
    for (int32_t bi = 0; bi < p->batch; bi++) {
        const int8_t *A = a + (int64_t)bi * p->m * p->k;
        const int8_t *B = b + (int64_t)bi * p->k * p->n;
        int8_t *O = out + (int64_t)bi * p->m * p->n;
        for (int32_t i = 0; i < p->m; i++) {
            for (int32_t j = 0; j < p->n; j++) {
                int32_t acc = 0;
                for (int32_t kk = 0; kk < p->k; kk++) {
                    int32_t av = p->adj_x ? A[(int64_t)kk * p->m + i] : A[(int64_t)i * p->k + kk];
                    int32_t bv = p->adj_y ? B[(int64_t)j * p->k + kk] : B[(int64_t)kk * p->n + j];
                    acc += (av - p->zp_a) * (bv - p->zp_b);
                }
                int32_t v = tfl_mul_qm(acc, p->mult, p->shift) + p->zp_out;
                O[(int64_t)i * p->n + j] = tfl_clamp_i8(v, p->act_min, p->act_max);
            }
        }
    }
}

/* ---- integer-only float32 emulation for non-negative values (used by softmax) ----
 * A value is m * 2^e. sf_round() rounds an exact m * 2^e to the nearest float32 (ties to even,
 * including denormals), so every operation below equals the IEEE single-precision result. */
typedef struct { uint64_t m; int32_t e; } sf_t;

static int sf_bitlen(uint64_t m)
{
    int n = 0;
    if (m >> 32) { n += 32; m >>= 32; }
    if (m >> 16) { n += 16; m >>= 16; }
    if (m >> 8) { n += 8; m >>= 8; }
    if (m >> 4) { n += 4; m >>= 4; }
    if (m >> 2) { n += 2; m >>= 2; }
    if (m >> 1) { n += 1; m >>= 1; }
    return n + (int)m;
}

static sf_t sf_from_bits(uint32_t u)
{
    uint32_t ex = (u >> 23) & 0xff, fr = u & 0x7fffff;
    sf_t r;
    if (ex == 0) { r.m = fr; r.e = -149; }
    else { r.m = fr | (1u << 23); r.e = (int32_t)ex - 150; }
    return r;
}

static sf_t sf_round(uint64_t m, int32_t e)
{
    sf_t z = {0, 0};
    if (m == 0) return z;
    int nb = sf_bitlen(m);
    int32_t lsb = e + (nb > 24 ? nb - 24 : 0);
    if (lsb < -149) lsb = -149;
    int32_t sh = lsb - e;
    sf_t r;
    r.e = lsb;
    if (sh <= 0) { r.m = m << (-sh); return r; }
    if (sh > 64) return z;
    uint64_t q, rem, half;
    if (sh == 64) { q = 0; rem = m; half = 1ULL << 63; }
    else { q = m >> sh; rem = m & ((1ULL << sh) - 1); half = 1ULL << (sh - 1); }
    if (rem > half || (rem == half && (q & 1))) q++;
    r.m = q;
    return r;
}

static sf_t sf_add(sf_t a, sf_t b)   /* float32 a + b, a and b >= 0 */
{
    if (a.m == 0) return b;
    if (b.m == 0) return a;
    int ta = a.e + sf_bitlen(a.m), tb = b.e + sf_bitlen(b.m);
    if (ta - tb > 30) return a;   /* b is below half an ulp of a */
    if (tb - ta > 30) return b;
    int32_t e = a.e < b.e ? a.e : b.e;
    return sf_round((a.m << (a.e - e)) + (b.m << (b.e - e)), e);
}

static sf_t sf_mul(sf_t a, sf_t b)
{
    sf_t z = {0, 0};
    if (a.m == 0 || b.m == 0) return z;
    return sf_round(a.m * b.m, a.e + b.e);
}

static sf_t sf_recip(sf_t x)   /* 1.0f / x, x > 0 */
{
    uint64_t n = 1ULL << 60;
    uint64_t q = n / x.m;
    if (n % x.m) q |= 1;       /* sticky bit: q has far more than 24 significant bits */
    return sf_round(q, -x.e - 60);
}

/* Straightforward emulation, one generic float32 operation at a time (slow). Kept as the oracle
 * for op_softmax_i8 below; tools/eval/softmax_selftest.c checks that both agree bit for bit. */
void op_softmax_i8_slow(const softmax_params_t *p, const int8_t *in, int8_t *out)
{
    const sf_t scale = sf_from_bits(p->out_scale_bits);
    for (int32_t r = 0; r < p->rows; r++) {
        const int8_t *x = in + (int64_t)r * p->cols;
        int8_t *y = out + (int64_t)r * p->cols;
        int32_t mx = -128;
        for (int32_t j = 0; j < p->cols; j++) {
            if (x[j] > mx) mx = x[j];
        }
        sf_t sum = {0, 0};   /* sequential float32 sum, as TFLite */
        for (int32_t j = 0; j < p->cols; j++) {
            sum = sf_add(sum, sf_from_bits(p->exp_f32[mx - x[j]]));
        }
        const sf_t inv = sf_recip(sf_mul(sum, scale));
        for (int32_t j = 0; j < p->cols; j++) {
            sf_t s = sf_mul(sf_from_bits(p->exp_f32[mx - x[j]]), inv);   /* float32 p */
            int32_t q;                       /* floor(p + 0.5), exact */
            if (s.e >= 0) q = s.e > 20 ? 1 << 20 : (int32_t)(s.m << s.e);
            else q = -s.e >= 64 ? 0 : (int32_t)((s.m + (1ULL << (-s.e - 1))) >> -s.e);
            if (q > 1000) q = 1000;
            y[j] = tfl_clamp_i8(q + p->zp_out, -128, 127);
        }
    }
}

/*
 * Fast bit-exact softmax. Same float32 semantics as op_softmax_i8_slow, specialised:
 *  - table entries are normal float32 (or zero; the generator rejects denormals), so a value is a
 *    24-bit mantissa in [2^23, 2^24) and an exponent; no normalisation loops are needed;
 *  - sum: sequential float32 add of non-negative values, rounded to nearest even each step;
 *  - e*inv: the 48-bit product is rounded to 24 bits (float32 p); the quantized value is
 *    floor(p + 0.5) computed exactly. TFLite does NOT round p + 0.5 to float32 first: with
 *    p = 0.5 - 2^-25 a float32 add gives 1.0 (quantized 1) but TFLite returns 0; measured on 6
 *    validation images, the exact form has 0 mismatches and the float32 add 6.
 */
void op_softmax_i8(const softmax_params_t *p, const int8_t *in, int8_t *out)
{
    const sf_t scale = sf_from_bits(p->out_scale_bits);
    for (int32_t r = 0; r < p->rows; r++) {
        const int8_t *x = in + (int64_t)r * p->cols;
        int8_t *y = out + (int64_t)r * p->cols;
        int32_t mx = -128;
        for (int32_t j = 0; j < p->cols; j++) {
            if (x[j] > mx) mx = x[j];
        }

        uint64_t S = 0;   /* running sum = S * 2^E, S in [2^23, 2^24) once non-zero */
        int32_t E = 0;
        for (int32_t j = 0; j < p->cols; j++) {
            uint32_t u = p->exp_f32[mx - x[j]];
            if (u == 0) continue;
            uint64_t vm = (u & 0x7fffffu) | 0x800000u;
            int32_t ve = (int32_t)(u >> 23) - 150;
            if (S == 0) { S = vm; E = ve; continue; }
            int32_t ta = E + 24, tb = ve + 24;
            if (ta - tb > 30) continue;                      /* v below half an ulp of the sum */
            if (tb - ta > 30) { S = vm; E = ve; continue; }  /* the sum is below half an ulp of v */
            int32_t emin = E < ve ? E : ve;
            uint64_t T = (S << (E - emin)) + (vm << (ve - emin));
            int32_t tt = (ta > tb ? ta : tb) - emin;         /* bit count of the larger term */
            int32_t sh = tt - 24 + (T >> tt ? 1 : 0);
            if (sh > 0) {
                uint64_t q = T >> sh, rem = T & ((1ULL << sh) - 1), half = 1ULL << (sh - 1);
                if (rem > half || (rem == half && (q & 1))) q++;
                T = q;
            }
            S = T;
            E = emin + sh;
            if (S >> 24) { S >>= 1; E++; }
        }

        sf_t sum = {S, E};
        sf_t inv = sf_recip(sf_mul(sum, scale));
        if (inv.m >> 24) { inv.m >>= 1; inv.e++; }

        for (int32_t j = 0; j < p->cols; j++) {
            uint32_t u = p->exp_f32[mx - x[j]];
            int32_t q = 0;
            if (u) {
                uint64_t M = (uint64_t)((u & 0x7fffffu) | 0x800000u) * inv.m;   /* [2^46, 2^48) */
                int32_t ep = (int32_t)(u >> 23) - 150 + inv.e;
                int32_t sh = 23 + (int32_t)(M >> 47);
                uint64_t P = M >> sh, rem = M & ((1ULL << sh) - 1), half = 1ULL << (sh - 1);
                if (rem > half || (rem == half && (P & 1))) P++;
                ep += sh;                                    /* p = P * 2^ep */
                int32_t k = ep + 25;
                if (k > 36) q = 1000;
                else {
                    uint64_t X = k >= 0 ? (P << k) : (-k >= 64 ? 0 : (P >> -k));
                    q = (int32_t)((X + (1ULL << 24)) >> 25);
                    if (q > 1000) q = 1000;
                }
            } else {
                q = 0;
            }
            y[j] = tfl_clamp_i8(q + p->zp_out, -128, 127);
        }
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

void op_conv2d_i8_ref(const conv_params_t *p, const int8_t *in, int8_t *out)
{
    for (int32_t oy = 0; oy < p->out_h; oy++) {
        for (int32_t ox = 0; ox < p->out_w; ox++) {
            int8_t *o = out + ((int64_t)oy * p->out_w + ox) * p->out_c;
            for (int32_t oc = 0; oc < p->out_c; oc++) {
                const int8_t *w = p->w + (int64_t)oc * p->k_h * p->k_w * p->in_c;
                int32_t acc = 0;
                for (int32_t ky = 0; ky < p->k_h; ky++) {
                    int32_t iy = oy * p->stride_h - p->pad_top + ky;
                    if (iy < 0 || iy >= p->in_h) continue;
                    for (int32_t kx = 0; kx < p->k_w; kx++) {
                        int32_t ix = ox * p->stride_w - p->pad_left + kx;
                        if (ix < 0 || ix >= p->in_w) continue;
                        const int8_t *x = in + ((int64_t)iy * p->in_w + ix) * p->in_c;
                        const int8_t *wk = w + ((int64_t)ky * p->k_w + kx) * p->in_c;
                        for (int32_t c = 0; c < p->in_c; c++) {
                            acc += ((int32_t)x[c] - p->in_zp) * (int32_t)wk[c];
                        }
                    }
                }
                acc += p->bias[oc];
                acc = tfl_mul_qm_up(acc, p->mult[oc], p->shift[oc]);
                o[oc] = tfl_clamp_i8(acc + p->out_zp, p->act_min, p->act_max);
            }
        }
    }
}

void op_concat_part_i8(const concat_params_t *p, const int8_t *in, int8_t *out,
                       int32_t in_chunk, int32_t offset)
{
    for (int32_t r = 0; r < p->outer; r++) {
        const int8_t *s = in + (int64_t)r * in_chunk;
        int8_t *d = out + (int64_t)r * p->out_chunk + offset;
        for (int32_t i = 0; i < in_chunk; i++) {
            d[i] = s[i];
        }
    }
}

void op_requant_i8(const int8_t *in, int8_t *out, int32_t n, int32_t zp_in, int32_t zp_out,
                   int32_t mult, int32_t shift)
{
    for (int32_t i = 0; i < n; i++) {
        if (mult == 0) {
            out[i] = in[i];
        } else {
            int32_t v = tfl_mul_qm((int32_t)in[i] - zp_in, mult, shift) + zp_out;
            out[i] = tfl_clamp_i8(v, -128, 127);
        }
    }
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
