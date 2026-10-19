/*
 * Bare-metal driver for a generated model (Spike / RV64, built with -DMODEL_PROF).
 *
 * Loads the embedded input into the arena, runs model_run(), compares every boundary output with
 * the embedded TFLite golden bytes, and prints one PROF line per op plus a summary over HTIF.
 * Exit code: 0 = all outputs match, 1 = mismatch.
 */
#include <stddef.h>
#include <stdint.h>

#include "model.h"
#include "model_io.h"

volatile uint64_t tohost __attribute__((section(".tohost"), aligned(64)));
volatile uint64_t fromhost __attribute__((section(".tohost"), aligned(64)));

/* gcc may emit calls to these even in freestanding code */
void *memcpy(void *dst, const void *src, size_t n)
{
    uint8_t *d = dst;
    const uint8_t *s = src;
    while (n--) *d++ = *s++;
    return dst;
}

void *memset(void *dst, int v, size_t n)
{
    uint8_t *d = dst;
    while (n--) *d++ = (uint8_t)v;
    return dst;
}

static void htif_write(const char *buf, size_t len)
{
    static volatile uint64_t magic[8] __attribute__((aligned(64)));
    magic[0] = 64; /* SYS_write */
    magic[1] = 1;  /* stdout */
    magic[2] = (uintptr_t)buf;
    magic[3] = len;
    __sync_synchronize();
    tohost = (uintptr_t)magic;
    while (fromhost == 0) {
    }
    fromhost = 0;
}

static char line[160];
static int line_len;

static void put_s(const char *s)
{
    while (*s && line_len < (int)sizeof line - 1) line[line_len++] = *s++;
}

static void put_u(uint64_t v)
{
    char tmp[24];
    int n = 0;
    do {
        tmp[n++] = (char)('0' + v % 10);
        v /= 10;
    } while (v);
    while (n && line_len < (int)sizeof line - 1) line[line_len++] = tmp[--n];
}

static void put_i(int64_t v)
{
    if (v < 0) {
        put_s("-");
        v = -v;
    }
    put_u((uint64_t)v);
}

static void flush_line(void)
{
    line[line_len++] = '\n';
    htif_write(line, (size_t)line_len);
    line_len = 0;
}

static int argmax(const int8_t *v, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

int main(void)
{
    int8_t *arena = model_arena();
    for (int i = 0; i < model_io_count; i++) {
        const model_io_t *t = &model_io[i];
        if (t->is_input) memcpy(arena + t->offset, t->data, (size_t)t->nbytes);
    }

    model_run();

    for (int i = 0; i < model_op_count; i++) {
        put_s("PROF op=");
        put_i(model_ops[i].idx);
        put_s(" kind=");
        put_s(model_ops[i].kind);
        put_s(" macs=");
        put_i(model_ops[i].macs);
        put_s(" instret=");
        put_u(model_prof_instret[i + 1] - model_prof_instret[i]);
        put_s(" cycles=");
        put_u(model_prof_cycle[i + 1] - model_prof_cycle[i]);
        flush_line();
    }
    put_s("TOTAL instret=");
    put_u(model_prof_instret[model_op_count] - model_prof_instret[0]);
    put_s(" cycles=");
    put_u(model_prof_cycle[model_op_count] - model_prof_cycle[0]);
    flush_line();

    int failed = 0;
    for (int i = 0; i < model_io_count; i++) {
        const model_io_t *t = &model_io[i];
        if (t->is_input) continue;
        const int8_t *got = arena + t->offset;
        int mism = 0, maxd = 0;
        for (int j = 0; j < t->nbytes; j++) {
            int d = got[j] - t->data[j];
            if (d < 0) d = -d;
            if (d) mism++;
            if (d > maxd) maxd = d;
        }
        failed |= mism != 0;
        put_s("CHECK t");
        put_i(t->idx);
        put_s(" n=");
        put_i(t->nbytes);
        put_s(" mismatches=");
        put_i(mism);
        put_s(" max_diff=");
        put_i(maxd);
        put_s(" top1=");
        put_i(argmax(got, t->nbytes));
        put_s(" golden_top1=");
        put_i(argmax(t->data, t->nbytes));
        flush_line();
    }
    put_s(failed ? "RESULT FAIL" : "RESULT PASS");
    flush_line();

    tohost = failed ? 3 : 1;
    for (;;) {
    }
}
