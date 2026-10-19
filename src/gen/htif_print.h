#ifndef HTIF_PRINT_H
#define HTIF_PRINT_H

/* Minimal bare-metal text output over the HTIF write syscall (Spike). Include in exactly one file
 * per program, after defining nothing else: it defines tohost/fromhost. */

#include <stddef.h>
#include <stdint.h>

volatile uint64_t tohost __attribute__((section(".tohost"), aligned(64)));
volatile uint64_t fromhost __attribute__((section(".tohost"), aligned(64)));

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

static void put_x(uint64_t v)
{
    put_s("0x");
    for (int i = 60; i >= 0; i -= 4) {
        int d = (int)((v >> i) & 0xf);
        char c[2] = { (char)(d < 10 ? '0' + d : 'a' + d - 10), 0 };
        put_s(c);
    }
}

static void flush_line(void)
{
    line[line_len++] = '\n';
    htif_write(line, (size_t)line_len);
    line_len = 0;
}

#endif
