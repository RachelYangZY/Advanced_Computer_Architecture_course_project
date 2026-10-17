#include <stdio.h>
#include <stdlib.h>

#include "requantize.h"

static void expect_eq(int8_t got, int8_t want, const char *label)
{
    if (got != want) {
        fprintf(stderr, "%s: got %d want %d\n", label, got, want);
        exit(1);
    }
}

int main(void)
{
    expect_eq(requantize_i32_to_i8(0, 1, 0, 0), 0, "zero");
    expect_eq(requantize_i32_to_i8(100, 1, 1, 0), 50, "round_pos");
    expect_eq(requantize_i32_to_i8(-100, 1, 1, 0), -50, "round_neg");
    expect_eq(requantize_i32_to_i8(1000, 1, 0, 7), 127, "sat_hi");
    expect_eq(requantize_i32_to_i8(-1000, 1, 0, 0), -128, "sat_lo");
    expect_eq(requantize_i32_to_i8(10, 3, 1, 2), 17, "mul_shift_zp");

    int32_t acc[4] = {0, 100, -100, 1000};
    int8_t out[4];
    requantize_i32_to_i8_vector(acc, out, 4, 1, 1, 0);
    expect_eq(out[0], 0, "vec0");
    expect_eq(out[1], 50, "vec1");
    expect_eq(out[2], -50, "vec2");
    expect_eq(out[3], 127, "vec3");

    puts("requantize reference test passed");
    return 0;
}
