#ifndef XQMAC8_H
#define XQMAC8_H

#include <stdint.h>

/* Oracle for `xqmac8 rd, rs1, rs2`: rd = sum over 8 int8 lanes of rs1[i] * rs2[i].
 * rd is written only; the caller accumulates (acc += xqmac8(a, b)). */
int64_t xqmac8_reference(uint64_t packed_a, uint64_t packed_b);

#endif
