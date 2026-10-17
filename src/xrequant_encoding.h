#ifndef XREQUANT_ENCODING_H
#define XREQUANT_ENCODING_H

#include <stdint.h>

/* Custom-0 opcode family shared with Xqmac8; distinct funct3. */
#define XREQUANT_OPCODE 0x0bU
#define XREQUANT_FUNCT3 0x1U
#define XREQUANT_FUNCT7 0x2bU

/*
 * xrequant rd, rs1, rs2
 *
 *   ctrl = rd
 *     ctrl[7:0]   = signed zero-point
 *     ctrl[15:8]  = unsigned shift amount
 *   acc  = sext32(rs1)
 *   mult = sext32(rs2)
 *   tmp  = (acc * mult + (1 << (shift-1))) >> shift   (if shift > 0)
 *   rd   = sext8(sat_int8(tmp + zero_point))
 */
static inline uint32_t xrequant_encode(unsigned rd, unsigned rs1, unsigned rs2)
{
    return (XREQUANT_FUNCT7 << 25) | (rs2 << 20) | (rs1 << 15) |
           (XREQUANT_FUNCT3 << 12) | (rd << 7) | XREQUANT_OPCODE;
}

static inline uint64_t xrequant_pack_ctrl(int shift, int32_t zero_point)
{
    return ((uint64_t)(shift & 0xff) << 8) | ((uint64_t)(zero_point & 0xff));
}

#endif
