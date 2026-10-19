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
 *   ctrl = rd: [7:0] zero point, [15:8] shift, [17:16] mode
 *   mode 0: legacy single rounding (acc*mult + 2^(s-1)) >> s
 *   mode 1/2: TFLite two-step rounding (high multiply, then half-up / half-away shift)
 *   rd   = sext8(sat_int8(result + zero_point))
 * Full definition and oracle: xrequant_ref.h.
 */
static inline uint32_t xrequant_encode(unsigned rd, unsigned rs1, unsigned rs2)
{
    return (XREQUANT_FUNCT7 << 25) | (rs2 << 20) | (rs1 << 15) |
           (XREQUANT_FUNCT3 << 12) | (rd << 7) | XREQUANT_OPCODE;
}

/* Legacy (mode 0, single rounding) control word. */
static inline uint64_t xrequant_pack_ctrl(int shift, int32_t zero_point)
{
    return ((uint64_t)(shift & 0xff) << 8) | ((uint64_t)(zero_point & 0xff));
}

/* Control word with a mode (bits 17:16): 1 = TFLite half-up, 2 = TFLite half-away. See xrequant_ref.h. */
static inline uint64_t xrequant_pack_ctrl_mode(int shift, int32_t zero_point, int mode)
{
    return xrequant_pack_ctrl(shift, zero_point) | ((uint64_t)(mode & 0x3) << 16);
}

#endif
