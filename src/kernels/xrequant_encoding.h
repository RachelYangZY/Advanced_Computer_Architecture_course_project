#ifndef XREQUANT_ENCODING_H
#define XREQUANT_ENCODING_H

#include <stdint.h>

/* Custom-0 opcode family shared with Xqmac8; distinct funct3. */
#define XREQUANT_OPCODE 0x0bU
#define XREQUANT_FUNCT3 0x1U
#define XREQUANT_FUNCT7 0x2bU

/*
 * xrequant rd, rs1, rs2        (two sources, one destination; rd is write-only)
 *
 *   rs1 = accumulator, sext32(rs1[31:0])
 *   rs2 = packed operand:
 *           rs2[31:0]   multiplier (signed Q31)
 *           rs2[39:32]  zero point (signed)
 *           rs2[47:40]  shift field
 *           rs2[49:48]  mode
 *   mode 0: legacy single rounding (acc*mult + 2^(s-1)) >> s, s = unsigned shift field 0..63
 *   mode 1/2: TFLite two-step rounding (high multiply, then half-up / half-away shift); the shift
 *             field is TFLite's signed shift: > 0 shifts the accumulator left first, <= 0 shifts the
 *             high-multiply result right by -shift (both 0..31)
 *   rd   = sext8(sat_int8(result + zero_point))
 * Full definition and oracle: xrequant_ref.h.
 */
static inline uint32_t xrequant_encode(unsigned rd, unsigned rs1, unsigned rs2)
{
    return (XREQUANT_FUNCT7 << 25) | (rs2 << 20) | (rs1 << 15) |
           (XREQUANT_FUNCT3 << 12) | (rd << 7) | XREQUANT_OPCODE;
}

/* Upper half of rs2: [7:0] zero point, [15:8] shift field, [17:16] mode (so ctrl << 32 in rs2). */
static inline uint64_t xrequant_pack_ctrl(int shift, int32_t zero_point)
{
    return ((uint64_t)(shift & 0xff) << 8) | ((uint64_t)(zero_point & 0xff));
}

static inline uint64_t xrequant_pack_ctrl_mode(int shift, int32_t zero_point, int mode)
{
    return xrequant_pack_ctrl(shift, zero_point) | ((uint64_t)(mode & 0x3) << 16);
}

/* Full rs2 operand from a multiplier and a control word. */
static inline uint64_t xrequant_pack_rs2(int32_t multiplier, uint64_t ctrl)
{
    return (ctrl << 32) | (uint64_t)(uint32_t)multiplier;
}

/* Convenience: rs2 for TFLite modes (1 = half-up FC/CONV, 2 = half-away BMM); shift is TFLite's signed shift. */
static inline uint64_t xrequant_pack(int32_t multiplier, int shift, int32_t zero_point, int mode)
{
    return xrequant_pack_rs2(multiplier, xrequant_pack_ctrl_mode(shift, zero_point, mode));
}

#endif
