#ifndef XQMAC8_ENCODING_H
#define XQMAC8_ENCODING_H

#include <stdint.h>

#define XQMAC8_OPCODE 0x0bU
#define XQMAC8_FUNCT3 0x0U  
#define XQMAC8_FUNCT7 0x2bU

static inline uint32_t xqmac8_encode(unsigned rd, unsigned rs1, unsigned rs2)
{
    return (XQMAC8_FUNCT7 << 25) | (rs2 << 20) | (rs1 << 15) |
           (XQMAC8_FUNCT3 << 12) | (rd << 7) | XQMAC8_OPCODE;  //0x56c5850b
}

#endif