#include <stdint.h>
#include <stdio.h>

#include "xqmac8_encoding.h"

int main(void)
{
    uint32_t encoding = xqmac8_encode(10, 11, 12);
    uint32_t expected = UINT32_C(0x56c5850b);

    if (encoding != expected) {
        fprintf(stderr, "encoding mismatch: got 0x%08x, expected 0x%08x\n",
                encoding, expected);
        return 1;
    }

    printf("Xqmac8 encoding test passed: 0x%08x\n", encoding);
    return 0;
}