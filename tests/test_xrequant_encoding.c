#include <stdint.h>
#include <stdio.h>

#include "xrequant_encoding.h"

int main(void)
{
    uint32_t encoding = xrequant_encode(10, 11, 12);
    uint32_t expected = UINT32_C(0x56c5950b);

    if (encoding != expected) {
        fprintf(stderr, "encoding mismatch: got 0x%08x, expected 0x%08x\n",
                encoding, expected);
        return 1;
    }

    uint64_t ctrl = xrequant_pack_ctrl(31, 7);
    if (ctrl != UINT64_C(0x1f07)) {
        fprintf(stderr, "ctrl pack mismatch: got 0x%llx\n",
                (unsigned long long)ctrl);
        return 1;
    }

    printf("Xrequant encoding test passed: 0x%08x\n", encoding);
    return 0;
}
