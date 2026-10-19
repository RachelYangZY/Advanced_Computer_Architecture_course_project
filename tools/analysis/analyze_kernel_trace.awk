BEGIN {
    start_value = strtonum("0x" start)
    end_value = strtonum("0x" end)
    total = 0
    mulw = 0
    loads = 0
    stores = 0
    branches = 0
    xqmac8 = 0
}

/^core/ {
    pc = $3
    gsub("0x", "", pc)
    pc_value = strtonum("0x" pc)
    if (pc_value < start_value || pc_value >= end_value) {
        next
    }

    total++
    mnemonic = $5
    if (mnemonic == ">>>>") {
        # Spike sometimes inserts disassembler mode markers; skip them.
        total--
        next
    }
    if (mnemonic == "mulw" || mnemonic == "mul") {
        mulw++
    }
    if (mnemonic ~ /^(c\.)?(lb|lbu|lh|lhu|lw|ld|lwu)$/) {
        loads++
    }
    if (mnemonic ~ /^(c\.)?(sb|sh|sw|sd)$/) {
        stores++
    }
    if (mnemonic ~ /^(c\.)?(beq|bne|blt|bge|bltu|bgeu|jal|jalr|j|jr)$/) {
        branches++
    }
    if (mnemonic == "xqmac8" || $0 ~ /0x56c5850b/) {
        xqmac8++
    }
}

END {
    printf("kernel_instructions=%d\n", total)
    printf("mulw=%d\n", mulw)
    printf("xqmac8=%d\n", xqmac8)
    printf("loads=%d\n", loads)
    printf("stores=%d\n", stores)
    printf("control_flow=%d\n", branches)
    if (total > 0) {
        printf("macs_per_instruction=%.6f\n", mac_count / total)
    } else {
        printf("macs_per_instruction=0\n")
    }
}
