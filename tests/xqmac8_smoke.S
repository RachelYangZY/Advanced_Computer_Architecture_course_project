.section .text.init
.global _start
.type _start, @function
_start:
    la sp, __stack_top
    li a0, 5
    li a1, 0x0101010101010101
    li a2, 0x0202020202020202
    .word 0x56c5850b

    li a3, 21
    bne a0, a3, fail
    li a3, 1
    j report

fail:
    li a3, 3

report:
    la a4, tohost
    sd a3, 0(a4)
1:
    j 1b

.section .tohost
.align 6
.global tohost
.global fromhost
tohost:
    .dword 0
fromhost:
    .dword 0