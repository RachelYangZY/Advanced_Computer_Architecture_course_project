/* Linker script for generated models: like b0.ld, plus small-data sections and a 64 KiB stack. */
ENTRY(_start)

SECTIONS
{
    . = 0x80000000;

    .text : {
        *(.text.init)
        *(.text*)
    }

    .rodata : {
        *(.rodata*)
        *(.srodata*)
    }

    .data : {
        *(.data*)
        *(.sdata*)
    }

    .tohost : ALIGN(64) {
        *(.tohost)
    }

    .bss : {
        *(.bss*)
        *(.sbss*)
        *(COMMON)
        . = ALIGN(16);
        . += 65536;
        __stack_top = .;
    }
}
