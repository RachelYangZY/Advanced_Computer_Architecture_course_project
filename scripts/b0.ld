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
    }

    .data : {
        *(.data*)
    }

    .tohost : ALIGN(64) {
        *(.tohost)
    }

    .bss : {
        *(.bss*)
        *(COMMON)
        . = ALIGN(16);
        . += 4096;
        __stack_top = .;
    }
}