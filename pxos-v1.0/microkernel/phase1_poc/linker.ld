/* Linker script for pxOS multiboot kernel */

ENTRY(_start)

SECTIONS
{
    /* Kernel loaded at 1MB */
    . = 1M;

    /* Multiboot header must be early */
    .multiboot ALIGN(8) : {
        *(.multiboot)
    }

    /* Code sections */
    .text ALIGN(4K) : {
        *(.text)
    }

    .text64 ALIGN(4K) : {
        *(.text64)
    }

    /* Read-only data */
    .rodata ALIGN(4K) : {
        *(.rodata)
    }

    /* Initialized data */
    .data ALIGN(4K) : {
        *(.data)
    }

    /* Uninitialized data */
    .bss ALIGN(4K) : {
        *(.bss)
        *(COMMON)
    }
}
