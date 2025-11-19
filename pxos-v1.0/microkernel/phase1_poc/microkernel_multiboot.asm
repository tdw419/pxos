; microkernel_multiboot.asm - A minimal Multiboot2-compliant kernel
BITS 32
GLOBAL start
EXTERN kernel_main

; Multiboot2 header
SECTION .multiboot
align 8

MB2_MAGIC      equ 0xE85250D6
MB2_ARCH_I386  equ 0
MB2_HEADER_LEN equ mb2_header_end - mb2_header
MB2_CHECKSUM   equ -(MB2_MAGIC + MB2_ARCH_I386 + MB2_HEADER_LEN)

mb2_header:
    dd MB2_MAGIC
    dd MB2_ARCH_I386
    dd MB2_HEADER_LEN
    dd MB2_CHECKSUM

    ; end tag
    dw 0
    dw 0
    dd 8
mb2_header_end:

SECTION .text
align 16

start:
    mov esp, stack_top

    ; Write "GRUB OK" to VGA
    mov dword [0xB8000], 0x0F4B0F4F
    mov dword [0xB8004], 0x0F200F42
    mov dword [0xB8008], 0x0F520F55
    mov dword [0xB800C], 0x0F47

.hang:
    cli
    hlt
    jmp .hang

SECTION .bss
align 16
stack_bottom:
    resb 4096
stack_top:
