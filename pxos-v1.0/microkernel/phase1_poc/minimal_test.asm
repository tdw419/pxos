; Absolute minimal multiboot2 kernel to test if GRUB loads it
BITS 32

section .multiboot
align 8
multiboot_start:
    dd 0xE85250D6                   ; Magic
    dd 0                            ; Architecture
    dd multiboot_end - multiboot_start
    dd 0x100000000 - (0xE85250D6 + 0 + (multiboot_end - multiboot_start))

    align 8
    dw 0    ; type
    dw 0    ; flags
    dd 8    ; size
multiboot_end:

section .text
global _start
_start:
    ; Write 'OK' to VGA immediately
    mov dword [0xB8000], 0x0F4B0F4F    ; 'OK' in white on black

    ; Write to serial port
    mov dx, 0x3F8       ; COM1
    mov al, 'T'
    out dx, al
    mov al, 'E'
    out dx, al
    mov al, 'S'
    out dx, al
    mov al, 'T'
    out dx, al
    mov al, 13          ; CR
    out dx, al
    mov al, 10          ; LF
    out dx, al

    ; Halt
    cli
.loop:
    hlt
    jmp .loop
