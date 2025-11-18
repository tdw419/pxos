; guest_real.asm - Simple real mode guest for pxHV testing
; This guest runs in 16-bit real mode and writes "A" to VGA memory

BITS 16
ORG 0x7C00

start:
    ; Clear direction flag
    cld

    ; Setup segments for real mode
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00          ; Stack grows down from code

    ; Write "A" to VGA text mode memory at 0xB8000
    ; In real mode, we access this via segment:offset
    mov ax, 0xB800          ; VGA text mode segment
    mov es, ax
    mov di, 0               ; Offset 0 (top-left corner)

    mov al, 'A'             ; Character to write
    mov ah, 0x0F            ; White on black attribute
    mov [es:di], ax         ; Write char + attribute

    ; Halt the guest
    hlt

    ; In case of spurious wake, keep halting
.halt_loop:
    hlt
    jmp .halt_loop

; Boot sector signature (for completeness)
times 510-($-$$) db 0
dw 0xAA55
