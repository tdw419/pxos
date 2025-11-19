; Minimal test bootloader - just print message and halt
BITS 16
ORG 0x7C00

start:
    ; Set up segments
    xor ax, ax
    mov ds, ax

    ; Print message
    mov si, msg
    call print

    ; Halt
    cli
    hlt

print:
    lodsb
    test al, al
    jz .done
    mov ah, 0x0E
    int 0x10
    jmp print
.done:
    ret

msg: db 'pxOS Test Boot!', 13, 10, 0

times 510-($-$$) db 0
dw 0xAA55
