; pxHV Boot Sector - Stage 1
; Loads hypervisor from disk and transfers control
BITS 16
ORG 0x7C00

STAGE2_SEGMENT equ 0x1000
STAGE2_OFFSET equ 0x0000
STAGE2_SECTORS equ 40

start:
    cli
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00
    sti

    mov [boot_drive], dl

    mov ah, 0x02
    mov al, STAGE2_SECTORS
    mov ch, 0
    mov cl, 2
    mov dh, 0
    mov dl, [boot_drive]
    mov bx, STAGE2_SEGMENT
    mov es, bx
    mov bx, STAGE2_OFFSET
    int 0x13
    jc disk_error

    jmp STAGE2_SEGMENT:STAGE2_OFFSET

disk_error:
    mov si, msg_error
    call print_string
    cli
    hlt

print_string:
    lodsb
    test al, al
    jz .done
    mov ah, 0x0E
    int 0x10
    jmp print_string
.done:
    ret

boot_drive  db 0
msg_error   db 'Disk read error!', 13, 10, 0

times 510-($-$$) db 0
dw 0xAA55
