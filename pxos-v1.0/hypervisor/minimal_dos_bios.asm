; minimal_dos_bios.asm - pxDOS using BIOS INT services
; Stage 4c: Uses INT 10h for video output instead of direct I/O

BITS 16
ORG 0x7C00

start:
    ; Setup segments for real mode
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00          ; Stack grows down

    ; Clear screen using INT 10h, AH=00h (Set Video Mode)
    mov ah, 0x00
    mov al, 0x03            ; 80x25 text mode
    int 0x10

    ; Print separator using INT 10h teletype
    mov si, msg_separator
    call print_string_bios

    ; Print OS name
    mov si, msg_os_name
    call print_string_bios

    ; Print version
    mov si, msg_version
    call print_string_bios

    ; Print copyright
    mov si, msg_copyright
    call print_string_bios

    ; Print empty line
    mov si, msg_newline
    call print_string_bios

    ; Print command prompt
    mov si, msg_prompt
    call print_string_bios

    ; For now, halt after displaying prompt
    ; Future: Implement command loop with INT 16h for keyboard
    hlt

    ; If we somehow return, loop forever
    jmp $

;-----------------------------------------------------------------------------
; print_string_bios: Print null-terminated string using BIOS INT 10h
; Input: SI = pointer to string
; Uses INT 10h, AH=0Eh (Teletype output)
;-----------------------------------------------------------------------------
print_string_bios:
    pusha
    mov ah, 0x0E            ; BIOS teletype function
    mov bx, 0x0007          ; Page 0, white on black attribute

.loop:
    lodsb                   ; Load byte from [DS:SI] into AL, increment SI
    test al, al             ; Check for null terminator
    jz .done
    int 0x10                ; Call BIOS - triggers VM exit (reason 0)
    jmp .loop               ; Next character

.done:
    popa
    ret

;-----------------------------------------------------------------------------
; Data section
;-----------------------------------------------------------------------------
msg_separator:  db '='
                times 60 db '='
                db 13, 10, 0

msg_os_name:    db 13, 10
                db '  pxDOS - Pixel Disk Operating System', 13, 10, 0

msg_version:    db '  Version 0.2 (BIOS Services Edition)', 13, 10, 0

msg_copyright:  db '  Copyright (C) 2025 pxOS Project', 13, 10, 0

msg_newline:    db 13, 10, 0

msg_prompt:     db 'A:\\> ', 0

;-----------------------------------------------------------------------------
; Boot sector signature
;-----------------------------------------------------------------------------
times 510-($-$$) db 0       ; Pad to 510 bytes
dw 0xAA55                   ; Boot signature (0x55AA little-endian)
