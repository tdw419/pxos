; minimal_dos.asm - Simplest bootable "OS" for pxHV
; Prints OS name and prompt via I/O port 0xE9
; This validates the entire hypervisor stack works

BITS 16
ORG 0x7C00

start:
    ; Setup segments for real mode
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00          ; Stack grows down

    ; Clear screen marker (optional, just a visual separator)
    mov si, msg_separator
    call print_io

    ; Print OS name
    mov si, msg_os_name
    call print_io

    ; Print version
    mov si, msg_version
    call print_io

    ; Print copyright
    mov si, msg_copyright
    call print_io

    ; Print empty line
    mov si, msg_newline
    call print_io

    ; Print command prompt
    mov si, msg_prompt
    call print_io

    ; For now, just halt
    ; Future: implement command loop with INT 16h for keyboard
    hlt

    ; If we somehow return, loop forever
    jmp $

;-----------------------------------------------------------------------------
; print_io: Print null-terminated string via port 0xE9
; Input: SI = pointer to string
; Clobbers: AX, DX, SI (but we save/restore in pusha/popa)
;-----------------------------------------------------------------------------
print_io:
    pusha
    mov dx, 0xE9            ; QEMU/pxHV debug port
.loop:
    lodsb                   ; Load byte from [DS:SI] into AL, increment SI
    test al, al             ; Check for null terminator
    jz .done
    out dx, al              ; Output byte - triggers VM exit (reason 30)
    jmp .loop               ; Next character
.done:
    popa
    ret

;-----------------------------------------------------------------------------
; Data section
;-----------------------------------------------------------------------------
msg_separator:  db 13, 10, '='
                times 60 db '='
                db 13, 10, 0

msg_os_name:    db 13, 10
                db '  pxDOS - Pixel Disk Operating System', 13, 10, 0

msg_version:    db '  Version 0.1 (pxHV Hypervisor Build)', 13, 10, 0

msg_copyright:  db '  Copyright (C) 2025 pxOS Project', 13, 10, 0

msg_newline:    db 13, 10, 0

msg_prompt:     db 'A:\\> ', 0

;-----------------------------------------------------------------------------
; Boot sector signature
;-----------------------------------------------------------------------------
times 510-($-$$) db 0       ; Pad to 510 bytes
dw 0xAA55                   ; Boot signature (0x55AA little-endian)
