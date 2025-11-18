; guest_real.asm - Real mode guest with I/O port test for pxHV
; Tests I/O VM exits by outputting to QEMU debug port 0xE9

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

    ; Test I/O exits by writing to port 0xE9 (QEMU debug port)
    ; This will cause VM exit (exit reason 30)
    mov si, msg_hello
    call print_io

    ; Also write to VGA memory for visual confirmation
    mov ax, 0xB800          ; VGA text mode segment
    mov es, ax
    mov di, 0               ; Offset 0 (top-left corner)
    mov al, 'O'             ; Character to write
    mov ah, 0x0F            ; White on black attribute
    mov [es:di], ax         ; Write char + attribute

    ; Halt the guest
    hlt

    ; In case of spurious wake, keep halting
.halt_loop:
    hlt
    jmp .halt_loop

;-----------------------------------------------------------------------------
; print_io: Write string to port 0xE9 (QEMU debug console)
; Input: SI = pointer to null-terminated string
;-----------------------------------------------------------------------------
print_io:
    push ax
    push dx
    push si

    mov dx, 0xE9            ; QEMU debug port

.loop:
    lodsb                   ; Load byte from [DS:SI] into AL
    test al, al             ; Check for null terminator
    jz .done
    out dx, al              ; Write to port 0xE9 (causes VM exit)
    jmp .loop

.done:
    pop si
    pop dx
    pop ax
    ret

;-----------------------------------------------------------------------------
; Data section
;-----------------------------------------------------------------------------
msg_hello: db 'Hello from guest!', 13, 10, 0

; Boot sector signature (for completeness)
times 510-($-$$) db 0
dw 0xAA55
