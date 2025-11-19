; pxOS Bootloader Utility Functions
; Shared functions for stage 2 bootloader

BITS 16

SERIAL_PORT equ 0x3F8

;-----------------------------------------------------------------------------
; print16 - Print null-terminated string (16-bit real mode)
; Input: SI = pointer to string
; CRITICAL: Uses serial port with AH-save fix from kernel debugging
;-----------------------------------------------------------------------------
print16:
    pusha
.loop:
    lodsb
    test al, al
    jz .done
    call write_serial
    jmp .loop
.done:
    popa
    ret

;-----------------------------------------------------------------------------
; write_serial - Write character to serial port
; Input: AL = character to write
; CRITICAL FIX: Save character in AH before status check!
;-----------------------------------------------------------------------------
write_serial:
    push ax
    push dx

    mov ah, al              ; CRITICAL: Save character in AH before status read

    ; Wait for transmit buffer to be empty
    mov dx, SERIAL_PORT + 5
.wait:
    in al, dx               ; Read line status (OVERWRITES AL!)
    test al, 0x20           ; Check if transmit buffer empty
    jz .wait

    mov al, ah              ; Restore character from AH
    mov dx, SERIAL_PORT
    out dx, al              ; Send character

    pop dx
    pop ax
    ret

;-----------------------------------------------------------------------------
; print_hex16 - Print 16-bit hex value
; Input: AX = value to print
;-----------------------------------------------------------------------------
print_hex16:
    pusha
    mov cx, 4               ; 4 hex digits
.digit:
    rol ax, 4               ; Rotate next nibble into low 4 bits
    push ax
    and al, 0x0F
    add al, '0'
    cmp al, '9'
    jle .print
    add al, 7               ; 'A'-'9'-1
.print:
    call write_serial
    pop ax
    loop .digit
    popa
    ret

;-----------------------------------------------------------------------------
; print_hex32 - Print 32-bit hex value
; Input: EAX = value to print
;-----------------------------------------------------------------------------
print_hex32:
    pusha

    ; Print "0x" prefix
    push eax
    mov al, '0'
    call write_serial
    mov al, 'x'
    call write_serial
    pop eax

    ; Print 8 hex digits
    mov cx, 8
.digit:
    rol eax, 4
    push eax
    and al, 0x0F
    add al, '0'
    cmp al, '9'
    jle .print
    add al, 7
.print:
    call write_serial
    pop eax
    loop .digit
    popa
    ret

;-----------------------------------------------------------------------------
; newline - Print CR+LF
;-----------------------------------------------------------------------------
newline:
    pusha
    mov al, 13              ; Carriage return
    call write_serial
    mov al, 10              ; Line feed
    call write_serial
    popa
    ret

;-----------------------------------------------------------------------------
; check_cpuid - Check if CPUID instruction is supported
; Output: CF clear if supported, set if not supported
;-----------------------------------------------------------------------------
check_cpuid:
    pushfd
    pop eax
    mov ecx, eax            ; Save original FLAGS

    xor eax, 0x00200000     ; Flip ID bit (bit 21)
    push eax
    popfd

    pushfd
    pop eax

    xor eax, ecx            ; Check if bit changed
    jz .no_cpuid

    clc                     ; CPUID supported
    ret

.no_cpuid:
    stc                     ; CPUID not supported
    ret

;-----------------------------------------------------------------------------
; check_long_mode - Check if CPU supports 64-bit long mode
; Output: CF clear if supported, set if not supported
;-----------------------------------------------------------------------------
check_long_mode:
    ; First check if CPUID is available
    call check_cpuid
    jc .no_long_mode

    ; Check if extended CPUID functions are available
    mov eax, 0x80000000
    cpuid
    cmp eax, 0x80000001
    jb .no_long_mode

    ; Check for long mode support (bit 29 of EDX)
    mov eax, 0x80000001
    cpuid
    test edx, 1 << 29
    jz .no_long_mode

    clc                     ; Long mode supported
    ret

.no_long_mode:
    stc                     ; Long mode not supported
    ret

;-----------------------------------------------------------------------------
; enable_a20 - Enable A20 line (allows access to memory above 1MB)
;-----------------------------------------------------------------------------
enable_a20:
    pusha

    ; Try BIOS method first
    mov ax, 0x2401
    int 0x15
    jnc .done

    ; Try keyboard controller method
    call .wait_input
    mov al, 0xAD            ; Disable keyboard
    out 0x64, al

    call .wait_input
    mov al, 0xD0            ; Read output port
    out 0x64, al

    call .wait_output
    in al, 0x60
    push ax

    call .wait_input
    mov al, 0xD1            ; Write output port
    out 0x64, al

    call .wait_input
    pop ax
    or al, 2                ; Set A20 bit
    out 0x60, al

    call .wait_input
    mov al, 0xAE            ; Enable keyboard
    out 0x64, al

    call .wait_input

.done:
    popa
    ret

.wait_input:
    in al, 0x64
    test al, 2
    jnz .wait_input
    ret

.wait_output:
    in al, 0x64
    test al, 1
    jz .wait_output
    ret

;-----------------------------------------------------------------------------
; Data
;-----------------------------------------------------------------------------
hex_chars: db '0123456789ABCDEF'
