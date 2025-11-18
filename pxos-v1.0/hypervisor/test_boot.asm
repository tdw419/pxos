; test_boot.asm - Simple bootloader to test pxHV BIOS emulation
; Tests INT 10h, INT 13h, and INT 16h
; This will be loaded by INT 19h from the virtual disk

BITS 16
ORG 0x7C00

start:
    ; Setup segments
    cli
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00          ; Stack grows down from boot sector
    sti

    ; Clear screen using INT 10h
    mov ah, 0x00
    mov al, 0x03            ; 80x25 text mode
    int 0x10

    ; Print boot message
    mov si, msg_boot
    call print_string

    ; Test INT 13h - Get drive parameters
    mov si, msg_disk_test
    call print_string

    mov ah, 0x08            ; Get drive parameters
    mov dl, 0x00            ; Drive A:
    int 0x13
    jc .disk_error

    ; Print success message
    mov si, msg_disk_ok
    call print_string

    ; Skip geometry display to save space
    ; mov si, msg_newline
    ; call print_string

    ; Test INT 13h - Read sector 1 to memory at 0x8000
    mov si, msg_read_test
    call print_string

    mov ah, 0x02            ; Read sectors
    mov al, 0x01            ; 1 sector
    mov ch, 0x00            ; Cylinder 0
    mov cl, 0x02            ; Sector 2 (1-based)
    mov dh, 0x00            ; Head 0
    mov dl, 0x00            ; Drive A:
    mov bx, 0x0800          ; ES:BX = 0x0000:0x0800
    int 0x13
    jc .disk_error

    mov si, msg_read_ok
    call print_string

    ; Print prompt and wait
    mov si, msg_prompt
    call print_string

    ; Wait for keypress using INT 16h
    mov ah, 0x00            ; Read keystroke
    int 0x16                ; Returns scan code in AH, ASCII in AL

    ; Print the key
    call print_char
    mov si, msg_newline
    call print_string

    ; Success - halt
    mov si, msg_success
    call print_string

.halt:
    hlt
    jmp .halt

.disk_error:
    mov si, msg_disk_error
    call print_string
    jmp .halt

;-----------------------------------------------------------------------------
; print_string: Print null-terminated string using INT 10h
; Input: SI = pointer to string
;-----------------------------------------------------------------------------
print_string:
    pusha
    mov ah, 0x0E            ; Teletype output
.loop:
    lodsb
    test al, al
    jz .done
    int 0x10
    jmp .loop
.done:
    popa
    ret

;-----------------------------------------------------------------------------
; print_char: Print character in AL using INT 10h
;-----------------------------------------------------------------------------
print_char:
    pusha
    mov ah, 0x0E
    int 0x10
    popa
    ret

; Removed print_hex_byte to save space

;-----------------------------------------------------------------------------
; Data section
;-----------------------------------------------------------------------------
msg_boot:       db 13, 10
                db '====================================', 13, 10
                db '  pxHV Stage 5: BIOS Boot Test', 13, 10
                db '====================================', 13, 10, 13, 10, 0

msg_disk_test:  db 'INT 13h Get Params...', 0
msg_disk_ok:    db ' OK', 13, 10, 0
msg_disk_error: db ' ERR!', 13, 10, 0
msg_read_test:  db 'INT 13h Read...', 0
msg_read_ok:    db ' OK', 13, 10, 0
msg_newline:    db 13, 10, 0
msg_prompt:     db 13, 10, 'Press key: ', 0
msg_success:    db 13, 10, 'Tests OK! Ready for FreeDOS.', 13, 10, 0

;-----------------------------------------------------------------------------
; Padding and boot signature
;-----------------------------------------------------------------------------
times 510-($-$$) db 0       ; Pad to 510 bytes
dw 0xAA55                   ; Boot signature
