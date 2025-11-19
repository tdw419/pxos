; Test bootloader with serial port output
BITS 16
ORG 0x7C00

SERIAL_PORT equ 0x3F8

start:
    ; Initialize serial port
    mov dx, SERIAL_PORT + 1
    mov al, 0x00
    out dx, al              ; Disable interrupts

    mov dx, SERIAL_PORT + 3
    mov al, 0x80
    out dx, al              ; Enable DLAB

    mov dx, SERIAL_PORT
    mov al, 0x03
    out dx, al              ; Divisor low (38400 baud)

    mov dx, SERIAL_PORT + 1
    mov al, 0x00
    out dx, al              ; Divisor high

    mov dx, SERIAL_PORT + 3
    mov al, 0x03
    out dx, al              ; 8 bits, no parity, 1 stop

    mov dx, SERIAL_PORT + 2
    mov al, 0xC7
    out dx, al              ; Enable FIFO

    mov dx, SERIAL_PORT + 4
    mov al, 0x0B
    out dx, al              ; IRQs enabled, RTS/DSR set

    ; Print message to serial
    mov si, msg
.loop:
    lodsb
    test al, al
    jz .done
    call write_serial
    jmp .loop

.done:
    cli
    hlt

write_serial:
    push ax
    push dx

    mov ah, al              ; Save character in AH

    mov dx, SERIAL_PORT + 5
.wait:
    in al, dx
    test al, 0x20
    jz .wait

    mov al, ah              ; Restore character
    mov dx, SERIAL_PORT
    out dx, al

    pop dx
    pop ax
    ret

msg: db 'pxOS Custom Bootloader Test - Serial Output Working!', 13, 10, 0

times 510-($-$$) db 0
dw 0xAA55
