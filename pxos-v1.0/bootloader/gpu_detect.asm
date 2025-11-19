; pxOS Bootloader - GPU Detection
; Scans PCI bus for VGA/GPU device and reads BAR0

BITS 16

;-----------------------------------------------------------------------------
; detect_gpu - Scan PCI bus for VGA-compatible GPU
; Output:
;   CF clear if GPU found
;   [gpu_vendor] = vendor ID
;   [gpu_device] = device ID
;   [gpu_bus] = bus number
;   [gpu_dev] = device number
;   [gpu_bar0] = BAR0 base address (physical)
;-----------------------------------------------------------------------------
detect_gpu:
    pusha

    mov si, msg_scanning
    call print16

    ; Scan bus 0, devices 0-31
    xor bx, bx              ; BH = bus 0, BL = device 0

.scan_device:
    ; Read vendor ID using PCI BIOS
    mov ax, 0xB109          ; Read config word
    mov cx, 0               ; Register 0 (vendor ID)
    mov di, 0               ; Function 0
    int 0x1A
    jc .next_device         ; Skip if error

    ; Check if device exists (vendor ID != 0xFFFF)
    cmp cx, 0xFFFF
    je .next_device

    ; Device exists - read class code to check if it's VGA
    mov ax, 0xB109
    mov cx, 0x0A            ; Register 0x0A (class code high byte)
    int 0x1A
    jc .next_device

    ; Check if class code is 0x03 (Display controller)
    mov al, ch              ; Class code is in CH
    cmp al, 0x03
    jne .next_device

    ; Found a VGA device!
    jmp .found_gpu

.next_device:
    inc bl                  ; Next device
    cmp bl, 32              ; Scanned all 32 devices?
    jb .scan_device

    ; No GPU found
    mov si, msg_no_gpu
    call print16
    popa
    stc
    ret

.found_gpu:
    mov si, msg_found
    call print16

    ; Save device location
    mov [gpu_bus], bh
    mov [gpu_dev], bl

    ; Read vendor ID
    mov ax, 0xB109
    mov cx, 0
    int 0x1A
    mov [gpu_vendor], cx

    ; Read device ID
    mov ax, 0xB109
    mov cx, 2               ; Register 2 (device ID)
    int 0x1A
    mov [gpu_device], cx

    ; Print vendor:device
    mov si, msg_vendor
    call print16
    mov ax, [gpu_vendor]
    call print_hex16

    mov si, msg_device
    call print16
    mov ax, [gpu_device]
    call print_hex16
    call newline

    ; Read BAR0 (register 0x10)
    mov ax, 0xB10A          ; Read config dword
    mov bh, [gpu_bus]
    mov bl, [gpu_dev]
    mov di, 0x10            ; BAR0 register
    int 0x1A
    jc .bar0_error

    ; Mask out flag bits (bottom 4 bits)
    and ecx, 0xFFFFFFF0
    mov [gpu_bar0], ecx

    ; Print BAR0 address
    mov si, msg_bar0
    call print16
    mov eax, [gpu_bar0]
    call print_hex32
    call newline

    ; Check if BAR0 is valid (not 0)
    cmp dword [gpu_bar0], 0
    je .bar0_error

    popa
    clc                     ; Success
    ret

.bar0_error:
    mov si, msg_bar0_error
    call print16
    popa
    stc
    ret

;-----------------------------------------------------------------------------
; Data
;-----------------------------------------------------------------------------
; GPU information (set by detect_gpu)
gpu_vendor: dw 0
gpu_device: dw 0
gpu_bus:    db 0
gpu_dev:    db 0
gpu_bar0:   dd 0

; Messages
msg_scanning:    db "Scanning for GPU... ", 0
msg_found:       db "Found!", 13, 10, 0
msg_no_gpu:      db "No GPU found!", 13, 10, 0
msg_vendor:      db "  Vendor: 0x", 0
msg_device:      db " Device: 0x", 0
msg_bar0:        db "  BAR0: ", 0
msg_bar0_error:  db "ERROR: Could not read BAR0!", 13, 10, 0
