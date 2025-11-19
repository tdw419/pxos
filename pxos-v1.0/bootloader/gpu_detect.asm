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
    mov byte [.current_dev], 0

.scan_device:
    ; Read vendor ID using PCI BIOS
    ; BH = bus, BL = device<<3 | function, DI = register
    mov ax, 0xB109          ; Read config word
    mov bh, 0               ; Bus 0
    mov bl, [.current_dev]  ; Device number
    shl bl, 3               ; Shift to bits 7-3
    mov di, 0               ; Register 0 (vendor ID)
    int 0x1A
    jc .next_device         ; Skip if error

    ; Check if device exists (vendor ID != 0xFFFF)
    cmp cx, 0xFFFF
    je .next_device

    ; DEBUG: Print device found
    push bx
    push cx
    mov si, msg_dev_found
    call print16
    mov al, [.current_dev]
    xor ah, ah
    call print_hex16
    mov si, msg_vendor_debug
    call print16
    pop cx
    push cx
    mov ax, cx
    call print_hex16
    call newline
    pop cx
    pop bx

    ; Device exists - read class/subclass word
    ; Offset 0x0A contains: subclass (low byte) + base class (high byte)
    mov ax, 0xB109
    mov bh, 0               ; Bus 0
    mov bl, [.current_dev]
    shl bl, 3               ; Device << 3
    mov di, 0x0A            ; Register 0x0A (subclass + class)
    int 0x1A
    jc .next_device

    ; DEBUG: Print class code (CL = subclass, CH = base class)
    push bx
    push cx
    mov si, msg_class
    call print16
    mov al, ch              ; Base class in CH
    xor ah, ah
    call print_hex16
    mov al, cl              ; Subclass in CL
    xor ah, ah
    call print_hex16
    call newline
    pop cx
    pop bx

    ; Check if base class is 0x03 (Display controller)
    cmp ch, 0x03
    jne .next_device

    ; Found a VGA device!
    jmp .found_gpu

.next_device:
    inc byte [.current_dev]  ; Next device
    cmp byte [.current_dev], 32  ; Scanned all 32 devices?
    jb .scan_device

    ; No GPU found
    mov si, msg_no_gpu
    call print16
    popa
    stc
    ret

.current_dev: db 0

.found_gpu:
    mov si, msg_found
    call print16

    ; Save device location
    mov al, [.current_dev]
    mov [gpu_dev], al
    mov byte [gpu_bus], 0

    ; Read vendor ID
    mov ax, 0xB109
    mov bh, 0
    mov bl, [.current_dev]
    shl bl, 3
    mov di, 0               ; Register 0 (vendor ID)
    int 0x1A
    mov [gpu_vendor], cx

    ; Read device ID
    mov ax, 0xB109
    mov bh, 0
    mov bl, [.current_dev]
    shl bl, 3
    mov di, 2               ; Register 2 (device ID)
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
    mov bh, 0               ; Bus 0
    mov bl, [.current_dev]
    shl bl, 3               ; Device << 3
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
msg_scanning:    db "Scanning for GPU... ", 13, 10, 0
msg_dev_found:   db "  Dev ", 0
msg_vendor_debug: db " vendor=", 0
msg_class:       db "    class=", 0
msg_found:       db "Found GPU!", 13, 10, 0
msg_no_gpu:      db "No GPU found!", 13, 10, 0
msg_vendor:      db "  Vendor: 0x", 0
msg_device:      db " Device: 0x", 0
msg_bar0:        db "  BAR0: ", 0
msg_bar0_error:  db "ERROR: Could not read BAR0!", 13, 10, 0
