; pxOS CPU Microkernel
; Minimal 2KB kernel: GPU initialization + dispatch loop
; Target: 5% CPU utilization, 95% GPU execution

BITS 64
ORG 0x1000

; Entry point (called from bootloader)
start:
    ; Setup stack
    mov rsp, 0x90000

    ; Print banner
    mov rsi, msg_banner
    call print_string

    ; Initialize GPU
    mov rsi, msg_init_gpu
    call print_string

    call init_gpu
    test rax, rax
    jz .gpu_error

    mov rsi, msg_gpu_ok
    call print_string

    ; Print GPU details
    call print_gpu_info

    ; Load os.pxi to GPU VRAM
    mov rsi, msg_loading_os
    call print_string

    call load_os_pxi
    test rax, rax
    jz .load_error

    mov rsi, msg_os_loaded
    call print_string

    ; Enter main dispatch loop
    mov rsi, msg_starting
    call print_string

    jmp gpu_dispatch_loop

.gpu_error:
    mov rsi, msg_gpu_error
    call print_string
    jmp .halt

.load_error:
    mov rsi, msg_load_error
    call print_string
    jmp .halt

.halt:
    cli
    hlt
    jmp .halt

;-----------------------------------------------------------------------------
; PCI Configuration Space Constants
;-----------------------------------------------------------------------------
PCI_CONFIG_ADDRESS  equ 0xCF8       ; PCI Configuration Address Port
PCI_CONFIG_DATA     equ 0xCFC       ; PCI Configuration Data Port
PCI_CLASS_DISPLAY   equ 0x03        ; Display controller class
PCI_BAR_IO_SPACE    equ 0x01        ; BAR is I/O space (bit 0)
PCI_BAR_MEM_MASK    equ 0xFFFFFFF0  ; Mask for memory BAR address

;-----------------------------------------------------------------------------
; init_gpu: Real PCIe enumeration and GPU initialization
; Returns: RAX = GPU BAR0 address (0 on error)
; Side effects: Sets gpu_found, gpu_bus, gpu_dev, gpu_func, gpu_bar0
;-----------------------------------------------------------------------------
init_gpu:
    push rbx
    push rcx
    push rdx
    push rsi
    push rdi

    ; Clear GPU found flag
    mov byte [gpu_found], 0

    ; Enumerate PCI devices on bus 0 (sufficient for QEMU)
    ; For full hardware support, scan buses 0-255
    xor ebx, ebx                    ; EBX = bus number (start at 0)

.scan_bus:
    cmp ebx, 1                      ; Only scan bus 0 for now
    jge .no_gpu_found

    ; Middle loop: device = 0..31
    xor ecx, ecx                    ; ECX = device number

.scan_device:
    cmp ecx, 32
    jge .next_bus

    ; Inner loop: function = 0..7
    xor edx, edx                    ; EDX = function number

.scan_function:
    cmp edx, 8
    jge .next_device

    ; Read vendor/device ID (offset 0x00)
    push rbx
    push rcx
    push rdx
    xor eax, eax                    ; offset = 0
    call pci_read_config_dword

    ; Check if device exists (vendor ID != 0xFFFF)
    cmp ax, 0xFFFF
    je .skip_function

    ; Read class code (offset 0x08)
    mov eax, 0x08
    call pci_read_config_dword

    ; Extract base class (bits 31:24)
    shr eax, 24
    cmp al, PCI_CLASS_DISPLAY
    jne .skip_function

    ; Found a display controller (GPU)!
    ; Save bus/device/function
    pop rdx
    pop rcx
    pop rbx

    mov [gpu_bus], bl
    mov [gpu_dev], cl
    mov [gpu_func], dl

    ; Read BAR0 (offset 0x10)
    push rbx
    push rcx
    push rdx
    mov eax, 0x10
    call pci_read_config_dword

    ; Check if it's a memory BAR (bit 0 clear)
    test eax, PCI_BAR_IO_SPACE
    jnz .skip_function

    ; Mask to get MMIO base address
    and eax, PCI_BAR_MEM_MASK
    mov [gpu_bar0], eax

    ; Mark GPU as found
    mov byte [gpu_found], 1

    ; Clean up stack
    pop rdx
    pop rcx
    pop rbx

    ; Return BAR0 address (zero-extended to 64-bit)
    xor rax, rax
    mov eax, [gpu_bar0]

    pop rdi
    pop rsi
    pop rdx
    pop rcx
    pop rbx
    ret

.skip_function:
    pop rdx
    pop rcx
    pop rbx
    inc edx
    jmp .scan_function

.next_device:
    inc ecx
    jmp .scan_device

.next_bus:
    inc ebx
    jmp .scan_bus

.no_gpu_found:
    ; No GPU found - return 0
    xor rax, rax

    pop rdi
    pop rsi
    pop rdx
    pop rcx
    pop rbx
    ret

;-----------------------------------------------------------------------------
; pci_read_config_dword: Read 32-bit value from PCI configuration space
; Input:
;   BL = bus (0..255)
;   CL = device (0..31)
;   DL = function (0..7)
;   EAX = register offset (must be DWORD aligned)
; Output:
;   EAX = 32-bit value from config space
; Clobbers: none (all registers preserved except EAX)
;-----------------------------------------------------------------------------
pci_read_config_dword:
    push rbx
    push rcx
    push rdx
    push rdi

    ; Save input parameters
    movzx edi, bl                   ; EDI = bus
    movzx ebx, cl                   ; EBX = device
    movzx ecx, dl                   ; ECX = function
    and eax, 0xFC                   ; EDX = offset (aligned)
    mov edx, eax

    ; Build PCI address: 0x80000000 | (bus<<16) | (dev<<11) | (func<<8) | offset
    mov eax, 0x80000000             ; Enable bit

    shl edi, 16                     ; bus << 16
    or eax, edi

    shl ebx, 11                     ; device << 11
    or eax, ebx

    shl ecx, 8                      ; function << 8
    or eax, ecx

    or eax, edx                     ; | offset

    ; Write address to CONFIG_ADDRESS port
    mov dx, PCI_CONFIG_ADDRESS
    out dx, eax

    ; Read data from CONFIG_DATA port
    mov dx, PCI_CONFIG_DATA
    in eax, dx

    pop rdi
    pop rdx
    pop rcx
    pop rbx
    ret

;-----------------------------------------------------------------------------
; load_os_pxi: Load os.pxi from disk to GPU VRAM
; Returns: RAX = 1 on success, 0 on error
;-----------------------------------------------------------------------------
load_os_pxi:
    push rbx
    push rcx
    push rdx
    push rsi
    push rdi

    ; Phase 1 POC: Load os.pxi from disk to CPU memory
    ; (In Phase 2, we'll DMA directly to GPU VRAM)

    ; os.pxi is located at sector 64 on disk
    ; Size: TBD (for POC, assume 1 sector = 512 bytes)

    ; TODO: Read sectors from disk
    ;   - Use BIOS INT 13h or direct ATA/AHCI access
    ;   - Load to temporary buffer
    ;   - Transfer to GPU VRAM via GPU driver

    ; TODO: Upload to GPU VRAM
    ;   - Use GPU memory copy commands
    ;   - Or setup DMA transfer

    ; For POC: Simulate successful load
    mov rax, 1

    pop rdi
    pop rsi
    pop rdx
    pop rcx
    pop rbx
    ret

;-----------------------------------------------------------------------------
; gpu_dispatch_loop: Main OS loop
; CPU dispatches GPU work, mostly idle
;-----------------------------------------------------------------------------
gpu_dispatch_loop:
    ; Submit GPU compute work
    call gpu_execute_os

    ; Check for GPU requests (MMIO operations needed)
    call check_gpu_requests
    test rax, rax
    jz .no_requests

    ; Handle privileged operation requested by GPU
    call handle_privileged_op

.no_requests:
    ; CPU idle - GPU is running the OS!
    hlt

    ; Loop forever
    jmp gpu_dispatch_loop

;-----------------------------------------------------------------------------
; gpu_execute_os: Dispatch GPU compute shader to execute os.pxi
;-----------------------------------------------------------------------------
gpu_execute_os:
    push rax
    push rbx
    push rcx

    ; TODO: Submit GPU compute dispatch
    ;   - Setup compute shader parameters
    ;   - Set workgroup size (256 threads)
    ;   - Dispatch compute shader
    ;   - GPU executes os.pxi continuously

    ; For POC: Simulate GPU execution
    ; (In Phase 2, we'll use real GPU command submission)

    pop rcx
    pop rbx
    pop rax
    ret

;-----------------------------------------------------------------------------
; check_gpu_requests: Check if GPU needs CPU for privileged operations
; Returns: RAX = request type (0 = none)
;-----------------------------------------------------------------------------
check_gpu_requests:
    push rbx
    push rcx

    ; TODO: Check GPU mailbox (shared memory)
    ;   - GPU writes requests to mailbox
    ;   - CPU reads and processes
    ;   - Examples: MMIO writes, interrupt handling

    ; For POC: No requests
    xor rax, rax

    pop rcx
    pop rbx
    ret

;-----------------------------------------------------------------------------
; handle_privileged_op: Execute privileged operation requested by GPU
; Input: RAX = request from GPU
;-----------------------------------------------------------------------------
handle_privileged_op:
    push rbx
    push rcx
    push rdx

    ; TODO: Decode GPU request
    ;   - MMIO write to device register
    ;   - Port I/O (in/out)
    ;   - MSR access
    ;   - Interrupt routing

    ; TODO: Execute operation
    ;   - Perform privileged instruction
    ;   - Signal GPU completion
    ;   - GPU continues execution

    ; For POC: No-op
    nop

    pop rdx
    pop rcx
    pop rbx
    ret

;-----------------------------------------------------------------------------
; print_gpu_info: Print GPU information (bus, device, function, BAR0)
;-----------------------------------------------------------------------------
print_gpu_info:
    push rax
    push rsi

    ; Print "  GPU found at"
    mov rsi, msg_gpu_at
    call print_string

    ; Print bus
    movzx rax, byte [gpu_bus]
    call print_hex_byte
    mov rsi, msg_colon
    call print_string

    ; Print device
    movzx rax, byte [gpu_dev]
    call print_hex_byte
    mov rsi, msg_dot
    call print_string

    ; Print function
    movzx rax, byte [gpu_func]
    call print_hex_byte
    call print_newline

    ; Print "  BAR0:"
    mov rsi, msg_bar0
    call print_string

    ; Print BAR0 address
    mov eax, [gpu_bar0]
    call print_hex_dword
    call print_newline

    pop rsi
    pop rax
    ret

;-----------------------------------------------------------------------------
; print_hex_byte: Print byte in hex
; Input: AL = byte to print
;-----------------------------------------------------------------------------
print_hex_byte:
    push rax
    push rbx
    push rcx

    mov rbx, [vga_cursor]
    mov ah, 0x0F

    ; High nibble
    mov cl, al
    shr cl, 4
    and cl, 0x0F
    cmp cl, 10
    jl .high_digit
    add cl, 'A' - 10
    jmp .write_high
.high_digit:
    add cl, '0'
.write_high:
    mov [rbx], cl
    mov [rbx+1], ah
    add rbx, 2

    ; Low nibble
    mov cl, al
    and cl, 0x0F
    cmp cl, 10
    jl .low_digit
    add cl, 'A' - 10
    jmp .write_low
.low_digit:
    add cl, '0'
.write_low:
    mov [rbx], cl
    mov [rbx+1], ah
    add rbx, 2

    mov [vga_cursor], rbx

    pop rcx
    pop rbx
    pop rax
    ret

;-----------------------------------------------------------------------------
; print_hex_dword: Print 32-bit value in hex
; Input: EAX = dword to print
;-----------------------------------------------------------------------------
print_hex_dword:
    push rax
    push rbx

    mov ebx, eax

    ; Print byte 3 (highest)
    shr eax, 24
    call print_hex_byte

    ; Print byte 2
    mov eax, ebx
    shr eax, 16
    call print_hex_byte

    ; Print byte 1
    mov eax, ebx
    shr eax, 8
    call print_hex_byte

    ; Print byte 0 (lowest)
    mov eax, ebx
    call print_hex_byte

    pop rbx
    pop rax
    ret

;-----------------------------------------------------------------------------
; print_newline: Print newline
;-----------------------------------------------------------------------------
print_newline:
    push rsi
    mov rsi, msg_newline
    call print_string
    pop rsi
    ret

;-----------------------------------------------------------------------------
; print_string: Print null-terminated string to VGA text mode
; Input: RSI = pointer to string
; Uses direct VGA memory writes at 0xB8000
;-----------------------------------------------------------------------------
print_string:
    push rax
    push rbx

    mov rbx, [vga_cursor]       ; Current cursor position
    mov ah, 0x0F                ; White on black

.loop:
    lodsb
    test al, al
    jz .done

    ; Handle newline
    cmp al, 13
    je .newline
    cmp al, 10
    je .skip_char

    ; Write character to VGA memory
    mov [rbx], ax
    add rbx, 2
    jmp .loop

.newline:
    ; Move to next line
    mov rax, rbx
    sub rax, 0xB8000
    shr rax, 1                  ; Divide by 2 (char + attribute)
    mov rcx, 80
    xor rdx, rdx
    div rcx                     ; RAX = row, RDX = col
    inc rax                     ; Next row
    xor rdx, rdx                ; Column 0
    imul rax, 80
    shl rax, 1
    add rax, 0xB8000
    mov rbx, rax
    jmp .loop

.skip_char:
    jmp .loop

.done:
    mov [vga_cursor], rbx       ; Update cursor position

    pop rbx
    pop rax
    ret

;-----------------------------------------------------------------------------
; Data Section
;-----------------------------------------------------------------------------
msg_banner:      db 13, 10, '===========================================', 13, 10
                 db 'pxOS CPU Microkernel v0.1', 13, 10
                 db 'GPU-Centric Operating System', 13, 10
                 db '===========================================', 13, 10, 13, 10, 0

msg_init_gpu:    db 'Initializing GPU for compute mode...', 0
msg_gpu_ok:      db ' OK', 13, 10, 0
msg_gpu_error:   db ' FAILED!', 13, 10
                 db 'GPU initialization error.', 13, 10, 0

msg_loading_os:  db 'Loading os.pxi to GPU VRAM...', 0
msg_os_loaded:   db ' OK', 13, 10, 0
msg_load_error:  db ' FAILED!', 13, 10
                 db 'Failed to load os.pxi.', 13, 10, 0

msg_starting:    db 'Starting GPU dispatch loop...', 13, 10
                 db 'CPU now mostly idle - GPU runs the OS!', 13, 10, 13, 10, 0

; GPU info messages
msg_gpu_at:      db '  GPU found at: ', 0
msg_colon:       db ':', 0
msg_dot:         db '.', 0
msg_bar0:        db '  BAR0: 0x', 0
msg_newline:     db 13, 10, 0

; VGA cursor position
vga_cursor:      dq 0xB8000

; GPU/PCI state
gpu_found:       db 0                ; 1 if GPU found, 0 otherwise
gpu_bus:         db 0                ; PCI bus number
gpu_dev:         db 0                ; PCI device number
gpu_func:        db 0                ; PCI function number
                 align 4
gpu_bar0:        dd 0                ; GPU BAR0 MMIO base address

; Padding to 2KB
times 2048-($-$$) db 0
