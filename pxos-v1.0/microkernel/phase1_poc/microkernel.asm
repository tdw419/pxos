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
; init_gpu: Initialize GPU for compute mode
; Returns: RAX = GPU handle (0 on error)
;-----------------------------------------------------------------------------
init_gpu:
    push rbx
    push rcx
    push rdx
    push rsi

    ; Phase 1 POC: Simulate GPU initialization
    ; In Phase 2, we'll add real PCIe enumeration

    ; TODO: Enumerate PCIe devices
    ;   - Scan bus 0-255
    ;   - Check vendor ID / device ID
    ;   - Find VGA class device (GPU)

    ; TODO: Map GPU BAR (Base Address Register)
    ;   - Read BAR0 from PCIe config space
    ;   - Setup page tables for MMIO
    ;   - Map GPU memory-mapped registers

    ; TODO: Initialize GPU for compute
    ;   - Write to GPU control registers
    ;   - Enable compute mode
    ;   - Allocate GPU VRAM for OS

    ; For POC: Return simulated GPU handle
    mov rax, 0x1000000          ; Fake GPU handle

    pop rsi
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

; VGA cursor position
vga_cursor:      dq 0xB8000

; Padding to 2KB
times 2048-($-$$) db 0
