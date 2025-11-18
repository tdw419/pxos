; pxOS CPU Microkernel v0.1 - Privilege Broker Final
; Minimal 2KB kernel: GPU initialization + dispatch loop + Privilege Broker

BITS 64
ORG 0x10000

; ---------------------------------------------------------------------------
; PXI-OS CONSTANTS & MAILBOX SETUP
; ---------------------------------------------------------------------------
MAILBOX_ADDR equ 0x20000          ; Shared physical address for GPU -> CPU requests
UART_PORT    equ 0x3F8            ; Standard COM1 port for privileged I/O

OP_MMIO_WRITE_UART  equ 0x80      ; PXI Opcode: Print one character
OP_CPU_HALT         equ 0x8F      ; PXI Opcode: Halt the system

; ---------------------------------------------------------------------------
; Entry point (called from bootloader)
; ---------------------------------------------------------------------------
start:
    ; Setup stack
    mov rsp, 0x90000

    ; Print banner
    mov rsi, msg_microkernel
    call print_string

    ; Initialize GPU
    call init_gpu
    test rax, rax
    jz .gpu_error

    ; Load os.pxi to GPU VRAM
    call load_os_pxi

    ; Clear mailbox to prepare for GPU requests
    xor rax, rax
    mov qword [MAILBOX_ADDR], rax

    ; Enter GPU dispatch loop
    jmp gpu_dispatch_loop

.gpu_error:
    mov rsi, msg_gpu_error
    call print_string
    cli
    hlt

;-----------------------------------------------------------------------------
; init_gpu: Initialize GPU for compute (Stub)
;-----------------------------------------------------------------------------
init_gpu:
    push rbx
    push rcx
    push rdx
    mov rsi, msg_scanning_pcie
    call print_string
    ; Simulate success
    call print_ok
    mov rax, 1
    pop rdx
    pop rcx
    pop rbx
    ret

;-----------------------------------------------------------------------------
; load_os_pxi: Load os.pxi from disk to GPU VRAM (Stub)
;-----------------------------------------------------------------------------
load_os_pxi:
    mov rsi, msg_loading_os
    call print_string
    ; Simulate success
    call print_ok
    mov rax, 1
    ret

;-----------------------------------------------------------------------------
; gpu_dispatch_loop: Main dispatch loop (CPU is the work distributor)
;-----------------------------------------------------------------------------
gpu_dispatch_loop:
    ; Submit GPU work (Simulated)
    call gpu_execute_os

    ; Check for GPU requests (Mailbox Check)
    call check_gpu_requests
    test rax, rax
    jz .no_requests

    ; Handle privileged operation
    call handle_privileged_op

.no_requests:
    ; Brief pause so we don't burn the CPU
    pause
    jmp gpu_dispatch_loop

;-----------------------------------------------------------------------------
; check_gpu_requests: Reads mailbox status
; Returns: RAX = 1 if request is pending, 0 otherwise
;-----------------------------------------------------------------------------
check_gpu_requests:
    ; Read the full 32-bit mailbox word (Opcode|TID|Payload)
    mov eax, [MAILBOX_ADDR]
    test eax, eax
    jnz .request_pending

    mov rax, 0
    ret
.request_pending:
    mov rax, 1
    ret

;-----------------------------------------------------------------------------
; handle_privileged_op: Executes the request received from the GPU.
; Clobbers: RBX, RCX, RDX, RDI
;-----------------------------------------------------------------------------
handle_privileged_op:
    push rbx
    push rcx
    push rdx
    push rdi

    ; Read request word from shared memory (Mailbox)
    mov ebx, [MAILBOX_ADDR]

    ; 1. Extract Opcode (Bits 31:24, contained in BH)
    mov al, bh                  ; AL = Opcode

    cmp al, OP_MMIO_WRITE_UART
    je .handle_uart_write

    cmp al, OP_CPU_HALT
    je .handle_halt

    jmp .clear_mailbox          ; Unknown or unhandled opcode

.handle_uart_write:
    ; Payload is the ASCII character (Bits 15:0)
    and ebx, 0xFFFF             ; EBX = ASCII value
    mov al, bl                  ; AL = Character

    ; Execute the privileged I/O instruction!
    mov dx, UART_PORT           ; DX = 0x3F8
    out dx, al                  ; [PRIVILEGED] Write char to UART

    ; Output confirmation to VGA (optional debugging)
    mov rsi, msg_uart_ok
    call print_string           ; Prints "[W]" to screen

    jmp .clear_mailbox

.handle_halt:
    mov rsi, msg_gpu_halted
    call print_string
    cli
    hlt                         ; CPU halts indefinitely

.clear_mailbox:
    ; Acknowledge completion by clearing the mailbox word
    xor eax, eax
    mov [MAILBOX_ADDR], eax     ; Signals the GPU that the CPU is done.

    pop rdi
    pop rdx
    pop rcx
    pop rbx
    ret

;-----------------------------------------------------------------------------
; Helper functions
;-----------------------------------------------------------------------------
print_string:
    push rax
    push rbx
    push rcx
    push rdx
    push rsi
    push rdi

    mov ax, 0x0E00      ; AH=0x0E (Teletype), AL is char
.loop:
    lodsb
    test al, al
    jz .done
    int 0x10            ; Relying on INT 10h BIOS service (slow, but works for POC)
    jmp .loop
.done:
    pop rdi
    pop rsi
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret

print_ok:
    push rsi
    mov rsi, msg_ok
    call print_string
    pop rsi
    ret

gpu_execute_os:
    ; In Phase 1, we simulate GPU execution by writing a request to the mailbox
    ; to immediately trigger the CPU broker.

    ; 1. Check if we've finished the 'H' character
    cmp byte [hello_pos], 1
    jge .done_sim

    ; Simulate GPU writing the first character request ('H')
    ; (OP_MMIO_WRITE_UART << 24) | 'H'
    mov eax, (OP_MMIO_WRITE_UART << 24) | 0x48       ; 0x80000048 (Opcode 0x80, Char 'H')
    mov [MAILBOX_ADDR], eax
    inc byte [hello_pos]

    mov rsi, msg_gpu_exec
    call print_string                              ; Prints "GPU Executed Cycle."
    ret

.done_sim:
    ; After 'H', simulate the GPU sending HALT
    mov eax, (OP_CPU_HALT << 24)                   ; 0x8F000000 (Opcode 0x8F, Halt)
    mov [MAILBOX_ADDR], eax

    ret

;-----------------------------------------------------------------------------
; Data
;-----------------------------------------------------------------------------
msg_microkernel:   db 13, 10, 'pxOS CPU Microkernel v0.1', 13, 10, 0
msg_scanning_pcie: db 'Scanning PCIe for GPU... ', 0
msg_loading_os:    db 'Loading os.pxi to VRAM... ', 0
msg_gpu_error:     db 'GPU initialization failed!', 13, 10, 0
msg_gpu_exec:      db 'GPU Executed Cycle.', 13, 10, 0
msg_gpu_halted:    db 'GPU HALT received. CPU shutting down.', 13, 10, 0
msg_uart_ok:       db '[W]', 0                                          ; Debug confirmation of UART write
msg_ok:            db 'OK', 13, 10, 0

hello_pos:         db 0                                                ; Simulation helper

; Reserve space for microkernel (up to 2KB)
times 2048-($-$$) db 0
