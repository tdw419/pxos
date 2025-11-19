; pxOS CPU Microkernel v0.2
; GPU-Centric OS with Privilege Broker

BITS 64
ORG 0x1000

; --- Constants ---
MAILBOX_ADDR    equ 0xA0000     ; Shared memory for CPU-GPU communication
MAILBOX_SIZE    equ 256         ; Number of 32-bit slots in the mailbox

; Mailbox Request Opcodes (upper 8 bits of request)
OP_MMIO_WRITE_UART  equ 0x80
OP_CPU_HALT         equ 0xFF

; --- Entry Point ---
start:
    mov rsp, 0x90000
    mov rsi, msg_banner
    call print_string_vga
    call init_gpu
    call load_os_pxi
    call gpu_dispatch_loop

; --- GPU Dispatch Loop (Privilege Broker) ---
gpu_dispatch_loop:
    mov rsi, msg_dispatch_loop
    call print_string_vga

    ; Main loop: Poll mailbox for GPU requests
.poll_loop:
    mov rdi, MAILBOX_ADDR
    mov rcx, MAILBOX_SIZE
.check_slot:
    cmp rcx, 0
    je .no_request

    ; Read mailbox slot atomically (or just read for PoC)
    mov eax, [rdi]
    test eax, eax
    jz .next_slot ; Slot is empty

    ; Request found! Process it.
    call handle_gpu_request

    ; Clear the slot
    mov dword [rdi], 0

.next_slot:
    add rdi, 4
    dec rcx
    jmp .check_slot

.no_request:
    ; No request, briefly pause and poll again
    pause
    jmp .poll_loop

handle_gpu_request:
    ; EAX = request from GPU
    ; Bits 31:24 = Opcode
    ; Bits 23:0  = Payload
    mov r8d, eax        ; Save original request
    shr eax, 24         ; EAX = Opcode

    cmp al, OP_MMIO_WRITE_UART
    je .handle_uart_write

    cmp al, OP_CPU_HALT
    je .handle_cpu_halt

    ; Unknown opcode, ignore for now
    ret

.handle_uart_write:
    ; Payload (bits 7:0) contains the character
    mov r9d, r8d
    and r9d, 0xFF       ; R9D = character

    ; Write to COM1 serial port
    mov al, r9b
    mov dx, 0x3F8
    out dx, al

    ; Wait for transmit buffer to be empty
    ; (omitted for brevity in this PoC)
    ret

.handle_cpu_halt:
    mov rsi, msg_gpu_halt
    call print_string_vga
    cli
    hlt
    ret

; --- Stubs for GPU/PXI Loading (from previous step) ---
init_gpu:
    mov rsi, msg_init_gpu
    call print_string_vga
    ret
load_os_pxi:
    mov rsi, msg_loading_os
    call print_string_vga
    ret

; --- VGA Print Helper ---
print_string_vga:
    push rax
    push rbx
    mov rbx, 0xB8000
.loop:
    lodsb
    test al, al
    jz .done
    mov [rbx], al
    mov byte [rbx+1], 0x0F
    add rbx, 2
    jmp .loop
.done:
    pop rbx
    pop rax
    ret

; --- Data ---
msg_banner:         db 'pxOS CPU Microkernel v0.2', 13, 10, 0
msg_init_gpu:       db 'Initializing GPU... OK', 13, 10, 0
msg_loading_os:     db 'Loading os.pxi to VRAM... OK', 13, 10, 0
msg_dispatch_loop:  db 'Starting GPU dispatch loop...', 13, 10, 0
msg_gpu_halt:       db 'GPU requested CPU halt. System shutting down.', 13, 10, 0

times 2048-($-$$) db 0
