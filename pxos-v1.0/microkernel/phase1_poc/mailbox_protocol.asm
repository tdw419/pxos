; mailbox_protocol.asm - Hardware CPU-GPU Mailbox Implementation
; Provides low-level communication between CPU microkernel and GPU runtime

BITS 64

; Mailbox register layout (BAR0 + offsets)
%define MAILBOX_CMD_REG     0x0000   ; Command register (CPU writes)
%define MAILBOX_STATUS_REG  0x0004   ; Status register (GPU writes)
%define MAILBOX_RESP_REG    0x0008   ; Response register (GPU writes)
%define MAILBOX_DOORBELL    0x000C   ; Doorbell (trigger GPU processing)

; Status bits
%define STATUS_READY        0x00000001  ; GPU ready for commands
%define STATUS_BUSY         0x00000002  ; GPU processing command
%define STATUS_COMPLETE     0x00000004  ; Command completed
%define STATUS_ERROR        0x80000000  ; Error occurred

; Mailbox opcodes (from pixel ISA)
%define OP_UART_WRITE       0x80
%define OP_UART_READ        0x81
%define OP_GPU_EXECUTE      0x82
%define OP_MMIO_READ        0x83
%define OP_MMIO_WRITE       0x84
%define OP_HALT             0x8F

section .data
align 8
; External symbols from BAR0 mapper
extern gpu_bar0_virt

; Mailbox statistics
mailbox_stats:
    .commands_sent:     dq 0
    .commands_complete: dq 0
    .commands_failed:   dq 0
    .total_cycles:      dq 0

; Debug messages
msg_mailbox_init:   db "Initializing mailbox protocol...", 0
msg_mailbox_ready:  db " READY", 13, 10, 0
msg_mailbox_test:   db "Testing mailbox: sending 'H' via UART...", 0
msg_mailbox_ok:     db " OK", 13, 10, 0

section .text
global mailbox_init
global mailbox_send_command
global mailbox_poll_complete
global mailbox_read_response
global mailbox_test

extern serial_print_64
extern serial_putc_64

;-----------------------------------------------------------------------------
; mailbox_init()
; Initialize hardware mailbox protocol
; Must be called after map_gpu_bar0
;-----------------------------------------------------------------------------
mailbox_init:
    push rax
    push rdi
    push rsi

    ; Print init message
    lea rsi, [rel msg_mailbox_init]
    call serial_print_64

    ; Get BAR0 virtual address
    mov rdi, [rel gpu_bar0_virt]
    test rdi, rdi
    jz .no_hardware

    ; Clear mailbox registers
    mov dword [rdi + MAILBOX_CMD_REG], 0
    mov dword [rdi + MAILBOX_STATUS_REG], STATUS_READY
    mov dword [rdi + MAILBOX_RESP_REG], 0
    mov dword [rdi + MAILBOX_DOORBELL], 0

    ; Memory fence to ensure writes complete
    mfence

    ; Success message
    lea rsi, [rel msg_mailbox_ready]
    call serial_print_64

    jmp .done

.no_hardware:
    ; BAR0 not mapped - can't use hardware mailbox
    ; System will fall back to simulation

.done:
    pop rsi
    pop rdi
    pop rax
    ret

;-----------------------------------------------------------------------------
; mailbox_send_command(command)
; Send 32-bit command to GPU via mailbox
; Input: RDI = 32-bit command (opcode:8 | tid:8 | payload:16)
; Output: RAX = 0 on success, error code otherwise
;-----------------------------------------------------------------------------
mailbox_send_command:
    push rbx
    push rcx
    push rdx
    push rdi

    ; Get BAR0 virtual address
    mov rbx, [rel gpu_bar0_virt]
    test rbx, rbx
    jz .no_hardware

    ; Wait for mailbox to be ready (not busy)
    mov rcx, 100000              ; Timeout counter
.wait_ready:
    mov eax, [rbx + MAILBOX_STATUS_REG]
    test eax, STATUS_BUSY
    jz .send                     ; Not busy, can send
    pause
    loop .wait_ready

    ; Timeout
    mov rax, 0xFFFFFFFF
    jmp .done

.send:
    ; Write command to command register
    mov eax, edi
    mov [rbx + MAILBOX_CMD_REG], eax

    ; Memory fence to ensure command is written
    mfence

    ; Ring doorbell to notify GPU
    mov dword [rbx + MAILBOX_DOORBELL], 1
    mfence

    ; Update statistics
    inc qword [rel mailbox_stats.commands_sent]

    ; Success
    xor rax, rax
    jmp .done

.no_hardware:
    ; No BAR0 - return error
    mov rax, 1

.done:
    pop rdi
    pop rdx
    pop rcx
    pop rbx
    ret

;-----------------------------------------------------------------------------
; mailbox_poll_complete()
; Wait for GPU to complete current command
; Output: RAX = status (0=success, error code otherwise)
;-----------------------------------------------------------------------------
mailbox_poll_complete:
    push rbx
    push rcx

    ; Get BAR0 virtual address
    mov rbx, [rel gpu_bar0_virt]
    test rbx, rbx
    jz .no_hardware

    ; Poll status register with timeout
    mov rcx, 1000000             ; 1M iterations timeout
.poll_loop:
    mov eax, [rbx + MAILBOX_STATUS_REG]

    ; Check if complete
    test eax, STATUS_COMPLETE
    jnz .complete

    ; Check if error
    test eax, STATUS_ERROR
    jnz .error

    ; Still busy, keep polling
    pause
    loop .poll_loop

    ; Timeout
    mov rax, 0xFFFFFFFE
    jmp .done

.complete:
    ; Success - clear complete flag
    mov dword [rbx + MAILBOX_STATUS_REG], STATUS_READY
    mfence

    ; Update statistics
    inc qword [rel mailbox_stats.commands_complete]

    xor rax, rax
    jmp .done

.error:
    ; GPU reported error
    inc qword [rel mailbox_stats.commands_failed]
    mov eax, [rbx + MAILBOX_RESP_REG]
    or eax, STATUS_ERROR
    ; EAX write automatically zero-extends to RAX in 64-bit mode
    jmp .done

.no_hardware:
    xor rax, rax                 ; Assume success in simulation

.done:
    pop rcx
    pop rbx
    ret

;-----------------------------------------------------------------------------
; mailbox_read_response()
; Read response from GPU after command completion
; Output: RAX = 32-bit response value
;-----------------------------------------------------------------------------
mailbox_read_response:
    push rbx

    ; Get BAR0 virtual address
    mov rbx, [rel gpu_bar0_virt]
    test rbx, rbx
    jz .no_hardware

    ; Read response register
    mov eax, [rbx + MAILBOX_RESP_REG]
    jmp .done

.no_hardware:
    xor rax, rax

.done:
    pop rbx
    ret

;-----------------------------------------------------------------------------
; mailbox_test()
; Test mailbox by sending 'H' via UART
;-----------------------------------------------------------------------------
mailbox_test:
    push rdi
    push rsi

    ; Print test message
    lea rsi, [rel msg_mailbox_test]
    call serial_print_64

    ; Build UART write command
    ; Format: opcode:8 | tid:8 | payload:16
    ; OP_UART_WRITE (0x80) | tid=0 | char='H'
    mov rdi, (OP_UART_WRITE << 24) | ('H' & 0xFFFF)

    ; Send command
    call mailbox_send_command
    test rax, rax
    jnz .failed

    ; Wait for completion
    call mailbox_poll_complete
    test rax, rax
    jnz .failed

    ; Success
    lea rsi, [rel msg_mailbox_ok]
    call serial_print_64
    jmp .done

.failed:
    ; Test failed - print error
    mov al, 'F'
    call serial_putc_64
    mov al, 'A'
    call serial_putc_64
    mov al, 'I'
    call serial_putc_64
    mov al, 'L'
    call serial_putc_64
    mov al, 13
    call serial_putc_64
    mov al, 10
    call serial_putc_64

.done:
    pop rsi
    pop rdi
    ret
