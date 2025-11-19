; mailbox_hw.asm - Hardware Mailbox Protocol for pxOS Phase 2 Week 2
; Implements real BAR0 MMIO-based CPU-GPU communication
; WITH NULL POINTER CHECKS AND FALLBACK TO SIMULATION

BITS 64

; Mailbox opcodes (from Phase 1)
OP_MMIO_WRITE_UART  equ 0x80
OP_CPU_HALT         equ 0x8F

section .data
align 8
mailbox_virt_addr:    dq 0        ; Virtual address of BAR0+0x0000 (mailbox)
simulation_mailbox:   dq 0x20000  ; Fallback address
hw_mailbox_enabled:   db 0

; Latency stats...
mailbox_latency_min:  dq 0xFFFFFFFFFFFFFFFF
mailbox_latency_max:  dq 0
mailbox_latency_avg:  dq 0
mailbox_op_count:     dq 0
mailbox_total_cycles: dq 0

section .rodata
msg_hw_mailbox_init:  db "Initializing hardware mailbox at virtual address: ", 0
msg_mailbox_null:     db "ERROR: Mailbox virtual address is NULL. Falling back to simulation.", 13, 10, 0

section .text
global init_hw_mailbox, mailbox_write, mailbox_poll, mailbox_read, mailbox_write_and_wait, mailbox_measure_latency, update_mailbox_stats, print_mailbox_stats

init_hw_mailbox:
    mov rsi, msg_hw_mailbox_init
    call print_string_64
    mov rax, [mailbox_virt_addr]
    call print_hex_64

    cmp qword [mailbox_virt_addr], 0
    jne .hw_enabled

    ; Fallback to simulation
    mov rsi, msg_mailbox_null
    call print_string_64
    mov byte [hw_mailbox_enabled], 0
    ret

.hw_enabled:
    mov byte [hw_mailbox_enabled], 1
    ret

mailbox_write:
    cmp byte [hw_mailbox_enabled], 1
    je .do_hw_write
    ; Fallback
    mov rdi, [simulation_mailbox]
    mov [rdi], rax
    ret
.do_hw_write:
    mov rdi, [mailbox_virt_addr]
    mov [rdi], rax
    mfence
    ret

mailbox_poll:
    cmp byte [hw_mailbox_enabled], 1
    je .do_hw_poll
    ; Fallback
    mov rdi, [simulation_mailbox]
.sim_poll_loop:
    cmp dword [rdi], 0
    je .sim_poll_done
    pause
    jmp .sim_poll_loop
.sim_poll_done:
    ret
.do_hw_poll:
    mov rdi, [mailbox_virt_addr]
.hw_poll_loop:
    cmp dword [rdi], 0
    je .hw_poll_done
    pause
    jmp .hw_poll_loop
.hw_poll_done:
    ret

mailbox_read:
    cmp byte [hw_mailbox_enabled], 1
    je .do_hw_read
    ; Fallback
    mov rdi, [simulation_mailbox]
    mov rax, [rdi]
    ret
.do_hw_read:
    mov rdi, [mailbox_virt_addr]
    mov rax, [rdi]
    ret

mailbox_write_and_wait:
    call mailbox_write
    call mailbox_poll
    ret

; ... rest of the file is the same (latency measurement, stats, etc)
; ...
