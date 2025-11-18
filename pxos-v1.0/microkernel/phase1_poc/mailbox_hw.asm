; mailbox_hw.asm - Hardware Mailbox Protocol for pxOS Phase 2 Week 2
; Implements real BAR0 MMIO-based CPU-GPU communication

BITS 64

; Mailbox opcodes (from Phase 1)
OP_MMIO_WRITE_UART  equ 0x80
OP_CPU_HALT         equ 0x8F

section .data
align 8
mailbox_virt_addr:    dq 0        ; Virtual address of BAR0+0x0000 (mailbox)
mailbox_latency_min:  dq 0xFFFFFFFFFFFFFFFF
mailbox_latency_max:  dq 0
mailbox_latency_avg:  dq 0
mailbox_op_count:     dq 0
mailbox_total_cycles: dq 0

section .text

; ---------------------------------------------------------------------------
; init_hw_mailbox - Initialize hardware mailbox at BAR0
; ---------------------------------------------------------------------------
; Input: RDI = BAR0 virtual address
; ---------------------------------------------------------------------------
global init_hw_mailbox
init_hw_mailbox:
    push rax

    ; Store mailbox address (BAR0 + 0x0000)
    mov [mailbox_virt_addr], rdi

    ; Clear mailbox to known state
    xor rax, rax
    mov [rdi], eax

    ; Memory fence to ensure write completes
    mfence

    ; VGA debug: 'H' for hardware mailbox initialized
    mov byte [0xB801C], 'H'
    mov byte [0xB801D], 0x0A  ; Green

    pop rax
    ret

; ---------------------------------------------------------------------------
; mailbox_write - Write request to hardware mailbox
; ---------------------------------------------------------------------------
; Input:
;   RDI = opcode (8 bits)
;   RSI = thread ID (8 bits)
;   RDX = payload (16 bits)
; Output: None
; Modifies: RAX, RBX
; ---------------------------------------------------------------------------
global mailbox_write
mailbox_write:
    push rbx
    push rcx

    ; Build 32-bit mailbox request: [opcode:8 | tid:8 | payload:16]
    mov eax, edi
    shl eax, 24           ; Opcode in bits 31:24

    mov ebx, esi
    shl ebx, 16           ; TID in bits 23:16
    or eax, ebx

    mov ebx, edx
    and ebx, 0xFFFF       ; Payload in bits 15:0
    or eax, ebx

    ; Write to hardware mailbox (UC MMIO write)
    mov rbx, [mailbox_virt_addr]
    mov [rbx], eax

    ; Memory fence to ensure write visibility
    mfence

    pop rcx
    pop rbx
    ret

; ---------------------------------------------------------------------------
; mailbox_poll - Poll mailbox until GPU clears it
; ---------------------------------------------------------------------------
; Waits for mailbox value to become 0 (GPU has processed request)
; Output: None
; Modifies: RAX, RBX
; ---------------------------------------------------------------------------
global mailbox_poll
mailbox_poll:
    push rbx

    mov rbx, [mailbox_virt_addr]

.poll_loop:
    ; Read mailbox (UC MMIO read)
    mov eax, [rbx]
    test eax, eax
    jz .done              ; Mailbox cleared, GPU is done

    ; CPU hint: we're in a spin loop
    pause
    jmp .poll_loop

.done:
    pop rbx
    ret

; ---------------------------------------------------------------------------
; mailbox_read - Read response from mailbox
; ---------------------------------------------------------------------------
; Output: RAX = 32-bit response value
; ---------------------------------------------------------------------------
global mailbox_read
mailbox_read:
    push rbx

    ; Read from mailbox (UC MMIO read)
    mov rbx, [mailbox_virt_addr]
    mov eax, [rbx]

    pop rbx
    ret

; ---------------------------------------------------------------------------
; mailbox_write_and_wait - Write request and wait for completion
; ---------------------------------------------------------------------------
; Input:
;   RDI = opcode
;   RSI = thread ID
;   RDX = payload
; Output: None
; Modifies: RAX, RBX, RCX
; ---------------------------------------------------------------------------
global mailbox_write_and_wait
mailbox_write_and_wait:
    push rdi
    push rsi
    push rdx

    ; Write request
    call mailbox_write

    ; Wait for GPU to process
    call mailbox_poll

    pop rdx
    pop rsi
    pop rdi
    ret

; ---------------------------------------------------------------------------
; mailbox_measure_latency - Measure single mailbox operation latency
; ---------------------------------------------------------------------------
; Input:
;   RDI = opcode
;   RSI = thread ID
;   RDX = payload
; Output: RAX = latency in cycles
; ---------------------------------------------------------------------------
global mailbox_measure_latency
mailbox_measure_latency:
    push rbx
    push rcx
    push rdx
    push rdi
    push rsi

    ; Serialize before RDTSC
    xor eax, eax
    cpuid

    ; Start timestamp
    rdtsc
    shl rdx, 32
    or rdx, rax
    mov r8, rdx           ; Save start time in R8

    ; Perform mailbox operation
    pop rsi
    pop rdi
    push rdi
    push rsi
    call mailbox_write_and_wait

    ; End timestamp
    rdtsc
    shl rdx, 32
    or rdx, rax

    ; Serialize after RDTSC
    xor eax, eax
    cpuid

    ; Calculate latency
    sub rdx, r8
    mov rax, rdx

    ; Update statistics
    call update_mailbox_stats

    pop rsi
    pop rdi
    pop rdx
    pop rcx
    pop rbx
    ret

; ---------------------------------------------------------------------------
; update_mailbox_stats - Update mailbox latency statistics
; ---------------------------------------------------------------------------
; Input: RAX = latency in cycles
; ---------------------------------------------------------------------------
update_mailbox_stats:
    push rbx
    push rcx
    push rdx

    ; Update operation count
    inc qword [mailbox_op_count]

    ; Update total cycles
    add [mailbox_total_cycles], rax

    ; Update min latency
    mov rbx, [mailbox_latency_min]
    cmp rax, rbx
    jae .check_max
    mov [mailbox_latency_min], rax

.check_max:
    ; Update max latency
    mov rbx, [mailbox_latency_max]
    cmp rax, rbx
    jbe .calc_avg
    mov [mailbox_latency_max], rax

.calc_avg:
    ; Calculate average: total / count
    mov rax, [mailbox_total_cycles]
    mov rcx, [mailbox_op_count]
    xor rdx, rdx
    div rcx
    mov [mailbox_latency_avg], rax

    pop rdx
    pop rcx
    pop rbx
    ret

; ---------------------------------------------------------------------------
; get_mailbox_stats - Get mailbox statistics
; ---------------------------------------------------------------------------
; Output:
;   RAX = min latency (cycles)
;   RBX = max latency (cycles)
;   RCX = avg latency (cycles)
;   RDX = operation count
; ---------------------------------------------------------------------------
global get_mailbox_stats
get_mailbox_stats:
    mov rax, [mailbox_latency_min]
    mov rbx, [mailbox_latency_max]
    mov rcx, [mailbox_latency_avg]
    mov rdx, [mailbox_op_count]
    ret

; ---------------------------------------------------------------------------
; print_mailbox_stats - Print latency statistics to serial
; ---------------------------------------------------------------------------
global print_mailbox_stats
print_mailbox_stats:
    push rax
    push rbx
    push rcx
    push rdx
    push rsi

    ; Print header
    mov rsi, msg_stats_header
    extern print_string
    call print_string

    ; Get stats
    call get_mailbox_stats

    ; Print min latency
    mov rsi, msg_min
    call print_string
    mov rax, [mailbox_latency_min]
    call print_decimal
    mov rsi, msg_cycles
    call print_string

    ; Print max latency
    mov rsi, msg_max
    call print_string
    mov rax, [mailbox_latency_max]
    call print_decimal
    mov rsi, msg_cycles
    call print_string

    ; Print avg latency
    mov rsi, msg_avg
    call print_string
    mov rax, [mailbox_latency_avg]
    call print_decimal
    mov rsi, msg_cycles
    call print_string

    ; Print operation count
    mov rsi, msg_ops
    call print_string
    mov rax, [mailbox_op_count]
    call print_decimal
    mov rsi, msg_newline
    call print_string

    pop rsi
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret

; ---------------------------------------------------------------------------
; print_decimal - Print decimal number to serial
; ---------------------------------------------------------------------------
; Input: RAX = number to print
; ---------------------------------------------------------------------------
print_decimal:
    push rax
    push rbx
    push rcx
    push rdx
    push rsi

    ; Handle 0 case
    test rax, rax
    jnz .convert
    mov al, '0'
    extern serial_write_char
    call serial_write_char
    jmp .done

.convert:
    ; Convert to decimal string (on stack)
    mov rbx, 10
    xor rcx, rcx       ; Digit count

.div_loop:
    xor rdx, rdx
    div rbx            ; RAX = quotient, RDX = remainder
    add dl, '0'        ; Convert to ASCII
    push rdx           ; Save digit on stack
    inc rcx
    test rax, rax
    jnz .div_loop

.print_loop:
    pop rax
    extern serial_write_char
    call serial_write_char
    loop .print_loop

.done:
    pop rsi
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret

; ---------------------------------------------------------------------------
; serial_write_char - Write character to UART
; ---------------------------------------------------------------------------
; Input: AL = character
; ---------------------------------------------------------------------------
global serial_write_char
serial_write_char:
    push rdx
    mov dx, 0x3F8     ; COM1
    out dx, al
    pop rdx
    ret

section .rodata
msg_stats_header:  db 13, 10, '=== Mailbox Statistics ===', 13, 10, 0
msg_min:           db 'Min: ', 0
msg_max:           db 'Max: ', 0
msg_avg:           db 'Avg: ', 0
msg_ops:           db 'Ops: ', 0
msg_cycles:        db ' cycles', 13, 10, 0
msg_newline:       db 13, 10, 0
