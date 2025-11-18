; pxOS Microkernel v0.5 - Phase 2 Week 2: Hardware Mailbox (Combined)
; Replaces GPU simulation with real BAR0 MMIO operations
; Includes hardware mailbox implementation inline

BITS 32
ORG 0x10000

; Entry from bootloader (32-bit mode)
kernel_entry_32:
    mov byte [0xB800C], 'M'
    mov byte [0xB800D], 0x0F

    call setup_page_tables_32

    mov byte [0xB800E], 'T'
    mov byte [0xB800F], 0x0F

    mov eax, cr4
    or eax, 0x20
    mov cr4, eax

    mov eax, pml4_table
    mov cr3, eax

    mov ecx, 0xC0000080
    rdmsr
    or eax, 0x100
    wrmsr

    mov eax, cr0
    or eax, 0x80000000
    mov cr0, eax

    lgdt [gdt64_descriptor]
    jmp 0x08:kernel_entry_64

setup_page_tables_32:
    mov edi, pml4_table
    mov ecx, 4096 * 3 / 4
    xor eax, eax
    cld
    rep stosd

    mov eax, pdpt_table
    or eax, 0x3
    mov [pml4_table], eax

    mov eax, pd_table
    or eax, 0x3
    mov [pdpt_table], eax

    mov edi, pd_table
    mov eax, 0x83
    mov ecx, 512
.loop:
    mov [edi], eax
    add eax, 0x200000
    add edi, 8
    loop .loop
    ret

BITS 64
kernel_entry_64:
    mov byte [0xB8010], '6'
    mov byte [0xB8011], 0x0F

    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov rsp, 0x90000

    mov byte [0xB8012], 'G'
    mov byte [0xB8013], 0x0F

    call microkernel_main

    cli
.hang:
    hlt
    jmp .hang

; ===========================================================================
; MAIN MICROKERNEL
; ===========================================================================
UART_PORT equ 0x3F8
OP_MMIO_WRITE_UART equ 0x80
OP_CPU_HALT equ 0x8F

microkernel_main:
    mov rsi, msg_microkernel
    call print_string

    mov rsi, msg_init_pat
    call print_string
    call init_pat_cpu
    call print_ok

    mov rsi, msg_scanning_pcie
    call print_string
    call pcie_scan
    call print_ok

    mov rsi, msg_mapping_bars
    call print_string
    call init_bar_mapping
    call print_ok

    mov rsi, msg_init_hw_mailbox
    call print_string
    mov rdi, [gpu_bar0]
    call mailbox_init
    call print_ok

    mov rsi, msg_testing_mailbox
    call print_string
    call test_hw_mailbox_hello
    call print_ok

    call print_mailbox_stats

    mov rsi, msg_gpu_halted
    call print_string
    cli
    hlt

; ===========================================================================
; HARDWARE MAILBOX FUNCTIONS (Inline)
; ===========================================================================

; Initialize hardware mailbox
mailbox_init:
    push rax
    mov [mailbox_addr], rdi
    xor rax, rax
    mov [rdi], eax
    mfence
    mov byte [0xB801C], 'H'
    mov byte [0xB801D], 0x0A
    pop rax
    ret

; Write to mailbox: RDI=opcode, RSI=tid, RDX=payload
mailbox_write:
    push rbx
    push rcx
    mov eax, edi
    shl eax, 24
    mov ebx, esi
    shl ebx, 16
    or eax, ebx
    mov ebx, edx
    and ebx, 0xFFFF
    or eax, ebx
    mov rbx, [mailbox_addr]
    mov [rbx], eax
    mfence
    pop rcx
    pop rbx
    ret

; Poll mailbox until clear
mailbox_poll:
    push rbx
    mov rbx, [mailbox_addr]
.poll_loop:
    mov eax, [rbx]
    test eax, eax
    jz .done
    pause
    jmp .poll_loop
.done:
    pop rbx
    ret

; Write and wait
mailbox_write_and_wait:
    push rdi
    push rsi
    push rdx
    call mailbox_write
    call mailbox_poll
    pop rdx
    pop rsi
    pop rdi
    ret

; Measure latency
mailbox_measure_latency:
    push rbx
    push rcx
    push rdx
    push rdi
    push rsi

    xor eax, eax
    cpuid
    rdtsc
    shl rdx, 32
    or rdx, rax
    mov r8, rdx

    pop rsi
    pop rdi
    push rdi
    push rsi
    call mailbox_write_and_wait

    rdtsc
    shl rdx, 32
    or rdx, rax
    xor eax, eax
    cpuid

    sub rdx, r8
    mov rax, rdx
    call update_stats

    pop rsi
    pop rdi
    pop rdx
    pop rcx
    pop rbx
    ret

; Update statistics
update_stats:
    push rbx
    push rcx
    push rdx

    inc qword [stat_count]
    add [stat_total], rax

    mov rbx, [stat_min]
    cmp rax, rbx
    jae .check_max
    mov [stat_min], rax

.check_max:
    mov rbx, [stat_max]
    cmp rax, rbx
    jbe .calc_avg
    mov [stat_max], rax

.calc_avg:
    mov rax, [stat_total]
    mov rcx, [stat_count]
    xor rdx, rdx
    div rcx
    mov [stat_avg], rax

    pop rdx
    pop rcx
    pop rbx
    ret

; Print statistics
print_mailbox_stats:
    push rax
    push rsi

    mov rsi, msg_stats_header
    call print_string

    mov rsi, msg_min
    call print_string
    mov rax, [stat_min]
    call print_decimal
    mov rsi, msg_cycles
    call print_string

    mov rsi, msg_max
    call print_string
    mov rax, [stat_max]
    call print_decimal
    mov rsi, msg_cycles
    call print_string

    mov rsi, msg_avg
    call print_string
    mov rax, [stat_avg]
    call print_decimal
    mov rsi, msg_cycles
    call print_string

    mov rsi, msg_ops
    call print_string
    mov rax, [stat_count]
    call print_decimal
    mov rsi, msg_newline
    call print_string

    pop rsi
    pop rax
    ret

; Print decimal number
print_decimal:
    push rax
    push rbx
    push rcx
    push rdx

    test rax, rax
    jnz .convert
    mov al, '0'
    call uart_write
    jmp .done

.convert:
    mov rbx, 10
    xor rcx, rcx

.div_loop:
    xor rdx, rdx
    div rbx
    add dl, '0'
    push rdx
    inc rcx
    test rax, rax
    jnz .div_loop

.print_loop:
    pop rax
    call uart_write
    loop .print_loop

.done:
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret

; ===========================================================================
; TEST FUNCTION
; ===========================================================================
test_hw_mailbox_hello:
    push rax
    push rcx
    push rdx
    push rsi
    push rdi

    mov rsi, hello_msg
    xor rcx, rcx

.send_loop:
    movzx rax, byte [rsi + rcx]
    test al, al
    jz .done

    mov rdi, OP_MMIO_WRITE_UART
    xor rsi, rsi
    movzx rdx, al
    call mailbox_measure_latency

    mov rsi, hello_msg
    inc rcx
    jmp .send_loop

.done:
    mov rdi, OP_MMIO_WRITE_UART
    xor rsi, rsi
    mov rdx, 10
    call mailbox_write_and_wait

    pop rdi
    pop rsi
    pop rdx
    pop rcx
    pop rax
    ret

; ===========================================================================
; PAT / PCIe / BAR MAPPING (from Phase 2 Week 1)
; ===========================================================================
init_pat_cpu:
    push rax
    push rcx
    push rdx
    mov ecx, 0x277
    rdmsr
    mov eax, 0x00000406
    mov edx, 0x01000406
    wrmsr
    pop rdx
    pop rcx
    pop rax
    ret

pcie_scan:
    push rax
    push rbx
    push rcx
    push rdx
    xor rbx, rbx

.scan_loop:
    cmp rbx, 32
    jge .scan_done

    mov rax, rbx
    shl rax, 15
    or rax, 0x80000000
    mov dx, 0xCF8
    out dx, eax
    mov dx, 0xCFC
    in eax, dx

    cmp ax, 0xFFFF
    je .next_device

    mov rax, rbx
    shl rax, 15
    or rax, 0x80000008
    or rax, 0x80000000
    mov dx, 0xCF8
    out dx, eax
    mov dx, 0xCFC
    in eax, dx
    shr eax, 16

    cmp ax, 0x0300
    je .found_gpu

.next_device:
    inc rbx
    jmp .scan_loop

.found_gpu:
    mov [gpu_device], rbx
    mov rax, rbx
    shl rax, 15
    or rax, 0x80000010
    or rax, 0x80000000
    mov dx, 0xCF8
    out dx, eax
    mov dx, 0xCFC
    in eax, dx
    and eax, 0xFFFFFFF0
    mov [gpu_bar0], rax

.scan_done:
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret

init_bar_mapping:
    push rax
    push rbx
    push rcx
    push rdx

    mov rax, [gpu_bar0]
    test rax, rax
    jz .no_gpu

    mov rbx, rax
    shr rbx, 21
    and rbx, 0x1FF
    mov rcx, pd_table
    shl rbx, 3
    add rcx, rbx
    mov rdx, rax
    and rdx, 0xFFFFFFFFFFE00000
    or rdx, 0x83
    or rdx, (1 << 12)
    mov [rcx], rdx
    invlpg [rax]

    add rax, 0x200000
    mov r8, 7

.map_wc_loop:
    mov rbx, rax
    shr rbx, 21
    and rbx, 0x1FF
    mov rcx, pd_table
    shl rbx, 3
    add rcx, rbx
    mov rdx, rax
    and rdx, 0xFFFFFFFFFFE00000
    or rdx, 0x83
    or rdx, (1 << 7)
    or rdx, (1 << 12)
    mov [rcx], rdx
    invlpg [rax]
    add rax, 0x200000
    dec r8
    jnz .map_wc_loop

.no_gpu:
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret

; ===========================================================================
; UTILITY FUNCTIONS
; ===========================================================================
print_string:
    push rax
.loop:
    lodsb
    test al, al
    jz .done

    mov rdi, [vga_ptr]
    mov [rdi], al
    inc rdi
    mov byte [rdi], 0x0F
    inc rdi
    mov [vga_ptr], rdi

    call uart_write
    jmp .loop

.done:
    pop rax
    ret

uart_write:
    push rdx
    mov dx, UART_PORT
    out dx, al
    pop rdx
    ret

print_ok:
    push rsi
    mov rsi, msg_ok
    call print_string
    pop rsi
    ret

; ===========================================================================
; DATA
; ===========================================================================
align 8
msg_microkernel:      db 'pxOS CPU Microkernel v0.5 (Week 2)', 13, 10, 0
msg_init_pat:         db 'Initializing PAT... ', 0
msg_scanning_pcie:    db 'Scanning PCIe... ', 0
msg_mapping_bars:     db 'Mapping BARs... ', 0
msg_init_hw_mailbox:  db 'Initializing HW mailbox... ', 0
msg_testing_mailbox:  db 'Testing HW mailbox... ', 0
msg_gpu_halted:       db 13, 10, 'System halted.', 13, 10, 0
msg_ok:               db 'OK', 13, 10, 0

msg_stats_header:     db 13, 10, '=== Mailbox Statistics ===', 13, 10, 0
msg_min:              db 'Min: ', 0
msg_max:              db 'Max: ', 0
msg_avg:              db 'Avg: ', 0
msg_ops:              db 'Ops: ', 0
msg_cycles:           db ' cycles', 13, 10, 0
msg_newline:          db 13, 10, 0

hello_msg:            db 'Hello from GPU OS!', 0
vga_ptr:              dq 0xB8014

gpu_device:           dq 0
gpu_bar0:             dq 0
mailbox_addr:         dq 0

stat_min:             dq 0xFFFFFFFFFFFFFFFF
stat_max:             dq 0
stat_avg:             dq 0
stat_total:           dq 0
stat_count:           dq 0

; ===========================================================================
; GDT
; ===========================================================================
align 16
gdt64_start:
    dq 0
    dq 0x00AF9A000000FFFF
    dq 0x00AF92000000FFFF
gdt64_end:

gdt64_descriptor:
    dw gdt64_end - gdt64_start - 1
    dq gdt64_start

; ===========================================================================
; PAGE TABLES
; ===========================================================================
align 4096
pml4_table: times 4096 db 0
pdpt_table: times 4096 db 0
pd_table:   times 4096 db 0
