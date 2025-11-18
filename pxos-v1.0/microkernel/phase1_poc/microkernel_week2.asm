; pxOS Microkernel v0.5 - Phase 2 Week 2: Hardware Mailbox
; Replaces GPU simulation with real BAR0 MMIO operations

BITS 32
ORG 0x10000

; Entry from bootloader (32-bit mode)
kernel_entry_32:
    ; VGA debug: 'M' - Microkernel reached
    mov byte [0xB800C], 'M'
    mov byte [0xB800D], 0x0F

    ; Setup page tables
    call setup_page_tables_32

    ; VGA debug: 'T' - Page tables ready
    mov byte [0xB800E], 'T'
    mov byte [0xB800F], 0x0F

    ; Enable PAE
    mov eax, cr4
    or eax, 0x20
    mov cr4, eax

    ; Load CR3
    mov eax, pml4_table
    mov cr3, eax

    ; Enable Long Mode
    mov ecx, 0xC0000080   ; EFER MSR
    rdmsr
    or eax, 0x100         ; LME bit
    wrmsr

    ; Enable Paging
    mov eax, cr0
    or eax, 0x80000000    ; PG bit
    mov cr0, eax

    ; Load 64-bit GDT
    lgdt [gdt64_descriptor]

    ; Far jump to 64-bit code
    jmp 0x08:kernel_entry_64

setup_page_tables_32:
    ; Clear tables
    mov edi, pml4_table
    mov ecx, 4096 * 3 / 4
    xor eax, eax
    cld
    rep stosd

    ; PML4[0] → PDPT
    mov eax, pdpt_table
    or eax, 0x3
    mov [pml4_table], eax

    ; PDPT[0] → PD
    mov eax, pd_table
    or eax, 0x3
    mov [pdpt_table], eax

    ; PD entries (512 × 2MB = 1GB)
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
    ; VGA debug: '6' - 64-bit mode
    mov byte [0xB8010], '6'
    mov byte [0xB8011], 0x0F

    ; Setup 64-bit segments
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov rsp, 0x90000

    ; VGA debug: 'G' - Going to main
    mov byte [0xB8012], 'G'
    mov byte [0xB8013], 0x0F

    ; Continue with microkernel main
    call microkernel_main

    ; Hang if we return
    cli
.hang:
    hlt
    jmp .hang

; ---------------------------------------------------------------------------
; Microkernel Main (64-bit) - Week 2 Version with Hardware Mailbox
; ---------------------------------------------------------------------------
UART_PORT    equ 0x3F8

OP_MMIO_WRITE_UART  equ 0x80
OP_CPU_HALT         equ 0x8F

microkernel_main:
    ; Print banner
    mov rsi, msg_microkernel
    call print_string

    ; Phase 2 Week 2: Initialize PAT
    mov rsi, msg_init_pat
    call print_string
    call init_pat_cpu
    call print_ok

    ; Scan PCIe bus
    mov rsi, msg_scanning_pcie
    call print_string
    call pcie_scan
    call print_ok

    ; Map GPU BAR regions
    mov rsi, msg_mapping_bars
    call print_string
    call init_bar_mapping
    call print_ok

    ; Week 2: Initialize hardware mailbox
    mov rsi, msg_init_hw_mailbox
    call print_string
    mov rdi, [gpu_bar0]           ; BAR0 virtual address
    extern init_hw_mailbox
    call init_hw_mailbox
    call print_ok

    ; Week 2: Test hardware mailbox with "Hello from GPU OS!"
    mov rsi, msg_testing_mailbox
    call print_string
    call test_hw_mailbox_hello
    call print_ok

    ; Print latency statistics
    extern print_mailbox_stats
    call print_mailbox_stats

    ; System halted message
    mov rsi, msg_gpu_halted
    call print_string
    cli
    hlt

; ---------------------------------------------------------------------------
; init_pat_cpu - Initialize Page Attribute Table
; ---------------------------------------------------------------------------
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

; ---------------------------------------------------------------------------
; pcie_scan - Scan PCIe bus for GPU device
; ---------------------------------------------------------------------------
global pcie_scan
pcie_scan:
    push rax
    push rbx
    push rcx
    push rdx

    xor rbx, rbx

.scan_loop:
    cmp rbx, 32
    jge .scan_done

    ; Read vendor ID
    mov rax, rbx
    shl rax, 15
    or rax, 0x80000000
    mov dx, 0xCF8
    out dx, eax

    mov dx, 0xCFC
    in eax, dx

    cmp ax, 0xFFFF
    je .next_device

    ; Check if VGA controller
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

    ; Read BAR0
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
    mov qword [gpu_bar0_size], 0x1000000

.scan_done:
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret

; ---------------------------------------------------------------------------
; init_bar_mapping - Map GPU BARs with correct cache attributes
; ---------------------------------------------------------------------------
init_bar_mapping:
    push rax
    push rbx
    push rcx
    push rdx

    mov rax, [gpu_bar0]
    test rax, rax
    jz .no_gpu

    ; Map mailbox region as UC
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

    ; Map buffer regions as WC
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

; ---------------------------------------------------------------------------
; test_hw_mailbox_hello - Test hardware mailbox with "Hello" message
; ---------------------------------------------------------------------------
; Uses real hardware mailbox to print "Hello from GPU OS!"
; ---------------------------------------------------------------------------
test_hw_mailbox_hello:
    push rax
    push rbx
    push rcx
    push rdx
    push rsi
    push rdi

    ; Point to hello string
    mov rsi, hello_msg
    xor rcx, rcx          ; Character index

.send_loop:
    ; Get character
    movzx rax, byte [rsi + rcx]
    test al, al
    jz .done

    ; Send via hardware mailbox with latency measurement
    mov rdi, OP_MMIO_WRITE_UART  ; Opcode
    xor rsi, rsi                  ; Thread ID = 0
    movzx rdx, al                 ; Payload = character

    extern mailbox_measure_latency
    call mailbox_measure_latency

    ; Next character
    mov rsi, hello_msg
    inc rcx
    jmp .send_loop

.done:
    ; Send newline
    mov rdi, OP_MMIO_WRITE_UART
    xor rsi, rsi
    mov rdx, 10
    extern mailbox_write_and_wait
    call mailbox_write_and_wait

    pop rdi
    pop rsi
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret

; ---------------------------------------------------------------------------
; Helper Functions
; ---------------------------------------------------------------------------
print_string:
    push rax
.loop:
    lodsb
    test al, al
    jz .done

    ; Write to VGA
    mov rdi, [vga_ptr]
    mov [rdi], al
    inc rdi
    mov byte [rdi], 0x0F
    inc rdi
    mov [vga_ptr], rdi

    ; Write to UART
    push rdx
    mov dx, UART_PORT
    out dx, al
    pop rdx

    jmp .loop

.done:
    pop rax
    ret

print_ok:
    push rsi
    mov rsi, msg_ok
    call print_string
    pop rsi
    ret

; ---------------------------------------------------------------------------
; Data Section
; ---------------------------------------------------------------------------
align 8
msg_microkernel:      db 'pxOS CPU Microkernel v0.5 (Phase 2 Week 2)', 13, 10, 0
msg_init_pat:         db 'Initializing PAT (cache types)... ', 0
msg_scanning_pcie:    db 'Scanning PCIe bus 0... ', 0
msg_mapping_bars:     db 'Mapping GPU BARs (UC/WC)... ', 0
msg_init_hw_mailbox:  db 'Initializing hardware mailbox... ', 0
msg_testing_mailbox:  db 'Testing hardware mailbox... ', 0
msg_gpu_halted:       db 13, 10, 'System halted.', 13, 10, 0
msg_ok:               db 'OK', 13, 10, 0

hello_msg:            db 'Hello from GPU OS!', 0
vga_ptr:              dq 0xB8014

; GPU PCIe information
gpu_device:           dq 0
gpu_bar0:             dq 0
gpu_bar0_size:        dq 0x1000000
gpu_bar2:             dq 0

; 64-bit GDT
align 16
gdt64_start:
    dq 0
    dq 0x00AF9A000000FFFF
    dq 0x00AF92000000FFFF
gdt64_end:

gdt64_descriptor:
    dw gdt64_end - gdt64_start - 1
    dq gdt64_start

; Page tables (aligned)
align 4096
global pml4_table
global pdpt_table
global pd_table
pml4_table: times 4096 db 0
pdpt_table: times 4096 db 0
pd_table:   times 4096 db 0
