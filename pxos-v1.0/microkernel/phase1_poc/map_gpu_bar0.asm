; map_gpu_bar0.asm - GPU BAR0 Memory Mapping for pxOS
; Maps GPU MMIO region into kernel address space with proper caching attributes

BITS 64

; Page table flags
%define PAGE_PRESENT       0x001
%define PAGE_WRITE         0x002
%define PAGE_USER          0x004
%define PAGE_WRITETHROUGH  0x008
%define PAGE_CACHE_DISABLE 0x010
%define PAGE_SIZE_2MB      0x080
%define PAGE_GLOBAL        0x100

; PAT index for uncacheable (UC) memory
%define PAT_UC             0x000

section .data
align 8
global gpu_bar0_phys
global gpu_bar0_virt
global gpu_bar0_size

gpu_bar0_phys: dq 0        ; Set by PCIe scan
gpu_bar0_virt: dq 0        ; Set by mapper
gpu_bar0_size: dq 0x1000000 ; Default 16MB

; Debug messages
msg_map_bar0:      db "Mapping GPU BAR0... ", 0
msg_map_success:   db "OK (virt=0x", 0
msg_map_failed:    db "FAILED (BAR0 not found)", 13, 10, 0
msg_closing_paren: db ")", 13, 10, 0

; Static page tables for MMIO mapping
; BAR0 at ~4GB is outside the boot identity mapping (1GB)
section .bss
align 4096
mmio_pdp:  resb 4096   ; PDP for high memory (entry in PML4)
mmio_pd:   resb 4096   ; PD for BAR0 region (entry in PDP)
mmio_pt:   resb 4096   ; PT for 4KB pages (entry in PD)

section .text
extern serial_print_64
extern serial_putc_64

global map_gpu_bar0
global map_mmio_page

;-----------------------------------------------------------------------------
; map_gpu_bar0()
; Map GPU BAR0 into kernel address space with UC (uncacheable) attributes
; Called after PCIe scan sets gpu_bar0_phys
;-----------------------------------------------------------------------------
map_gpu_bar0:
    push rax
    push rdi
    push rsi
    push rdx
    push rcx

    ; Debug output
    lea rsi, [rel msg_map_bar0]
    call serial_print_64

    ; Check if BAR0 was discovered by PCIe scan
    mov rax, [rel gpu_bar0_phys]
    test rax, rax
    jz .no_bar

    ; For now: identity map BAR0 (phys = virt)
    ; This is simple and works for early bring-up
    mov [rel gpu_bar0_virt], rax

    ; Map first 4KB page (enough for mailbox registers)
    ; For production: map multiple pages to cover full BAR
    mov rdi, rax                    ; phys_addr = BAR0 base
    mov rsi, rax                    ; virt_addr = BAR0 base (identity)
    mov rdx, PAT_UC                 ; Uncacheable for MMIO
    call map_mmio_page

    ; Success - print virtual address
    lea rsi, [rel msg_map_success]
    call serial_print_64

    mov rax, [rel gpu_bar0_virt]
    call print_hex_64

    lea rsi, [rel msg_closing_paren]
    call serial_print_64

    jmp .done

.no_bar:
    ; PCIe scan didn't find GPU or BAR0
    lea rsi, [rel msg_map_failed]
    call serial_print_64

.done:
    pop rcx
    pop rdx
    pop rsi
    pop rdi
    pop rax
    ret

;-----------------------------------------------------------------------------
; map_mmio_page(phys_addr, virt_addr, pat_index)
; Map a single 4KB MMIO page into kernel page tables
; Input:
;   RDI = physical address
;   RSI = virtual address
;   RDX = PAT index (0=UC, 1=WC, etc.)
;-----------------------------------------------------------------------------
map_mmio_page:
    push rax
    push rbx
    push rcx
    push rdx
    push rdi
    push rsi

    ; Align addresses to 4KB boundary
    and rdi, ~0xFFF
    and rsi, ~0xFFF

    ; Get page table indices from virtual address
    mov rax, rsi

    ; PML4 index (bits 39-47)
    mov rbx, rax
    shr rbx, 39
    and rbx, 0x1FF

    ; PDP index (bits 30-38)
    mov rcx, rax
    shr rcx, 30
    and rcx, 0x1FF

    ; Save physical address before we use rax for page table walk
    push r8
    push r9
    push r10
    push r11

    mov r8, rdi                     ; R8 = physical address
    mov r9, rsi                     ; R9 = virtual address (already in rax)

    ;-------------------------------------------------------------------------
    ; Walk page table hierarchy: PML4 → PDP → PD → PT
    ; Implementation based on KernelGuru expert via God Pixel Network
    ;-------------------------------------------------------------------------

    ; Step 1: Get PML4 base from CR3
    mov rax, cr3
    and rax, ~0xFFF

    ; Step 2: Get PDP from PML4[rbx]
    lea rax, [rax + rbx * 8]
    mov r10, rax                    ; Save PML4 entry address
    mov rax, [rax]

    test rax, PAGE_PRESENT
    jnz .pdp_exists

    ; Create PDP using static mmio_pdp
    lea rax, [rel mmio_pdp]
    or rax, PAGE_PRESENT | PAGE_WRITE
    mov [r10], rax

.pdp_exists:
    and rax, ~0xFFF                 ; Extract PDP base

    ; Step 3: Get PD from PDP[rcx]
    lea rax, [rax + rcx * 8]
    mov r10, rax                    ; Save PDP entry address
    mov rax, [rax]

    test rax, PAGE_PRESENT
    jnz .pd_exists

    ; Create PD using static mmio_pd
    lea rax, [rel mmio_pd]
    or rax, PAGE_PRESENT | PAGE_WRITE
    mov [r10], rax

.pd_exists:
    and rax, ~0xFFF                 ; Extract PD base

    ; PD index (bits 21-29) - calculate it now
    mov rdx, r9
    shr rdx, 21
    and rdx, 0x1FF

    ; Step 4: Get PT from PD[rdx]
    lea rax, [rax + rdx * 8]
    mov r10, rax                    ; Save PD entry address
    mov rax, [rax]

    test rax, PAGE_PRESENT
    jnz .pt_exists

    ; Create PT using static mmio_pt
    lea rax, [rel mmio_pt]
    or rax, PAGE_PRESENT | PAGE_WRITE
    mov [r10], rax
    and rax, ~0xFFF
    jmp .install_pte

.pt_exists:
    ; Check for 2MB huge page
    test rax, PAGE_SIZE_2MB
    jnz .done_ok

    and rax, ~0xFFF                 ; Extract PT base

.install_pte:
    ; PT index (bits 12-20)
    mov r11, r9
    shr r11, 12
    and r11, 0x1FF

    ; Step 5: Install PTE at PT[r11]
    lea rax, [rax + r11 * 8]

    ; Build PTE with UC memory type (Present | Writable | PCD | PWT)
    mov r10, r8                     ; Physical address
    or r10, PAGE_PRESENT | PAGE_WRITE | PAGE_CACHE_DISABLE | PAGE_WRITETHROUGH
    mov [rax], r10

    ; Step 6: Flush TLB for this page
    mov rax, r9
    invlpg [rax]

.done_ok:
    pop r11
    pop r10
    pop r9
    pop r8

    pop rsi
    pop rdi
    pop rdx
    pop rcx
    pop rbx
    pop rax
    xor rax, rax                    ; Return success
    ret

;-----------------------------------------------------------------------------
; print_hex_64(value)
; Print 64-bit value in hexadecimal
; Input: RAX = value
;-----------------------------------------------------------------------------
print_hex_64:
    push rax
    push rbx
    push rcx
    push rdx

    mov rbx, rax
    mov rcx, 16         ; 16 hex digits

.loop:
    rol rbx, 4          ; Rotate next nibble into position
    mov rax, rbx
    and rax, 0xF

    ; Convert to ASCII hex digit
    cmp al, 10
    jl .digit
    add al, 'A' - 10
    jmp .output
.digit:
    add al, '0'

.output:
    call serial_putc_64
    loop .loop

    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret
