; bar_map.asm - BAR Memory Mapping Infrastructure for pxOS Phase 2
; Maps GPU MMIO regions with correct cache attributes (UC/WC)

BITS 64

; Cache type constants
PAT_WB  equ 0x06  ; Write-Back (default)
PAT_WT  equ 0x04  ; Write-Through
PAT_UC  equ 0x00  ; Uncacheable (for mailbox)
PAT_WC  equ 0x01  ; Write-Combining (for buffers)

; BAR memory regions (physical addresses from PCIe enumeration)
section .data
bar0_phys:      dq 0  ; Populated by PCIe scan
bar0_size:      dq 0  ; Size in bytes
bar2_phys:      dq 0  ; Control registers
bar2_size:      dq 0

; Virtual addresses (identity-mapped)
bar0_virt:      dq 0
bar2_virt:      dq 0

; Mailbox region (UC - Uncacheable)
MAILBOX_OFFSET  equ 0x0000
MAILBOX_SIZE    equ 0x1000  ; 4KB

; Command buffer region (WC - Write-Combining)
CMDBUF_OFFSET   equ 0x1000
CMDBUF_SIZE     equ 0x1000  ; 4KB

; Pixel program region (WC)
PIXPROG_OFFSET  equ 0x2000
PIXPROG_SIZE    equ 0x10000  ; 64KB

; Framebuffer region (WC)
FB_OFFSET       equ 0x20000
; FB_SIZE = remaining BAR0 space

section .text

; ---------------------------------------------------------------------------
; init_pat - Initialize Page Attribute Table for custom cache types
; ---------------------------------------------------------------------------
; Sets up PAT entries:
;   PAT0 = WB (default)
;   PAT1 = WT
;   PAT2 = UC (for mailbox)
;   PAT3 = UC-
;   PAT4 = WB
;   PAT5 = WT
;   PAT6 = UC
;   PAT7 = WC (for buffers/framebuffer)
; ---------------------------------------------------------------------------
global init_pat
init_pat:
    push rax
    push rcx
    push rdx

    ; Read current PAT MSR (0x277)
    mov ecx, 0x277
    rdmsr

    ; EAX = PAT0-3, EDX = PAT4-7
    ; Current (Intel default):
    ;   PAT0=WB, PAT1=WT, PAT2=UC-, PAT3=UC
    ;   PAT4=WB, PAT5=WT, PAT6=UC-, PAT7=UC

    ; Modify to our layout:
    ;   EAX[7:0]   = PAT0 = 0x06 (WB)
    ;   EAX[15:8]  = PAT1 = 0x04 (WT)
    ;   EAX[23:16] = PAT2 = 0x00 (UC) ← mailbox
    ;   EAX[31:24] = PAT3 = 0x00 (UC-)
    ;   EDX[7:0]   = PAT4 = 0x06 (WB)
    ;   EDX[15:8]  = PAT5 = 0x04 (WT)
    ;   EDX[23:16] = PAT6 = 0x00 (UC)
    ;   EDX[31:24] = PAT7 = 0x01 (WC) ← buffers

    mov eax, 0x00000406  ; PAT3=UC, PAT2=UC, PAT1=WT, PAT0=WB
    mov edx, 0x01000406  ; PAT7=WC, PAT6=UC, PAT5=WT, PAT4=WB

    ; Write back to PAT MSR
    wrmsr

    ; VGA marker: 'M' for PAT initialized
    mov byte [0xB8014], 'M'
    mov byte [0xB8015], 0x0A  ; Green

    pop rdx
    pop rcx
    pop rax
    ret

; ---------------------------------------------------------------------------
; set_bar_addresses - Store BAR physical addresses from PCIe scan
; ---------------------------------------------------------------------------
; Input:
;   RDI = BAR0 physical address
;   RSI = BAR0 size
;   RDX = BAR2 physical address
;   RCX = BAR2 size
; ---------------------------------------------------------------------------
global set_bar_addresses
set_bar_addresses:
    mov [bar0_phys], rdi
    mov [bar0_size], rsi
    mov [bar2_phys], rdx
    mov [bar2_size], rcx

    ; VGA marker: 'B' for BAR addresses set
    mov byte [0xB8016], 'B'
    mov byte [0xB8017], 0x0A  ; Green

    ret

; ---------------------------------------------------------------------------
; map_bar0_mailbox - Map mailbox region as UC (Uncacheable)
; ---------------------------------------------------------------------------
; Maps BAR0 + 0x0000 (4KB) with Uncacheable attribute
; This ensures CPU-GPU synchronization works correctly
; ---------------------------------------------------------------------------
global map_bar0_mailbox
map_bar0_mailbox:
    push rax
    push rbx
    push rcx
    push rdx

    ; Get BAR0 physical address
    mov rax, [bar0_phys]
    test rax, rax
    jz .error

    ; Calculate PTE address for this mapping
    ; For identity mapping, we update existing page tables
    ; We need to find the PTE for BAR0 address

    ; Simplified: Assume we're using 2MB pages
    ; Find PD entry index: (BAR0 >> 21) & 0x1FF
    mov rbx, rax
    shr rbx, 21
    and rbx, 0x1FF

    ; Get PD table base (from earlier setup)
    extern pd_table
    mov rcx, pd_table

    ; Calculate PTE address: PD + (index * 8)
    shl rbx, 3
    add rcx, rbx

    ; Build PTE value with UC attribute
    ; Present | RW | PS (2MB) | PAT2 (UC)
    ; Flags: bit 0=P, bit 1=RW, bit 7=PS, bit 12=PAT
    mov rdx, rax
    and rdx, 0xFFFFFFFFFFFFF000  ; Clear low 12 bits
    or rdx, 0x83                  ; P=1, RW=1, PS=1
    or rdx, (1 << 12)             ; PAT bit for PAT2

    ; Write PTE
    mov [rcx], rdx

    ; Flush TLB for this address
    invlpg [rax]

    ; Store virtual address (identity-mapped)
    mov [bar0_virt], rax

    ; VGA marker: 'U' for UC mailbox mapped
    mov byte [0xB8018], 'U'
    mov byte [0xB8019], 0x0A  ; Green

    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret

.error:
    ; VGA marker: 'E' for error
    mov byte [0xB8018], 'E'
    mov byte [0xB8019], 0x0C  ; Red
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret

; ---------------------------------------------------------------------------
; map_bar0_buffers - Map command buffer/framebuffer as WC (Write-Combining)
; ---------------------------------------------------------------------------
; Maps BAR0 + 0x1000 onwards with Write-Combining attribute
; This improves performance for sequential writes
; ---------------------------------------------------------------------------
global map_bar0_buffers
map_bar0_buffers:
    push rax
    push rbx
    push rcx
    push rdx
    push rsi

    ; Get BAR0 physical address
    mov rax, [bar0_phys]
    test rax, rax
    jz .error

    ; Add offset for buffer region (starts at 0x1000)
    add rax, CMDBUF_OFFSET

    ; Calculate number of 2MB pages needed
    ; For simplicity, map 16MB (8 pages)
    mov rsi, 8  ; Number of pages

.map_loop:
    ; Calculate PD entry index
    mov rbx, rax
    shr rbx, 21
    and rbx, 0x1FF

    ; Get PD table base
    extern pd_table
    mov rcx, pd_table
    shl rbx, 3
    add rcx, rbx

    ; Build PTE with WC attribute
    ; Present | RW | PS | PAT7 (WC)
    mov rdx, rax
    and rdx, 0xFFFFFFFFFFE00000  ; Align to 2MB
    or rdx, 0x83                  ; P=1, RW=1, PS=1
    or rdx, (1 << 7)              ; PAT bit 0
    or rdx, (1 << 12)             ; PAT bit 1 (PAT7)

    ; Write PTE
    mov [rcx], rdx

    ; Flush TLB
    invlpg [rax]

    ; Next 2MB page
    add rax, 0x200000
    dec rsi
    jnz .map_loop

    ; VGA marker: 'W' for WC buffers mapped
    mov byte [0xB801A], 'W'
    mov byte [0xB801B], 0x0A  ; Green

    pop rsi
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret

.error:
    ; VGA marker: 'E' for error
    mov byte [0xB801A], 'E'
    mov byte [0xB801B], 0x0C  ; Red
    pop rsi
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret

; ---------------------------------------------------------------------------
; get_mailbox_addr - Get virtual address of mailbox
; ---------------------------------------------------------------------------
; Output: RAX = mailbox virtual address
; ---------------------------------------------------------------------------
global get_mailbox_addr
get_mailbox_addr:
    mov rax, [bar0_virt]
    add rax, MAILBOX_OFFSET
    ret

; ---------------------------------------------------------------------------
; get_cmdbuf_addr - Get virtual address of command buffer
; ---------------------------------------------------------------------------
; Output: RAX = command buffer virtual address
; ---------------------------------------------------------------------------
global get_cmdbuf_addr
get_cmdbuf_addr:
    mov rax, [bar0_virt]
    add rax, CMDBUF_OFFSET
    ret

; ---------------------------------------------------------------------------
; get_pixprog_addr - Get virtual address of pixel program region
; ---------------------------------------------------------------------------
; Output: RAX = pixel program virtual address
; ---------------------------------------------------------------------------
global get_pixprog_addr
get_pixprog_addr:
    mov rax, [bar0_virt]
    add rax, PIXPROG_OFFSET
    ret

; ---------------------------------------------------------------------------
; get_framebuffer_addr - Get virtual address of framebuffer
; ---------------------------------------------------------------------------
; Output: RAX = framebuffer virtual address
; ---------------------------------------------------------------------------
global get_framebuffer_addr
get_framebuffer_addr:
    mov rax, [bar0_virt]
    add rax, FB_OFFSET
    ret

; ---------------------------------------------------------------------------
; dump_bar_info - Debug function to print BAR info
; ---------------------------------------------------------------------------
global dump_bar_info
dump_bar_info:
    push rsi

    ; Print "BAR0:" message
    mov rsi, msg_bar0
    extern print_string
    call print_string

    ; Print BAR0 physical address (would need hex print function)
    mov rax, [bar0_phys]
    ; call print_hex  ; TODO: implement

    ; Print " Size:" message
    mov rsi, msg_size
    call print_string

    ; Print BAR0 size
    mov rax, [bar0_size]
    ; call print_hex  ; TODO: implement

    pop rsi
    ret

section .rodata
msg_bar0:       db 'BAR0: 0x', 0
msg_size:       db ' Size: 0x', 0
msg_mailbox:    db ' Mailbox: 0x', 0
msg_cmdbuf:     db ' CmdBuf: 0x', 0
