; pxHV Stage 2 - Hypervisor Loader
; Transitions to long mode and enables VT-x
;
; Entry: Real mode, loaded at 0x10000
; Exit: Long mode, VT-x enabled, jumps to stage3
;
; Compile: nasm -f bin -o pxhv_stage2.bin pxhv_stage2.asm

BITS 16
ORG 0x10000

; Memory layout constants
PML4_ADDR       equ 0x70000     ; Page Map Level 4 (host)
PDPT_ADDR       equ 0x71000     ; Page Directory Pointer Table (host)
PD_ADDR         equ 0x72000     ; Page Directory (host)
PT_ADDR         equ 0x73000     ; Page Table (host)

EPT_PML4        equ 0x80000     ; EPT PML4 (guest memory virtualization)
EPT_PDPT        equ 0x81000     ; EPT PDPT
EPT_PD          equ 0x82000     ; EPT PD

VMXON_REGION    equ 0x15000     ; VMXON region (4KB aligned)
VMCS_REGION     equ 0x16000     ; VMCS region (4KB aligned)
STACK_TOP       equ 0x9F000     ; Stack at ~640KB

; Virtual disk image location
DISK_IMAGE      equ 0x100000    ; Virtual disk at 1MB (1MB = 1048576 bytes)

stage2_entry:
    ; Setup segments
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x8000

    ; Print stage 2 message
    mov si, msg_stage2
    call print_string

    ; Check for long mode support
    call check_long_mode
    jc .no_long_mode

    ; Check for VT-x support
    call check_vmx
    jc .no_vmx

    ; Setup page tables for long mode
    call setup_page_tables

    ; Enable paging and long mode
    call enter_long_mode

    ; We should never get here
    cli
    hlt

.no_long_mode:
    mov si, msg_no_long_mode
    call print_string
    cli
    hlt

.no_vmx:
    mov si, msg_no_vmx
    call print_string
    cli
    hlt

;-----------------------------------------------------------------------------
; check_long_mode: Check if CPU supports 64-bit long mode
;-----------------------------------------------------------------------------
check_long_mode:
    pusha

    ; Check if CPUID is supported
    pushfd
    pop eax
    mov ecx, eax
    xor eax, 0x200000           ; Flip ID bit
    push eax
    popfd
    pushfd
    pop eax
    xor eax, ecx
    jz .no_cpuid

    ; Check for extended CPUID
    mov eax, 0x80000000
    cpuid
    cmp eax, 0x80000001
    jb .no_long_mode

    ; Check for long mode support
    mov eax, 0x80000001
    cpuid
    test edx, (1 << 29)         ; LM bit
    jz .no_long_mode

    popa
    clc
    ret

.no_cpuid:
.no_long_mode:
    popa
    stc
    ret

;-----------------------------------------------------------------------------
; check_vmx: Check if CPU supports Intel VT-x
;-----------------------------------------------------------------------------
check_vmx:
    pusha

    ; Check CPUID.1:ECX.VMX[bit 5]
    mov eax, 1
    cpuid
    test ecx, (1 << 5)
    jz .no_vmx

    ; Print VMX supported message
    mov si, msg_vmx_ok
    call print_string

    popa
    clc
    ret

.no_vmx:
    popa
    stc
    ret

;-----------------------------------------------------------------------------
; setup_page_tables: Create identity-mapped page tables for first 2MB
;-----------------------------------------------------------------------------
setup_page_tables:
    pusha

    ; Clear page table memory
    mov edi, PML4_ADDR
    mov ecx, 0x4000             ; 16KB (4 pages)
    xor eax, eax
    rep stosd

    ; Setup PML4 (maps to PDPT)
    mov edi, PML4_ADDR
    mov dword [edi], PDPT_ADDR | 3  ; Present, writable

    ; Setup PDPT (maps to PD)
    mov edi, PDPT_ADDR
    mov dword [edi], PD_ADDR | 3    ; Present, writable

    ; Setup PD with 2MB pages (identity mapped)
    ; This maps first 1GB with 2MB pages
    mov edi, PD_ADDR
    mov eax, 0x83                   ; Present, writable, 2MB page
    mov ecx, 512                    ; 512 entries = 1GB
.pd_loop:
    mov [edi], eax
    add eax, 0x200000               ; Next 2MB
    add edi, 8
    loop .pd_loop

    popa
    ret

;-----------------------------------------------------------------------------
; enter_long_mode: Enable PAE, long mode, and paging
;-----------------------------------------------------------------------------
enter_long_mode:
    ; Disable interrupts
    cli

    ; Load GDT
    lgdt [gdt_descriptor]

    ; Enable PAE (Physical Address Extension)
    mov eax, cr4
    or eax, (1 << 5)                ; Set PAE bit
    mov cr4, eax

    ; Load PML4 address
    mov eax, PML4_ADDR
    mov cr3, eax

    ; Enable long mode (set EFER.LME)
    mov ecx, 0xC0000080             ; EFER MSR
    rdmsr
    or eax, (1 << 8)                ; Set LME bit
    wrmsr

    ; Enable paging and protected mode
    mov eax, cr0
    or eax, (1 << 31) | (1 << 0)   ; Set PG and PE
    mov cr0, eax

    ; Jump to 64-bit code segment
    jmp 0x08:long_mode_entry

;-----------------------------------------------------------------------------
; print_string: Print null-terminated string (16-bit real mode)
;-----------------------------------------------------------------------------
print_string:
    pusha
    mov ah, 0x0E
.loop:
    lodsb
    test al, al
    jz .done
    int 0x10
    jmp .loop
.done:
    popa
    ret

;-----------------------------------------------------------------------------
; GDT for long mode
;-----------------------------------------------------------------------------
align 8
gdt_start:
    ; Null descriptor
    dq 0x0000000000000000

    ; 64-bit code segment
    dq 0x00AF9A000000FFFF          ; Base=0, Limit=0xFFFFF, Present, 64-bit, Code

    ; 64-bit data segment
    dq 0x00CF92000000FFFF          ; Base=0, Limit=0xFFFFF, Present, 64-bit, Data

gdt_end:

gdt_descriptor:
    dw gdt_end - gdt_start - 1     ; GDT size
    dd gdt_start                    ; GDT address

;-----------------------------------------------------------------------------
; 64-bit code (long mode entry point)
;-----------------------------------------------------------------------------
BITS 64
long_mode_entry:
    ; Setup segment registers
    mov ax, 0x10                    ; Data segment selector
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax

    ; Setup stack
    mov rsp, STACK_TOP

    ; Print long mode message
    mov rsi, msg_long_mode
    call print_string_64

    ; Enable VT-x
    call enable_vmx

    ; Initialize VMXON region
    call vmxon_init

    ; Execute VMXON
    call vmxon_exec

    ; If we get here, VMXON succeeded!
    mov rsi, msg_vmxon_ok
    call print_string_64

    ; Jump to Stage 3: VMCS setup and guest launch
    call setup_vmcs_and_guest

    ; If we return here, something failed
    mov rsi, msg_stage3_fail
    call print_string_64
    cli
    hlt

;-----------------------------------------------------------------------------
; enable_vmx: Enable VT-x by setting CR4.VMXE
;-----------------------------------------------------------------------------
enable_vmx:
    ; Set CR4.VMXE (bit 13)
    mov rax, cr4
    or rax, (1 << 13)
    mov cr4, rax

    ; Configure IA32_FEATURE_CONTROL MSR
    mov ecx, 0x3A
    rdmsr
    test eax, 1                     ; Check if locked
    jnz .already_configured

    ; Lock bit + enable VMX outside SMX
    or eax, 5
    wrmsr

.already_configured:
    ret

;-----------------------------------------------------------------------------
; vmxon_init: Initialize VMXON region
;-----------------------------------------------------------------------------
vmxon_init:
    ; Clear VMXON region
    mov rdi, VMXON_REGION
    mov rcx, 1024                   ; 4KB = 1024 qwords
    xor rax, rax
    rep stosq

    ; Read VMCS revision ID from IA32_VMX_BASIC MSR
    mov ecx, 0x480
    rdmsr

    ; Write revision ID to first 4 bytes of VMXON region
    mov [VMXON_REGION], eax

    ret

;-----------------------------------------------------------------------------
; vmxon_exec: Execute VMXON instruction
;-----------------------------------------------------------------------------
vmxon_exec:
    ; VMXON [VMXON_REGION]
    mov rax, VMXON_REGION
    vmxon [rax]

    jc .vmxon_failed                ; CF=1: VMfailInvalid
    jz .vmxon_failed                ; ZF=1: VMfailValid

    ret

.vmxon_failed:
    mov rsi, msg_vmxon_fail
    call print_string_64
    cli
    hlt

;-----------------------------------------------------------------------------
; setup_ept_tables: Create Extended Page Tables for guest memory
; Maps first 1GB of guest physical memory to host physical memory (identity mapped)
; Uses 2MB pages for simplicity
;-----------------------------------------------------------------------------
setup_ept_tables:
    push rax
    push rcx
    push rdi

    ; 1) Clear EPT table memory (12KB total: PML4 + PDPT + PD)
    mov rdi, EPT_PML4
    mov rcx, 3072               ; 12KB / 8 bytes = 1536 qwords
    xor rax, rax
    rep stosq

    ; 2) Setup EPT PML4 (points to PDPT)
    mov rdi, EPT_PML4
    mov rax, EPT_PDPT
    or rax, 0x7                 ; Read, Write, Execute permissions
    mov [rdi], rax

    ; 3) Setup EPT PDPT (points to PD)
    mov rdi, EPT_PDPT
    mov rax, EPT_PD
    or rax, 0x7                 ; Read, Write, Execute permissions
    mov [rdi], rax

    ; 4) Setup EPT PD with 2MB pages (identity map first 1GB)
    mov rdi, EPT_PD
    mov rax, 0x87               ; Read, Write, Execute + 2MB page
    mov rcx, 512                ; 512 entries * 2MB = 1GB

.pd_loop:
    mov [rdi], rax
    add rax, 0x200000           ; Next 2MB
    add rdi, 8
    loop .pd_loop

    ; EPT is now ready
    pop rdi
    pop rcx
    pop rax
    ret

;-----------------------------------------------------------------------------
; load_guest_code: Copy guest real mode code to memory at 0x7C00
; The guest code is embedded at the end of this binary
;-----------------------------------------------------------------------------
load_guest_code:
    push rsi
    push rdi
    push rcx

    ; Source: embedded guest binary
    mov rsi, guest_code_data
    ; Destination: 0x7C00 (guest memory)
    mov rdi, 0x7C00
    ; Length: 512 bytes (one sector)
    mov rcx, 512
    ; Copy byte by byte
    rep movsb

    pop rcx
    pop rdi
    pop rsi
    ret

;-----------------------------------------------------------------------------
; setup_ivt_bda: Initialize IVT (Interrupt Vector Table) and BDA (BIOS Data Area)
; IVT: 0x0000-0x03FF (1024 bytes, 256 vectors * 4 bytes each)
; BDA: 0x0400-0x04FF (256 bytes)
;-----------------------------------------------------------------------------
setup_ivt_bda:
    push rax
    push rbx
    push rcx
    push rdx
    push rdi

    ; -----------------------------------------------------------------------
    ; 1) Initialize IVT (0x0000-0x03FF)
    ; -----------------------------------------------------------------------
    ; For now, we'll set all vectors to point to a dummy IRET handler
    ; Our hypervisor traps INT 3, 10h, 13h, 16h, 19h via exception bitmap
    ; So these vectors won't actually be used, but DOS expects them present

    ; First, install a dummy IRET handler at 0xF000:0xFFF0
    ; (This is a safe location in the BIOS area)
    mov byte [0xFFFF0], 0xCF        ; IRET instruction

    ; Now fill IVT with vectors pointing to F000:FFF0
    mov rdi, 0x0000                 ; Start of IVT
    mov cx, 256                     ; 256 interrupt vectors

.ivt_loop:
    mov word [rdi], 0xFFF0          ; Offset = 0xFFF0
    mov word [rdi+2], 0xF000        ; Segment = 0xF000
    add rdi, 4
    loop .ivt_loop

    ; -----------------------------------------------------------------------
    ; 2) Initialize BDA (0x0400-0x04FF)
    ; -----------------------------------------------------------------------

    ; Clear entire BDA first
    mov rdi, 0x0400
    mov rcx, 64                     ; 256 bytes / 4 = 64 dwords
    xor eax, eax
    rep stosd

    ; Now set specific BDA fields

    ; COM port addresses (0x0400-0x0407)
    mov word [0x0400], 0x03F8       ; COM1 = 0x3F8
    mov word [0x0402], 0x02F8       ; COM2 = 0x2F8
    mov word [0x0404], 0x03E8       ; COM3 = 0x3E8
    mov word [0x0406], 0x02E8       ; COM4 = 0x2E8

    ; LPT port addresses (0x0408-0x040F)
    mov word [0x0408], 0x0378       ; LPT1 = 0x378
    mov word [0x040A], 0x0278       ; LPT2 = 0x278

    ; Equipment word (0x0410)
    ; Bit 0: Floppy installed
    ; Bit 1: Math coprocessor
    ; Bits 4-5: Initial video mode (01 = 80x25 color)
    ; Bits 6-7: Number of floppies - 1 (00 = 1 floppy)
    mov word [0x0410], 0x0021       ; Floppy + 80x25 color text

    ; Memory size in KB (0x0413)
    mov word [0x0413], 640          ; 640 KB conventional memory

    ; Keyboard flags (0x0417)
    mov byte [0x0417], 0x00         ; No keys pressed

    ; Video mode (0x0449)
    mov byte [0x0449], 0x03         ; Mode 3: 80x25 color text

    ; Screen columns (0x044A)
    mov word [0x044A], 80           ; 80 columns

    ; Video page size (0x044C)
    mov word [0x044C], 4000         ; 80*25*2 = 4000 bytes

    ; Current video page offset (0x044E)
    mov word [0x044E], 0            ; Page 0

    ; Cursor positions for 8 pages (0x0450-0x045F)
    mov rdi, 0x0450
    mov rcx, 8
    xor ax, ax
.cursor_loop:
    mov word [rdi], ax              ; Row 0, Col 0
    add rdi, 2
    loop .cursor_loop

    ; Active display page (0x0462)
    mov byte [0x0462], 0            ; Page 0

    ; Video controller base (0x0463)
    mov word [0x0463], 0x03D4       ; CRT controller for color

    ; Number of rows minus 1 (0x0484)
    mov byte [0x0484], 24           ; 25 rows - 1

    pop rdi
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret

;-----------------------------------------------------------------------------
; print_string_64: Print null-terminated string (64-bit mode)
; Input: RSI = string address
; Uses direct VGA text mode writes (0xB8000)
;-----------------------------------------------------------------------------
print_string_64:
    push rax
    push rbx
    push rcx

    mov rbx, 0xB8000                ; VGA text buffer
    mov ah, 0x0F                    ; White on black

.loop:
    lodsb
    test al, al
    jz .done
    mov [rbx], ax
    add rbx, 2
    jmp .loop

.done:
    pop rcx
    pop rbx
    pop rax
    ret

;-----------------------------------------------------------------------------
; setup_vmcs_and_guest: Complete VMCS setup for minimal HLT guest
;-----------------------------------------------------------------------------
setup_vmcs_and_guest:
    push rbx
    push rcx
    push rdx

    ; 1) Clear VMCS region
    mov rdi, VMCS_REGION
    mov rcx, 512                ; 4KB = 512 qwords
    xor rax, rax
    rep stosq

    ; 2) Write VMCS revision ID
    mov ecx, 0x480              ; IA32_VMX_BASIC MSR
    rdmsr
    and eax, 0x7FFFFFFF         ; Clear bit 31
    mov [VMCS_REGION], eax

    ; 3) Load VMCS pointer
    mov rax, VMCS_REGION
    vmptrld [rax]
    jc .error
    jz .error

    ; 4) Install guest code: HLT at 0x200000
    mov byte [0x200000], 0xF4   ; HLT opcode

    ; Print setup message
    mov rsi, msg_vmcs_setup
    call print_string_64

    ; 4.5) Setup EPT (Extended Page Tables)
    call setup_ept_tables

    ; 4.6) Setup IVT and BDA for DOS compatibility
    call setup_ivt_bda

    ; 4.7) Setup boot vector - Place INT 19h instruction at BIOS reset vector
    ; Real PCs boot from 0xF000:0xFFF0 (physical 0xFFFF0)
    ; We'll place INT 19h there to trigger the bootstrap loader
    mov byte [0xFFFF0], 0xCD    ; INT instruction opcode
    mov byte [0xFFFF1], 0x19    ; INT 19h (bootstrap loader)
    mov byte [0xFFFF2], 0xF4    ; HLT (in case we return)
    mov byte [0xFFFF3], 0xEB    ; JMP short
    mov byte [0xFFFF4], 0xFD    ; Jump to self (0xFFFF3)

    ; 4.8) Load guest real mode code to memory at 0x7C00 (for reference/testing)
    ; This will be overwritten by INT 19h when it loads the boot sector
    call load_guest_code

    ; ---------------------------------------------------------------------
    ; 5) Configure GUEST STATE (Real Mode 16-bit)
    ; ---------------------------------------------------------------------

    ; Guest CR0 (Real mode: PE=0, NE=1, ET=1)
    mov rax, 0x30               ; No paging, no protection, numerics enabled
    mov rdx, 0x6800             ; GUEST_CR0
    vmwrite rdx, rax
    jc .error

    ; Guest CR3 (not used in real mode, but set to 0)
    xor rax, rax
    mov rdx, 0x6802             ; GUEST_CR3
    vmwrite rdx, rax
    jc .error

    ; Guest CR4 (minimal, no VMX/PAE/PSE)
    mov rax, 0x2000             ; Enable VMXE for safety
    mov rdx, 0x6804             ; GUEST_CR4
    vmwrite rdx, rax
    jc .error

    ; Guest RIP = 0xFFF0 (BIOS reset vector - where INT 19h is placed)
    mov rax, 0xFFF0
    mov rdx, 0x681E             ; GUEST_RIP
    vmwrite rdx, rax
    jc .error

    ; Guest RSP = 0x7C00 (stack before code)
    mov rax, 0x7C00
    mov rdx, 0x681C             ; GUEST_RSP
    vmwrite rdx, rax
    jc .error

    ; Guest RFLAGS = 0x2 (reserved bit, IF will be set later)
    mov rax, 0x2
    mov rdx, 0x6820             ; GUEST_RFLAGS
    vmwrite rdx, rax
    jc .error

    ; Guest CS selector (real mode segment)
    mov rax, 0xF000             ; CS = 0xF000 (BIOS segment)
    mov rdx, 0x802              ; GUEST_CS_SELECTOR
    vmwrite rdx, rax
    jc .error

    ; Guest CS base (segment * 16)
    mov rax, 0xF0000            ; 0xF000 * 16 = 0xF0000
    mov rdx, 0x6808             ; GUEST_CS_BASE
    vmwrite rdx, rax
    jc .error

    ; Guest CS limit (64KB for real mode)
    mov rax, 0xFFFF
    mov rdx, 0x4800             ; GUEST_CS_LIMIT
    vmwrite rdx, rax
    jc .error

    ; Guest CS access rights (real mode: present, readable, accessed)
    mov rax, 0x0093             ; P=1, DPL=0, S=1, Type=3 (readable code)
    mov rdx, 0x4816             ; GUEST_CS_AR_BYTES
    vmwrite rdx, rax
    jc .error

    ; Guest DS selector (real mode)
    xor rax, rax
    mov rdx, 0x804              ; GUEST_DS_SELECTOR
    vmwrite rdx, rax
    jc .error

    ; Guest ES selector (real mode segment for VGA: 0xB800)
    mov rax, 0xB800             ; VGA text segment
    mov rdx, 0x806              ; GUEST_ES_SELECTOR
    vmwrite rdx, rax
    jc .error

    ; Guest SS selector (real mode)
    xor rax, rax
    mov rdx, 0x808              ; GUEST_SS_SELECTOR
    vmwrite rdx, rax
    jc .error

    ; Guest DS base = 0
    xor rax, rax
    mov rdx, 0x6804             ; GUEST_DS_BASE
    vmwrite rdx, rax
    jc .error

    ; Guest ES base = 0xB8000 (VGA text memory)
    mov rax, 0xB8000
    mov rdx, 0x6806             ; GUEST_ES_BASE
    vmwrite rdx, rax
    jc .error

    ; Guest SS base = 0
    xor rax, rax
    mov rdx, 0x6810             ; GUEST_SS_BASE
    vmwrite rdx, rax
    jc .error

    ; Guest DS/ES/SS limit (64KB for real mode)
    mov rax, 0xFFFF
    mov rdx, 0x4802             ; GUEST_DS_LIMIT
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x4804             ; GUEST_ES_LIMIT
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x4818             ; GUEST_SS_LIMIT
    vmwrite rdx, rax
    jc .error

    ; Guest DS/ES/SS access rights (real mode: present, writable data)
    mov rax, 0x0093             ; P=1, DPL=0, S=1, Type=3 (writable data)
    mov rdx, 0x4814             ; GUEST_DS_AR_BYTES
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x4802             ; GUEST_ES_AR_BYTES
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x481A             ; GUEST_SS_AR_BYTES
    vmwrite rdx, rax
    jc .error

    ; Guest FS selector (real mode)
    xor rax, rax
    mov rdx, 0x80A              ; GUEST_FS_SELECTOR
    vmwrite rdx, rax
    jc .error

    ; Guest GS selector (real mode)
    mov rdx, 0x80C              ; GUEST_GS_SELECTOR
    vmwrite rdx, rax
    jc .error

    ; Guest FS base
    mov rdx, 0x680A             ; GUEST_FS_BASE
    vmwrite rdx, rax
    jc .error

    ; Guest GS base
    mov rdx, 0x680C             ; GUEST_GS_BASE
    vmwrite rdx, rax
    jc .error

    ; Guest FS/GS limit
    mov rax, 0xFFFF
    mov rdx, 0x4806             ; GUEST_FS_LIMIT
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x4808             ; GUEST_GS_LIMIT
    vmwrite rdx, rax
    jc .error

    ; Guest FS/GS access rights (real mode data)
    mov rax, 0x0093
    mov rdx, 0x4810             ; GUEST_FS_AR_BYTES
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x4812             ; GUEST_GS_AR_BYTES
    vmwrite rdx, rax
    jc .error

    ; Guest GDTR
    mov rax, gdt_start
    mov rdx, 0x6816             ; GUEST_GDTR_BASE
    vmwrite rdx, rax
    jc .error
    mov rax, gdt_end - gdt_start - 1
    mov rdx, 0x4810             ; GUEST_GDTR_LIMIT
    vmwrite rdx, rax
    jc .error

    ; Guest IDTR (points to IVT in real mode)
    xor rax, rax
    mov rdx, 0x6818             ; GUEST_IDTR_BASE (0x0000 for real mode IVT)
    vmwrite rdx, rax
    jc .error
    mov rax, 0x3FF              ; IDTR limit for 256 vectors (0x000-0x3FF)
    mov rdx, 0x4812             ; GUEST_IDTR_LIMIT
    vmwrite rdx, rax
    jc .error

    ; Guest LDTR (null)
    xor rax, rax
    mov rdx, 0x80E              ; GUEST_LDTR_SELECTOR
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x680E             ; GUEST_LDTR_BASE
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x480A             ; GUEST_LDTR_LIMIT
    vmwrite rdx, rax
    jc .error
    mov rax, 0x10000            ; Unusable
    mov rdx, 0x480C             ; GUEST_LDTR_AR_BYTES
    vmwrite rdx, rax
    jc .error

    ; Guest TR (null but usable)
    xor rax, rax
    mov rdx, 0x80E              ; GUEST_TR_SELECTOR
    vmwrite rdx, rax
    jc .error
    mov rdx, 0x6814             ; GUEST_TR_BASE
    vmwrite rdx, rax
    jc .error
    mov rax, 0xFFFF
    mov rdx, 0x480E             ; GUEST_TR_LIMIT
    vmwrite rdx, rax
    jc .error
    mov rax, 0x8B               ; Present, TSS
    mov rdx, 0x4822             ; GUEST_TR_AR_BYTES
    vmwrite rdx, rax
    jc .error

    ; Guest IA32_DEBUGCTL
    xor rax, rax
    xor rdx, rdx
    mov rcx, 0x2802             ; GUEST_IA32_DEBUGCTL
    vmwrite rcx, rax
    jc .error
    vmwrite rcx, rdx
    jc .error

    ; Guest interruptibility state
    xor rax, rax
    mov rdx, 0x4824             ; GUEST_INTERRUPTIBILITY_INFO
    vmwrite rdx, rax
    jc .error

    ; Guest activity state (active)
    xor rax, rax
    mov rdx, 0x4826             ; GUEST_ACTIVITY_STATE
    vmwrite rdx, rax
    jc .error

    ; Guest pending debug exceptions
    xor rax, rax
    mov rdx, 0x6822             ; GUEST_PENDING_DBG_EXCEPTIONS
    vmwrite rdx, rax
    jc .error

    ; Guest VMCS link pointer (no shadowing)
    mov rax, 0xFFFFFFFFFFFFFFFF
    mov rdx, 0x2800             ; VMCS_LINK_POINTER
    vmwrite rdx, rax
    jc .error

    ; ---------------------------------------------------------------------
    ; 6) Configure HOST STATE
    ; ---------------------------------------------------------------------

    ; Host CR0
    mov rax, cr0
    mov rdx, 0x6C00             ; HOST_CR0
    vmwrite rdx, rax
    jc .error

    ; Host CR3
    mov rax, cr3
    mov rdx, 0x6C02             ; HOST_CR3
    vmwrite rdx, rax
    jc .error

    ; Host CR4
    mov rax, cr4
    mov rdx, 0x6C04             ; HOST_CR4
    vmwrite rdx, rax
    jc .error

    ; Host RIP = vm_exit_handler
    mov rax, vm_exit_handler
    mov rdx, 0x6C16             ; HOST_RIP
    vmwrite rdx, rax
    jc .error

    ; Host RSP
    mov rax, rsp
    mov rdx, 0x6C14             ; HOST_RSP
    vmwrite rdx, rax
    jc .error

    ; Host CS selector
    mov rax, 0x08
    mov rdx, 0x0C02             ; HOST_CS_SELECTOR
    vmwrite rdx, rax
    jc .error

    ; Host SS selector
    mov rax, 0x10
    mov rdx, 0x0C04             ; HOST_SS_SELECTOR
    vmwrite rdx, rax
    jc .error

    ; Host DS selector
    mov rdx, 0x0C06             ; HOST_DS_SELECTOR
    vmwrite rdx, rax
    jc .error

    ; Host ES selector
    mov rdx, 0x0C08             ; HOST_ES_SELECTOR
    vmwrite rdx, rax
    jc .error

    ; Host FS selector
    mov rdx, 0x0C0A             ; HOST_FS_SELECTOR
    vmwrite rdx, rax
    jc .error

    ; Host GS selector
    mov rdx, 0x0C0C             ; HOST_GS_SELECTOR
    vmwrite rdx, rax
    jc .error

    ; Host TR selector (null for now)
    xor rax, rax
    mov rdx, 0x0C0E             ; HOST_TR_SELECTOR
    vmwrite rdx, rax
    jc .error

    ; Host FS base
    xor rax, rax
    mov rdx, 0x6C06             ; HOST_FS_BASE
    vmwrite rdx, rax
    jc .error

    ; Host GS base
    mov rdx, 0x6C08             ; HOST_GS_BASE
    vmwrite rdx, rax
    jc .error

    ; Host TR base
    mov rdx, 0x6C0A             ; HOST_TR_BASE
    vmwrite rdx, rax
    jc .error

    ; Host GDTR base
    mov rax, gdt_start
    mov rdx, 0x6C0C             ; HOST_GDTR_BASE
    vmwrite rdx, rax
    jc .error

    ; Host IDTR base (null for now)
    xor rax, rax
    mov rdx, 0x6C0E             ; HOST_IDTR_BASE
    vmwrite rdx, rax
    jc .error

    ; ---------------------------------------------------------------------
    ; 7) Configure EXECUTION CONTROLS
    ; ---------------------------------------------------------------------

    ; Pin-based VM-execution controls
    mov ecx, 0x481              ; IA32_VMX_TRUE_PINBASED_CTLS
    rdmsr
    ; Use allowed0 in EAX and allowed1 in EDX
    and eax, edx                ; Allowed bits
    mov rdx, 0x4000             ; PIN_BASED_VM_EXEC_CONTROL
    vmwrite rdx, rax
    jc .error

    ; Primary processor-based VM-execution controls
    mov ecx, 0x482              ; IA32_VMX_TRUE_PROCBASED_CTLS
    rdmsr
    or eax, (1 << 7)            ; HLT exiting
    or eax, (1 << 24)           ; Unconditional I/O exiting
    or eax, (1 << 31)           ; Activate secondary controls
    mov rdx, 0x4002             ; CPU_BASED_VM_EXEC_CONTROL
    vmwrite rdx, rax
    jc .error

    ; Secondary processor-based VM-execution controls (for EPT + unrestricted guest)
    mov ecx, 0x48B              ; IA32_VMX_PROCBASED_CTLS2
    rdmsr
    or eax, (1 << 1)            ; Enable EPT
    or eax, (1 << 7)            ; Enable unrestricted guest (real mode support)
    mov rdx, 0x401E             ; SECONDARY_PROC_BASED_VM_EXEC_CONTROL
    vmwrite rdx, rax
    jc .error

    ; Set EPT pointer (points to EPT_PML4)
    mov rax, EPT_PML4
    or rax, 0x1E                ; Memory type: Write-back (6), Page-walk length: 3 (bits 3-5)
    mov rdx, 0x201A             ; EPT_POINTER
    vmwrite rdx, rax
    jc .error

    ; Exception bitmap (trap software interrupts for BIOS emulation)
    ; Bit 3 set = trap INT 3 (breakpoint)
    ; Bits 16, 19, 22, 25 = trap INT 10h, 13h, 16h, 19h (BIOS services)
    mov rax, (1 << 3) | (1 << 16) | (1 << 19) | (1 << 22) | (1 << 25)
    mov rdx, 0x4004             ; EXCEPTION_BITMAP
    vmwrite rdx, rax
    jc .error

    ; VM-entry controls (real mode, no IA-32e)
    mov ecx, 0x484              ; IA32_VMX_TRUE_ENTRY_CTLS
    rdmsr
    ; Do NOT set bit 9 (IA-32e mode) for real mode guest
    mov rdx, 0x4012             ; VM_ENTRY_CONTROLS
    vmwrite rdx, rax
    jc .error

    ; VM-exit controls
    mov ecx, 0x483              ; IA32_VMX_TRUE_EXIT_CTLS
    rdmsr
    or eax, (1 << 9)            ; Host address-space size
    mov rdx, 0x400C             ; VM_EXIT_CONTROLS
    vmwrite rdx, rax
    jc .error

    ; ---------------------------------------------------------------------
    ; 8) Launch the guest
    ; ---------------------------------------------------------------------
    mov rsi, msg_launching_guest
    call print_string_64

    vmlaunch

    ; If we get here, VMLAUNCH failed
    jmp .vmlaunch_failed

.error:
    mov rsi, msg_vmcs_error
    call print_string_64
    jmp .done

.vmlaunch_failed:
    mov rsi, msg_vmlaunch_failed
    call print_string_64

    ; Read VM_INSTRUCTION_ERROR
    mov rdx, 0x4400
    vmread rax, rdx
    mov rsi, msg_vm_error
    call print_string_64
    call print_hex_64

.done:
    pop rdx
    pop rcx
    pop rbx
    ret

;-----------------------------------------------------------------------------
; vm_exit_handler: Handle VM exits
;-----------------------------------------------------------------------------
vm_exit_handler:
    ; Save registers
    push rax
    push rbx
    push rcx
    push rdx
    push rsi
    push rdi

    ; Read VM_EXIT_REASON
    mov rdx, 0x4402             ; VM_EXIT_REASON
    vmread rax, rdx

    ; Mask to get basic exit reason (bits 0-15)
    and eax, 0xFFFF

    ; Check for Exception or NMI (exit reason 0) - handles software interrupts
    cmp eax, 0
    je .handle_exception

    ; Check for I/O instruction (exit reason 30)
    cmp eax, 30
    je .handle_io

    ; Check for HLT (exit reason 12)
    cmp eax, 12
    je .handle_hlt

    ; Check for EPT violation (exit reason 48)
    cmp eax, 48
    je .handle_ept_violation

    ; Check for triple fault (exit reason 2)
    cmp eax, 2
    je .handle_triple_fault

    ; Check for invalid guest state (exit reason 33)
    cmp eax, 33
    je .handle_invalid_state

    ; Unknown exit reason
    mov rsi, msg_unknown_exit
    call print_string_64
    call print_hex_64
    jmp .halt

.handle_io:
    ; Read EXIT_QUALIFICATION to get I/O details
    mov rdx, 0x6400             ; EXIT_QUALIFICATION
    vmread rbx, rdx             ; RBX = qualification

    ; Extract port number (bits 15:0)
    movzx ecx, bx               ; ECX = port number

    ; Check if this is port 0xE9 (QEMU debug port)
    cmp ecx, 0xE9
    jne .io_skip                ; Ignore other ports for now

    ; Check direction (bit 3: 0=OUT, 1=IN)
    test rbx, (1 << 3)
    jnz .io_skip                ; Only handle OUT for now

    ; Read guest RAX to get the byte being output
    mov rdx, 0x6828             ; GUEST_RAX  (low 64 bits)
    vmread rax, rdx

    ; AL now contains the byte to output
    ; Write it to host VGA memory
    mov rbx, [io_vga_cursor]
    cmp rbx, 0xB8FA0            ; End of VGA buffer (80x25)
    jae .io_wrap

    mov byte [rbx], al          ; Write character
    mov byte [rbx+1], 0x0F      ; White on black
    add rbx, 2                  ; Move cursor
    mov [io_vga_cursor], rbx
    jmp .io_advance_rip

.io_wrap:
    ; Wrap to beginning of VGA buffer
    mov rbx, 0xB8000
    mov [io_vga_cursor], rbx
    jmp .io_advance_rip

.io_skip:
.io_advance_rip:
    ; Advance guest RIP by instruction length
    mov rdx, 0x440C             ; VM_EXIT_INSTRUCTION_LEN
    vmread rax, rdx             ; RAX = instruction length

    mov rdx, 0x681E             ; GUEST_RIP
    vmread rbx, rdx             ; RBX = current RIP
    add rbx, rax                ; Add instruction length
    vmwrite rdx, rbx            ; Write back new RIP

    ; Resume guest execution
    vmresume

    ; If VMRESUME fails, fall through to error handling
    mov rsi, msg_vmresume_failed
    call print_string_64
    jmp .halt

.handle_exception:
    ; Read VM_EXIT_INTERRUPTION_INFORMATION to get interrupt vector
    mov rdx, 0x4404             ; VM_EXIT_INTR_INFO
    vmread rax, rdx

    ; Extract interrupt type (bits 10:8): 4 = software interrupt (INT n)
    mov rbx, rax
    shr rbx, 8
    and rbx, 0x7
    cmp rbx, 4                  ; Is it a software interrupt?
    jne .exception_skip         ; Not a software INT, skip

    ; Extract vector number (bits 7:0)
    and rax, 0xFF

    ; Check if it's a BIOS service INT
    cmp rax, 0x10               ; INT 10h - Video
    je .handle_int10h
    cmp rax, 0x13               ; INT 13h - Disk
    je .handle_int13h
    cmp rax, 0x16               ; INT 16h - Keyboard
    je .handle_int16h
    cmp rax, 0x19               ; INT 19h - Bootstrap loader
    je .handle_int19h
    cmp rax, 0x03               ; INT 3 - Breakpoint (for debugging)
    je .handle_int3

    ; Unknown INT, skip and advance RIP
.exception_skip:
    ; Advance RIP past the INT instruction (2 bytes)
    mov rdx, 0x681E             ; GUEST_RIP
    vmread rax, rdx
    add rax, 2                  ; INT is 2 bytes
    vmwrite rdx, rax
    vmresume
    jmp .halt

.handle_int3:
    ; INT 3 breakpoint - print debug message
    mov rsi, msg_int3
    call print_string_64
    ; Read and print guest RIP
    mov rdx, 0x681E
    vmread rax, rdx
    call print_hex_64
    jmp .halt

.handle_int10h:
    ; BIOS Video Service (INT 10h)
    call emulate_int10h
    vmresume
    mov rsi, msg_vmresume_failed
    call print_string_64
    jmp .halt

.handle_int13h:
    ; BIOS Disk Service (INT 13h)
    call emulate_int13h
    vmresume
    mov rsi, msg_vmresume_failed
    call print_string_64
    jmp .halt

.handle_int16h:
    ; BIOS Keyboard Service (INT 16h)
    call emulate_int16h
    vmresume
    mov rsi, msg_vmresume_failed
    call print_string_64
    jmp .halt

.handle_int19h:
    ; BIOS Bootstrap Loader (INT 19h)
    call emulate_int19h
    vmresume
    mov rsi, msg_vmresume_failed
    call print_string_64
    jmp .halt

.handle_hlt:
    mov rsi, msg_guest_hlt
    call print_string_64
    jmp .halt

.handle_ept_violation:
    mov rsi, msg_ept_violation
    call print_string_64
    ; Read guest-physical address that caused EPT violation
    mov rdx, 0x2400             ; GUEST_PHYSICAL_ADDRESS
    vmread rax, rdx
    call print_hex_64
    jmp .halt

.handle_triple_fault:
    mov rsi, msg_triple_fault
    call print_string_64
    jmp .halt

.handle_invalid_state:
    mov rsi, msg_invalid_state
    call print_string_64
    ; Read VM-instruction error
    mov rdx, 0x4400             ; VM_INSTRUCTION_ERROR
    vmread rax, rdx
    call print_hex_64
    jmp .halt

.halt:
    pop rdi
    pop rsi
    pop rdx
    pop rcx
    pop rbx
    pop rax
    cli
    hlt

;=============================================================================
; BIOS Emulation Functions
;=============================================================================

;-----------------------------------------------------------------------------
; emulate_int10h: BIOS Video Service (INT 10h)
; Reads guest AH to determine function
;-----------------------------------------------------------------------------
emulate_int10h:
    push rax
    push rbx
    push rcx
    push rdx
    push rsi
    push rdi

    ; Read guest RAX to get AH (function number)
    mov rdx, 0x6828             ; GUEST_RAX (encoding 0x6828 for full RAX)
    vmread rax, rdx

    ; Extract AH (bits 15:8)
    shr rax, 8
    and rax, 0xFF

    ; Dispatch based on function
    cmp rax, 0x0E               ; AH=0Eh: Teletype output
    je .int10_teletype
    cmp rax, 0x00               ; AH=00h: Set video mode
    je .int10_set_mode
    cmp rax, 0x03               ; AH=03h: Get cursor position
    je .int10_get_cursor

    ; Unsupported function - just advance RIP and return
    jmp .int10_done

.int10_teletype:
    ; Write character in AL to current cursor position
    ; Read guest RAX again to get AL
    mov rdx, 0x6828
    vmread rax, rdx
    and rax, 0xFF               ; AL = character

    ; Write to VGA memory at current io_vga_cursor
    mov rbx, [io_vga_cursor]
    mov byte [rbx], al
    mov byte [rbx+1], 0x0F      ; White on black
    add qword [io_vga_cursor], 2
    jmp .int10_done

.int10_set_mode:
    ; Set video mode - for now just acknowledge
    ; TODO: Implement mode switching
    jmp .int10_done

.int10_get_cursor:
    ; Return cursor position in DX
    ; For now, return 0,0
    xor rax, rax
    mov rdx, 0x6816             ; GUEST_RDX
    vmwrite rdx, rax
    jmp .int10_done

.int10_done:
    ; Advance guest RIP past INT instruction (2 bytes)
    mov rdx, 0x681E             ; GUEST_RIP
    vmread rax, rdx
    add rax, 2
    vmwrite rdx, rax

    pop rdi
    pop rsi
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret

;-----------------------------------------------------------------------------
; emulate_int13h: BIOS Disk Service (INT 13h)
; Reads guest AH to determine function
;-----------------------------------------------------------------------------
emulate_int13h:
    push rax
    push rbx
    push rdx

    ; Read guest RAX to get AH (function number)
    mov rdx, 0x6828             ; GUEST_RAX
    vmread rax, rdx

    ; Extract AH
    shr rax, 8
    and rax, 0xFF

    ; Dispatch based on function
    cmp rax, 0x00               ; AH=00h: Reset disk
    je .int13_reset
    cmp rax, 0x02               ; AH=02h: Read sectors
    je .int13_read
    cmp rax, 0x08               ; AH=08h: Get drive parameters
    je .int13_params

    ; Unsupported function - set CF (carry flag) to indicate error
    jmp .int13_error

.int13_reset:
    ; Reset disk - always succeed
    ; Clear CF in guest RFLAGS
    mov rdx, 0x6820             ; GUEST_RFLAGS
    vmread rax, rdx
    and rax, ~1                 ; Clear CF (bit 0)
    vmwrite rdx, rax
    ; Set AH=0 (success)
    xor rax, rax
    mov rdx, 0x6828             ; GUEST_RAX
    vmread rbx, rdx
    and rbx, 0xFFFFFFFFFFFF00FF ; Clear AH
    vmwrite rdx, rbx
    jmp .int13_done

.int13_read:
    ; Read sectors from virtual disk
    ; Input registers (from guest):
    ;   AL = number of sectors
    ;   CH = cylinder (low 8 bits)
    ;   CL = sector (bits 0-5) + cylinder high 2 bits (bits 6-7)
    ;   DH = head
    ;   DL = drive
    ;   ES:BX = destination buffer

    push rcx
    push rsi
    push rdi
    push r8
    push r9
    push r10
    push r11

    ; Read guest registers
    mov rdx, 0x6828             ; GUEST_RAX
    vmread rax, rdx
    mov r8, rax                 ; R8 = RAX (AL = sector count)

    mov rdx, 0x6810             ; GUEST_RCX
    vmread rcx, rdx             ; RCX = CX (CH=cylinder, CL=sector)

    mov rdx, 0x6816             ; GUEST_RDX
    vmread r9, rdx              ; R9 = DX (DH=head, DL=drive)

    mov rdx, 0x6808             ; GUEST_RBX
    vmread r10, rdx             ; R10 = BX (offset)

    mov rdx, 0x6802             ; GUEST_ES
    vmread r11, rdx             ; R11 = ES (segment)

    ; Extract parameters
    and r8, 0xFF                ; R8 = sector count (AL)
    test r8, r8
    jz .int13_read_error        ; No sectors to read

    ; Extract cylinder number (10 bits: CL[7:6] + CH[7:0])
    mov rsi, rcx
    shr rsi, 8                  ; SI = CH (cylinder low 8 bits)
    and rsi, 0xFF
    mov rdi, rcx
    shr rdi, 6                  ; DI = CL >> 6
    and rdi, 0x3                ; DI = cylinder high 2 bits
    shl rdi, 8
    or rsi, rdi                 ; RSI = full 10-bit cylinder number

    ; Extract sector number (6 bits: CL[5:0])
    and rcx, 0x3F               ; RCX = sector number (1-63)
    test rcx, rcx
    jz .int13_read_error        ; Sector 0 is invalid
    dec rcx                     ; Convert to 0-based

    ; Extract head number
    mov rdi, r9
    shr rdi, 8                  ; DI = DH (head)
    and rdi, 0xFF

    ; Calculate LBA = (C * heads_per_cylinder + H) * sectors_per_track + S
    ; For floppy: 2 heads, 18 sectors per track
    ; LBA = (cylinder * 2 + head) * 18 + sector

    mov rax, rsi                ; RAX = cylinder
    shl rax, 1                  ; RAX = cylinder * 2
    add rax, rdi                ; RAX = cylinder * 2 + head
    mov rbx, 18
    mul rbx                     ; RAX = (cylinder * 2 + head) * 18
    add rax, rcx                ; RAX = LBA

    ; Calculate source offset in disk image
    ; Source = DISK_IMAGE + LBA * 512
    shl rax, 9                  ; RAX = LBA * 512
    add rax, DISK_IMAGE         ; RAX = disk image + offset
    mov rsi, rax                ; RSI = source address

    ; Calculate destination address
    ; Destination = ES * 16 + BX
    shl r11, 4                  ; R11 = ES * 16
    add r11, r10                ; R11 = ES:BX linear address
    mov rdi, r11                ; RDI = destination address

    ; Copy data (sector count * 512 bytes)
    mov rcx, r8                 ; RCX = sector count
    shl rcx, 9                  ; RCX = sector count * 512 (bytes to copy)
    rep movsb                   ; Copy from RSI to RDI

    ; Set success status
    ; Clear CF, set AH=0, keep AL=sector count
    mov rdx, 0x6820             ; GUEST_RFLAGS
    vmread rax, rdx
    and rax, ~1                 ; Clear CF
    vmwrite rdx, rax

    ; Set AH=0 (success), AL already contains sector count
    mov rdx, 0x6828             ; GUEST_RAX
    vmread rax, rdx
    and rax, 0xFFFFFFFFFFFF00FF ; Clear AH
    vmwrite rdx, rax

    pop r11
    pop r10
    pop r9
    pop r8
    pop rdi
    pop rsi
    pop rcx
    jmp .int13_done

.int13_read_error:
    ; Set error status (CF=1, AH=1)
    pop r11
    pop r10
    pop r9
    pop r8
    pop rdi
    pop rsi
    pop rcx

    mov rdx, 0x6820
    vmread rax, rdx
    or rax, 1                   ; Set CF
    vmwrite rdx, rax

    mov rdx, 0x6828             ; GUEST_RAX
    vmread rax, rdx
    and rax, 0xFFFFFFFFFFFF00FF ; Clear AH
    or rax, 0x0100              ; Set AH=1 (error)
    vmwrite rdx, rax
    jmp .int13_done

.int13_params:
    ; Get drive parameters
    ; Input: DL = drive number
    ; Output:
    ;   CF = 0 (success)
    ;   AH = 0 (status)
    ;   CH = max cylinder number (lower 8 bits)
    ;   CL = max sector number (bits 0-5) + cylinder high 2 bits (bits 6-7)
    ;   DH = max head number
    ;   DL = number of drives
    ;   ES:DI = pointer to disk parameter table (can be 0:0)

    ; For standard 1.44MB floppy:
    ; 80 cylinders (0-79), 2 heads (0-1), 18 sectors (1-18)

    ; Clear CF (success)
    mov rdx, 0x6820             ; GUEST_RFLAGS
    vmread rax, rdx
    and rax, ~1
    vmwrite rdx, rax

    ; Set AH=0 (success)
    mov rdx, 0x6828             ; GUEST_RAX
    vmread rax, rdx
    and rax, 0xFFFFFFFFFFFF00FF ; Clear AH
    vmwrite rdx, rax

    ; Set CH=79 (max cylinder, lower 8 bits)
    ; Set CL=18 (max sector) + 0 (cylinder high 2 bits = 0)
    mov rax, 0x4F12             ; CH=0x4F (79), CL=0x12 (18)
    mov rdx, 0x6810             ; GUEST_RCX
    vmwrite rdx, rax

    ; Set DH=1 (max head), DL=1 (number of drives)
    mov rax, 0x0101
    mov rdx, 0x6816             ; GUEST_RDX
    vmread rbx, rdx
    and rbx, 0xFFFFFFFFFFFF0000 ; Clear DX
    or rbx, rax
    vmwrite rdx, rbx

    ; Set ES:DI = 0:0 (no parameter table)
    xor rax, rax
    mov rdx, 0x6802             ; GUEST_ES
    vmwrite rdx, rax
    mov rdx, 0x681A             ; GUEST_RDI
    vmwrite rdx, rax

    jmp .int13_done

.int13_error:
    ; Set CF to indicate error
    mov rdx, 0x6820
    vmread rax, rdx
    or rax, 1                   ; Set CF
    vmwrite rdx, rax

.int13_done:
    ; Advance guest RIP
    mov rdx, 0x681E
    vmread rax, rdx
    add rax, 2
    vmwrite rdx, rax

    pop rdx
    pop rbx
    pop rax
    ret

;-----------------------------------------------------------------------------
; emulate_int16h: BIOS Keyboard Service (INT 16h)
; Reads guest AH to determine function
;-----------------------------------------------------------------------------
emulate_int16h:
    push rax
    push rbx
    push rdx

    ; Read guest RAX to get AH
    mov rdx, 0x6828
    vmread rax, rdx

    ; Extract AH
    shr rax, 8
    and rax, 0xFF

    ; Dispatch based on function
    cmp rax, 0x00               ; AH=00h: Read keystroke
    je .int16_read
    cmp rax, 0x01               ; AH=01h: Check keystroke
    je .int16_check
    cmp rax, 0x02               ; AH=02h: Get shift flags
    je .int16_flags

    ; Unsupported function
    jmp .int16_done

.int16_read:
    ; Read keystroke - simulate Enter key for now
    ; In real implementation, this would read from keyboard buffer
    mov rax, 0x1C0D             ; Scan code 0x1C (Enter), ASCII 0x0D (CR)
    mov rdx, 0x6828             ; GUEST_RAX
    mov rbx, rax
    shl rbx, 8                  ; Shift to AX position
    vmwrite rdx, rbx
    jmp .int16_done

.int16_check:
    ; Check if keystroke available
    ; For now, always say yes (clear ZF)
    mov rdx, 0x6820             ; GUEST_RFLAGS
    vmread rax, rdx
    and rax, ~0x40              ; Clear ZF (bit 6)
    vmwrite rdx, rax
    ; Put dummy key in AX
    mov rax, 0x1C0D
    mov rdx, 0x6828
    vmread rbx, rdx
    and rbx, 0xFFFFFFFFFFFF0000 ; Clear AX
    or rbx, rax
    vmwrite rdx, rbx
    jmp .int16_done

.int16_flags:
    ; Get shift flags - return 0 (no keys pressed)
    mov rdx, 0x6828
    vmread rax, rdx
    and rax, 0xFFFFFFFFFFFFFF00 ; Clear AL
    vmwrite rdx, rax

.int16_done:
    ; Advance guest RIP
    mov rdx, 0x681E
    vmread rax, rdx
    add rax, 2
    vmwrite rdx, rax

    pop rdx
    pop rbx
    pop rax
    ret

;-----------------------------------------------------------------------------
; emulate_int19h: BIOS Bootstrap Loader (INT 19h)
; Loads boot sector from disk and transfers control to it
;-----------------------------------------------------------------------------
emulate_int19h:
    push rax
    push rbx
    push rcx
    push rdx
    push rsi
    push rdi

    ; 1) Read boot sector (sector 0) from virtual disk to 0x7C00
    ; Source: DISK_IMAGE + 0 (first sector)
    mov rsi, DISK_IMAGE         ; Source address
    mov rdi, 0x7C00             ; Destination: standard boot sector location
    mov rcx, 512                ; 512 bytes (one sector)
    rep movsb                   ; Copy boot sector

    ; 2) Verify boot signature (0xAA55 at offset 510-511)
    mov ax, [0x7C00 + 510]
    cmp ax, 0xAA55
    jne .int19_no_boot_signature

    ; 3) Set guest registers to boot from 0x0000:0x7C00
    ; Set CS = 0x0000
    xor rax, rax
    mov rdx, 0x802              ; GUEST_CS
    vmwrite rdx, rax
    jc .int19_error

    ; Set CS base = 0x0000
    mov rdx, 0x6808             ; GUEST_CS_BASE
    vmwrite rdx, rax
    jc .int19_error

    ; Set IP = 0x7C00
    mov rax, 0x7C00
    mov rdx, 0x681E             ; GUEST_RIP
    vmwrite rdx, rax
    jc .int19_error

    ; Set DS, ES, SS = 0x0000
    xor rax, rax
    mov rdx, 0x806              ; GUEST_DS
    vmwrite rdx, rax
    mov rdx, 0x680A             ; GUEST_DS_BASE
    vmwrite rdx, rax

    mov rdx, 0x800              ; GUEST_ES
    vmwrite rdx, rax
    mov rdx, 0x6806             ; GUEST_ES_BASE
    vmwrite rdx, rax

    mov rdx, 0x810              ; GUEST_SS
    vmwrite rdx, rax
    mov rdx, 0x6810             ; GUEST_SS_BASE
    vmwrite rdx, rax

    ; Set SP = 0x7C00 (stack grows down from boot sector)
    mov rax, 0x7C00
    mov rdx, 0x681C             ; GUEST_RSP
    vmwrite rdx, rax

    ; Set DL = boot drive (0x00 for floppy A:)
    xor rax, rax
    mov rdx, 0x6816             ; GUEST_RDX
    vmwrite rdx, rax

    ; Boot sector is loaded and guest is ready to execute from 0x7C00
    ; Note: We don't advance RIP here because we're setting it to 0x7C00

    pop rdi
    pop rsi
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret

.int19_no_boot_signature:
    ; No valid boot signature - halt
    ; In a real BIOS, this would try the next boot device
    ; For now, just set RIP to advance past INT 19h and let guest continue
    mov rdx, 0x681E
    vmread rax, rdx
    add rax, 2
    vmwrite rdx, rax

    pop rdi
    pop rsi
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret

.int19_error:
    ; VMWRITE error - should not happen
    pop rdi
    pop rsi
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret

;-----------------------------------------------------------------------------
; print_hex_64: Print 64-bit value in RAX as hex
;-----------------------------------------------------------------------------
print_hex_64:
    push rax
    push rbx
    push rcx
    push rdx
    push rsi

    mov rbx, 0xB8000            ; VGA buffer
    mov rcx, 16                 ; 16 hex digits

.loop:
    rol rax, 4                  ; Rotate left 4 bits
    mov dl, al
    and dl, 0x0F                ; Get low nibble
    cmp dl, 10
    jl .digit
    add dl, 'A' - 10
    jmp .print
.digit:
    add dl, '0'
.print:
    mov byte [rbx], dl
    mov byte [rbx+1], 0x0F      ; White on black
    add rbx, 2
    dec rcx
    jnz .loop

    pop rsi
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret

;-----------------------------------------------------------------------------
; Strings
;-----------------------------------------------------------------------------
msg_stage2:          db 'Stage 2: Hypervisor loader', 13, 10, 0
msg_no_long_mode:    db 'ERROR: Long mode not supported!', 13, 10, 0
msg_no_vmx:          db 'ERROR: VT-x not supported!', 13, 10, 0
msg_vmx_ok:          db 'VT-x supported', 13, 10, 0
msg_long_mode:       db 'Long mode enabled', 0
msg_vmxon_ok:        db 'VMXON executed successfully!', 0
msg_vmxon_fail:      db 'VMXON failed!', 0
msg_stage3_fail:     db 'Stage 3 failed!', 0
msg_vmcs_setup:      db 'Setting up VMCS...', 0
msg_vmcs_error:      db 'VMCS error!', 0
msg_vmlaunch_failed: db 'VMLAUNCH failed!', 0
msg_vm_error:        db 'VM instruction error: ', 0
msg_launching_guest:  db 'Launching guest...', 0
msg_guest_hlt:        db 'Guest executed HLT successfully!', 0
msg_unknown_exit:     db 'Unknown VM exit: ', 0
msg_ept_violation:    db 'EPT violation at GPA: ', 0
msg_triple_fault:     db 'Guest triple fault!', 0
msg_invalid_state:    db 'Invalid guest state, error: ', 0
msg_vmresume_failed:  db 'VMRESUME failed!', 0
msg_int3:             db 'INT 3 breakpoint at RIP: ', 0

; I/O handler state
align 8
io_vga_cursor:  dq 0xB8000          ; VGA cursor for I/O port output

;-----------------------------------------------------------------------------
; Embedded guest real mode code (512 bytes)
;-----------------------------------------------------------------------------
align 16
guest_code_data:
incbin "build/guest_real.bin"

;-----------------------------------------------------------------------------
; Pad to sector boundary (optional)
;-----------------------------------------------------------------------------
times 20480-($-$$) db 0             ; Pad to 20KB
