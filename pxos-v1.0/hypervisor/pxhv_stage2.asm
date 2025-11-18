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

    ; 4.6) Load guest real mode code to memory at 0x7C00
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

    ; Guest RIP = 0x7C00 (real mode boot sector location)
    mov rax, 0x7C00
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
    xor rax, rax                ; CS = 0x0000
    mov rdx, 0x802              ; GUEST_CS_SELECTOR
    vmwrite rdx, rax
    jc .error

    ; Guest CS base (segment * 16)
    xor rax, rax                ; 0x0000 * 16 = 0x00000
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

    ; Guest IDTR (minimal empty IDT)
    xor rax, rax
    mov rdx, 0x6818             ; GUEST_IDTR_BASE
    vmwrite rdx, rax
    jc .error
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
msg_launching_guest: db 'Launching guest...', 0
msg_guest_hlt:       db 'Guest executed HLT successfully!', 0
msg_unknown_exit:    db 'Unknown VM exit: ', 0
msg_ept_violation:   db 'EPT violation at GPA: ', 0
msg_triple_fault:    db 'Guest triple fault!', 0
msg_invalid_state:   db 'Invalid guest state, error: ', 0

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
