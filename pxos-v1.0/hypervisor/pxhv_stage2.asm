; pxHV Stage 2 & 3 - Hypervisor Loader & Minimal Guest
BITS 16
ORG 0x10000

PML4_ADDR    equ 0x70000
PDPT_ADDR    equ 0x71000
PD_ADDR      equ 0x72000
VMXON_REGION equ 0x15000
VMCS_REGION  equ 0x16000
GUEST_CODE   equ 0x200000

start:
    call check_cpuid
    call check_long_mode
    call check_vmx

    lgdt [gdt_descriptor]

    mov eax, cr0
    or al, 1
    mov cr0, eax
    jmp 0x08:protected_mode

BITS 32
protected_mode:
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax

    mov edi, PML4_ADDR
    mov ecx, 4096 * 3
    xor eax, eax
    rep stosd
    mov dword [PML4_ADDR], PDPT_ADDR | 3
    mov dword [PDPT_ADDR], PD_ADDR | 3
    mov edi, PD_ADDR
    mov eax, 0x83
    mov ecx, 512
.pd_loop:
    mov [edi], eax
    add eax, 0x200000
    add edi, 8
    loop .pd_loop

    mov eax, PML4_ADDR
    mov cr3, eax

    mov ecx, 0xC0000080
    rdmsr
    or eax, 1 << 8
    wrmsr

    mov eax, cr4
    or eax, 1 << 5
    mov cr4, eax

    mov eax, cr0
    or eax, 1 << 31
    mov cr0, eax

    jmp 0x08:long_mode

BITS 64
long_mode:
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    mov rsp, 0x90000

    call enable_vmx
    call vmxon_init
    call vmxon_exec

    call setup_guest

    vmlaunch
    ; if vmlaunch fails, we will fall through here
    mov rsi, msg_vmlaunch_fail
    call print_string_64
    cli
    hlt

setup_guest:
    ; Install guest code (HLT)
    mov byte [GUEST_CODE], 0xF4

    ; Clear and setup VMCS
    mov rdi, VMCS_REGION
    mov rcx, 1024
    xor rax, rax
    rep stosq
    mov ecx, 0x480
    rdmsr
    mov [VMCS_REGION], eax
    mov rax, VMCS_REGION
    vmptrld [rax]

    ; --- Guest State ---
    vmwrite 0x800, 0x10       ; GUEST_ES_SELECTOR
    vmwrite 0x802, 0x10       ; GUEST_CS_SELECTOR
    vmwrite 0x804, 0x10       ; GUEST_SS_SELECTOR
    vmwrite 0x806, 0x10       ; GUEST_DS_SELECTOR
    vmwrite 0x808, 0x10       ; GUEST_FS_SELECTOR
    vmwrite 0x80A, 0x10       ; GUEST_GS_SELECTOR
    vmwrite 0x4800, 0x18      ; GUEST_LDTR_SELECTOR
    vmwrite 0x4814, 0x2       ; GUEST_RFLAGS
    vmwrite 0x4816, 0x90000   ; GUEST_RSP
    vmwrite 0x481A, GUEST_CODE; GUEST_RIP
    vmwrite 0x4824, 0x2029    ; GUEST_CR0
    vmwrite 0x4826, 0x60040   ; GUEST_CR4
    vmwrite 0x482A, 0x1000    ; GUEST_EFER
    vmwrite 0x6800, 0         ; GUEST_ES_BASE
    vmwrite 0x6802, 0         ; GUEST_CS_BASE
    vmwrite 0x6804, 0         ; GUEST_SS_BASE
    vmwrite 0x6806, 0         ; GUEST_DS_BASE
    vmwrite 0x6808, 0         ; GUEST_FS_BASE
    vmwrite 0x680A, 0         ; GUEST_GS_BASE
    vmwrite 0x680C, 0         ; GUEST_LDTR_BASE
    vmwrite 0x681E, 0xFFFF    ; GUEST_ES_LIMIT
    vmwrite 0x6820, 0xFFFF    ; GUEST_CS_LIMIT
    vmwrite 0x6822, 0xFFFF    ; GUEST_SS_LIMIT
    vmwrite 0x6824, 0xFFFF    ; GUEST_DS_LIMIT
    vmwrite 0x6826, 0xFFFF    ; GUEST_FS_LIMIT
    vmwrite 0x6828, 0xFFFF    ; GUEST_GS_LIMIT
    vmwrite 0x682A, 0xFFFF    ; GUEST_LDTR_LIMIT
    vmwrite 0x6838, 3         ; GUEST_ES_AR_BYTES
    vmwrite 0x683A, 3         ; GUEST_CS_AR_BYTES
    vmwrite 0x683C, 3         ; GUEST_SS_AR_BYTES
    vmwrite 0x683E, 3         ; GUEST_DS_AR_BYTES
    vmwrite 0x6840, 3         ; GUEST_FS_AR_BYTES
    vmwrite 0x6842, 3         ; GUEST_GS_AR_BYTES
    vmwrite 0x6844, 0x10000   ; GUEST_LDTR_AR_BYTES

    ; --- Host State ---
    vmwrite 0xC00, 0x10       ; HOST_ES_SELECTOR
    vmwrite 0xC02, 0x08       ; HOST_CS_SELECTOR
    vmwrite 0xC04, 0x10       ; HOST_SS_SELECTOR
    vmwrite 0xC06, 0x10       ; HOST_DS_SELECTOR
    vmwrite 0xC08, 0x10       ; HOST_FS_SELECTOR
    vmwrite 0xC0A, 0x10       ; HOST_GS_SELECTOR
    vmwrite 0x6C00, 0         ; HOST_FS_BASE
    vmwrite 0x6C02, 0         ; HOST_GS_BASE
    vmwrite 0x6C14, 0x90000   ; HOST_RSP
    vmwrite 0x6C16, vm_exit   ; HOST_RIP
    vmwrite 0x6C0A, 0x2029    ; HOST_CR0
    vmwrite 0x6C0C, 0x60040   ; HOST_CR4
    vmwrite 0x6C12, 0x1000    ; HOST_EFER

    ; --- VM-Execution Control Fields ---
    vmwrite 0x4012, 0x4002007B ; VM_EXIT_CONTROLS
    vmwrite 0x401E, 0x11FB     ; VM_ENTRY_CONTROLS
    vmwrite 0x4002, 0x6401E    ; PIN_BASED_VM_EXEC_CONTROL
    vmwrite 0x401C, 0x33481E   ; CPU_BASED_VM_EXEC_CONTROL

    ret

vm_exit:
    mov rsi, msg_vm_exit
    call print_string_64
    cli
    hlt

; --- 16-bit Helper Functions ---
BITS 16
%include "pxhv_helpers.asm"

; --- 64-bit Helper Functions ---
BITS 64
%include "pxhv_helpers_64.asm"

msg_vmlaunch_fail db 'VMLAUNCH failed!', 13, 10, 0
msg_vm_exit       db 'Guest executed HLT!', 13, 10, 0

gdt_start:
    dq 0
    dq 0x00CF9A000000FFFF
    dq 0x00CF92000000FFFF
gdt_end:
gdt_descriptor:
    dw gdt_end - gdt_start - 1
    dq gdt_start
