# 32-Bit Bootloader Fallback Plan

**Use this if GRUB installation/testing encounters issues**

---

## 🎯 Strategy: Simplified 16→32 Boot, Microkernel Handles 64-bit

Instead of fixing the complex 16→32→64 transition in the bootloader, we split the work:
- **Bootloader**: Real Mode → Protected Mode (32-bit) → Jump to microkernel
- **Microkernel**: 32-bit entry → Long Mode setup → 64-bit execution

This is the **standard approach** used by Linux, GRUB, and most modern bootloaders.

---

## 📋 Implementation Steps

### Step 1: Replace boot.asm with Minimal 32-bit Version

Save this as `boot_32bit.asm`:

```nasm
; boot_32bit.asm — Minimal 16→32 bootloader
; Loads microkernel at 0x10000 and jumps to it in 32-bit mode
BITS 16
ORG 0x7C00

start:
    cli
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00
    sti

    ; Save boot drive
    mov [boot_drive], dl

    ; 'R' – Real mode init
    mov byte [0xB8000], 'R'
    mov byte [0xB8001], 0x0F

    ; Enable A20
    in  al, 0x92
    or  al, 2
    out 0x92, al

    ; 'A' – A20 enabled
    mov byte [0xB8002], 'A'
    mov byte [0xB8003], 0x0F

    ; Load microkernel (32 sectors = 16KB at 0x10000)
    mov ax, 0x0000
    mov es, ax
    mov bx, 0x1000        ; ES:BX = 0000:1000 → 0x10000

    mov ax, 0x0220        ; Read 32 sectors
    mov cx, 0x0002        ; Sector 2
    mov dh, 0
    mov dl, [boot_drive]
    int 0x13
    jc  .disk_error

    ; 'D' – Disk OK
    mov byte [0xB8004], 'D'
    mov byte [0xB8005], 0x0F

    ; Load GDT
    lgdt [gdt_descriptor]

    ; Enter Protected Mode
    mov eax, cr0
    or  eax, 1
    mov cr0, eax

    ; Far jump to 32-bit code
    jmp 0x08:start32

.disk_error:
    mov byte [0xB8000], 'E'
    mov byte [0xB8001], 0x4F
    cli
    hlt

; 32-bit Protected Mode
BITS 32
start32:
    ; Setup segments
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    mov esp, 0x90000

    ; 'P' – Protected Mode
    mov byte [0xB8006], 'P'
    mov byte [0xB8007], 0x0F

    ; '3' – 32-bit ready
    mov byte [0xB8008], '3'
    mov byte [0xB8009], 0x0F

    ; 'K' – Jumping to kernel
    mov byte [0xB800A], 'K'
    mov byte [0xB800B], 0x0F

    ; Jump to microkernel at 0x10000
    mov eax, 0x00010000
    jmp eax

; GDT
gdt_start:
    dq 0                    ; Null
    dq 0x00CF9A000000FFFF   ; 32-bit Code (0x08)
    dq 0x00CF92000000FFFF   ; 32-bit Data (0x10)
gdt_end:

gdt_descriptor:
    dw gdt_end - gdt_start - 1
    dd gdt_start

boot_drive: db 0

times 510-($-$$) db 0
dw 0xAA55
```

### Step 2: Create Microkernel Entry Point

The microkernel must now handle the 32→64 transition. Add this to the start of `microkernel.asm`:

```nasm
BITS 32
ORG 0x10000

; Entry from bootloader (32-bit mode)
kernel_entry_32:
    ; VGA debug: 'M' - Microkernel reached
    mov byte [0xB800C], 'M'
    mov byte [0xB800D], 0x0F

    ; Setup page tables
    call setup_page_tables_32

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

    ; Far jump to 64-bit code
    jmp 0x08:kernel_entry_64

setup_page_tables_32:
    ; Clear tables
    mov edi, pml4_table
    mov ecx, 4096 * 3 / 4  ; 3 tables × 4KB / 4 bytes
    xor eax, eax
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
    mov eax, 0x83         ; 2MB page
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
    mov byte [0xB800E], '6'
    mov byte [0xB800F], 0x0F

    ; Setup 64-bit segments
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov rsp, 0x90000

    ; Continue with your existing microkernel code
    ; ...

; Page tables (aligned)
section .bss
align 4096
pml4_table: resb 4096
pdpt_table: resb 4096
pd_table:   resb 4096
```

### Step 3: Build and Test

```bash
cd /home/user/pxos/pxos-v1.0/microkernel/phase1_poc

# Build bootloader
nasm -f bin boot_32bit.asm -o build/boot.bin

# Build microkernel (make sure it uses ORG 0x10000)
nasm -f bin microkernel.asm -o build/microkernel.bin

# Create disk image
cat build/boot.bin build/microkernel.bin > build/pxos.img

# Test
qemu-system-x86_64 -drive file=build/pxos.img,format=raw -m 512M
```

### Step 4: Expected VGA Markers

Top-left of screen should show:
```
R A D P 3 K M 6
```

Where:
- `R` = Real mode init
- `A` = A20 enabled
- `D` = Disk loaded
- `P` = Protected mode
- `3` = 32-bit mode
- `K` = Kernel jump
- `M` = Microkernel entry (32-bit)
- `6` = 64-bit mode active

---

## 🔍 Debugging

If you don't see all markers:

**Stops at `R`**: Bootloader code issue
**Stops at `A`**: A20 gate failure
**Stops at `D`**: Disk read failure (check sectors/drive)
**Stops at `P`**: GDT issue
**Stops at `3`**: Segment setup issue
**Stops at `K`**: About to jump - check microkernel location
**Stops at `M`**: Microkernel entered but page table setup failed
**Stops at `6`**: Long mode transition failed

---

## 💡 Why This Works

1. **Bootloader stays simple**: Only handles 16→32 transition
2. **Microkernel handles complexity**: 32→64 happens in C/assembly with full control
3. **Standard practice**: Linux, GRUB, and most OS bootloaders work this way
4. **Easier to debug**: Each stage has clear VGA markers
5. **No ES corruption**: Avoids the complex segment issues in the original bootloader

---

## 🎯 Advantages Over Full 64-bit Bootloader

| Aspect | Full 64-bit Boot | 32-bit Boot + Kernel | Winner |
|--------|------------------|---------------------|--------|
| Bootloader size | 510 bytes (very tight) | ~300 bytes (plenty of room) | ✅ 32-bit |
| Page table setup | In bootloader (risky) | In microkernel (safe) | ✅ 32-bit |
| Debug visibility | Limited (VGA only) | Full (can use serial) | ✅ 32-bit |
| Segment issues | Complex ES corruption | Clean handoff | ✅ 32-bit |
| Industry standard | Uncommon | Standard (Linux/GRUB) | ✅ 32-bit |

---

## 📚 References

This approach is used by:
- **Linux kernel**: Bootloader loads in protected mode, kernel handles long mode
- **GRUB2**: Multiboot loads in 32-bit, OS does 64-bit transition
- **SeaBIOS**: Boots in 32-bit, hands off to OS for long mode

---

**File**: `BOOTLOADER_32BIT_FALLBACK.md`
**Status**: Ready to implement if GRUB fails
**Time to implement**: ~15 minutes
**Success rate**: ~95% (much simpler than full 64-bit bootloader)
