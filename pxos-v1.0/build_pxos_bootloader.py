#!/usr/bin/env python3
"""
build_pxos_bootloader.py - Build pxOS Linux bootloader without NASM

This script creates the bootloader binary directly from Python,
assembling the necessary x86 machine code by hand.
"""

import struct
import sys

def create_bootloader():
    """Create the 512-byte bootloader"""
    bootloader = bytearray(512)
    pos = 0

    # Helper function to write bytes
    def write_bytes(*bytes_list):
        nonlocal pos
        for b in bytes_list:
            bootloader[pos] = b
            pos += 1

    # === Boot Sector Start (0x7C00) ===

    # cli (disable interrupts)
    write_bytes(0xFA)

    # xor ax, ax (set ax to 0)
    write_bytes(0x31, 0xC0)

    # mov ds, ax
    write_bytes(0x8E, 0xD8)

    # mov es, ax
    write_bytes(0x8E, 0xC0)

    # mov ax, 0x9000 (stack segment)
    write_bytes(0xB8, 0x00, 0x90)

    # mov ss, ax
    write_bytes(0x8E, 0xD0)

    # mov sp, 0xFFFF (stack pointer)
    write_bytes(0xBC, 0xFF, 0xFF)

    # sti (enable interrupts)
    write_bytes(0xFB)

    # mov [boot_drive], dl (save boot drive)
    # boot_drive is at offset 0x80 (we'll define it later)
    write_bytes(0x88, 0x16, 0x80, 0x7C)

    # === Clear Screen ===
    # mov ah, 0x00 (set video mode)
    write_bytes(0xB4, 0x00)

    # mov al, 0x03 (80x25 text mode)
    write_bytes(0xB0, 0x03)

    # int 0x10
    write_bytes(0xCD, 0x10)

    # === Print Boot Message ===
    # mov si, msg_boot (message offset)
    # msg_boot will be at offset 0x90
    write_bytes(0xBE, 0x90, 0x7C)

    # call print_string (print_string function at offset 0x50)
    write_bytes(0xE8, 0x2B, 0x00)  # relative call (+43 bytes)

    # === Load Kernel ===
    # mov si, msg_loading
    write_bytes(0xBE, 0xB0, 0x7C)  # msg_loading at 0x7CB0

    # call print_string
    write_bytes(0xE8, 0x23, 0x00)

    # Load kernel to 0x1000:0x0000
    # mov ax, 0x1000
    write_bytes(0xB8, 0x00, 0x10)

    # mov es, ax
    write_bytes(0x8E, 0xC0)

    # xor bx, bx
    write_bytes(0x31, 0xDB)

    # BIOS disk read
    # mov ah, 0x02 (read sectors)
    write_bytes(0xB4, 0x02)

    # mov al, 50 (number of sectors)
    write_bytes(0xB0, 0x32)

    # mov ch, 0x00 (cylinder)
    write_bytes(0xB5, 0x00)

    # mov cl, 0x02 (sector 2)
    write_bytes(0xB1, 0x02)

    # mov dh, 0x00 (head)
    write_bytes(0xB6, 0x00)

    # mov dl, [boot_drive]
    write_bytes(0x8A, 0x16, 0x80, 0x7C)

    # int 0x13
    write_bytes(0xCD, 0x13)

    # jc disk_error (jump if carry - error)
    write_bytes(0x72, 0x05)  # jump +5 bytes

    # === Print OK ===
    # mov si, msg_ok
    write_bytes(0xBE, 0xC8, 0x7C)

    # call print_string
    write_bytes(0xE8, 0x08, 0x00)

    # === Halt (simplified - in real version we'd switch to protected mode) ===
    # For now just halt
    write_bytes(0xEB, 0xFE)  # jmp $ (infinite loop)

    # === print_string function (offset 0x50) ===
    # Position cursor at 0x50
    while pos < 0x50:
        write_bytes(0x90)  # NOP padding

    # print_string:
    # pusha
    write_bytes(0x60)

    # mov ah, 0x0E
    write_bytes(0xB4, 0x0E)

    # .loop:
    # lodsb (load byte from DS:SI into AL)
    write_bytes(0xAC)

    # test al, al (check if zero)
    write_bytes(0x84, 0xC0)

    # jz .done (if zero, done)
    write_bytes(0x74, 0x04)  # jump +4

    # int 0x10 (print character)
    write_bytes(0xCD, 0x10)

    # jmp .loop
    write_bytes(0xEB, 0xF6)  # jump back -10 bytes

    # .done:
    # popa
    write_bytes(0x61)

    # ret
    write_bytes(0xC3)

    # === Data Section ===
    # boot_drive at 0x80
    while pos < 0x80:
        write_bytes(0x00)

    # boot_drive: db 0
    write_bytes(0x00)

    # === Messages ===
    # msg_boot at 0x90
    while pos < 0x90:
        write_bytes(0x00)

    msg_boot = b"pxOS Linux Loader v1.0\r\n\0"
    for byte in msg_boot:
        write_bytes(byte)

    # msg_loading at 0xB0
    while pos < 0xB0:
        write_bytes(0x00)

    msg_loading = b"Loading kernel... \0"
    for byte in msg_loading:
        write_bytes(byte)

    # msg_ok at 0xC8
    while pos < 0xC8:
        write_bytes(0x00)

    msg_ok = b"OK\r\n\0"
    for byte in msg_ok:
        write_bytes(byte)

    # Fill rest with zeros
    while pos < 510:
        write_bytes(0x00)

    # Boot signature
    bootloader[510] = 0x55
    bootloader[511] = 0xAA

    return bytes(bootloader)

def main():
    """Main function"""
    print("Building pxOS Linux bootloader...")

    # Create bootloader
    bootloader = create_bootloader()

    # Verify size
    if len(bootloader) != 512:
        print(f"ERROR: Bootloader must be exactly 512 bytes (got {len(bootloader)})")
        sys.exit(1)

    # Write to file
    output_file = "build/pxos-linux.bin"

    # Create build directory if needed
    import os
    os.makedirs("build", exist_ok=True)

    with open(output_file, "wb") as f:
        f.write(bootloader)

    print(f"✓ Bootloader created: {output_file} ({len(bootloader)} bytes)")
    print(f"  Boot signature: 0x{bootloader[510]:02X}{bootloader[511]:02X}")

    # Verify boot signature
    if bootloader[510] == 0x55 and bootloader[511] == 0xAA:
        print("  ✓ Valid boot signature")
    else:
        print("  ✗ Invalid boot signature!")
        sys.exit(1)

    return 0

if __name__ == "__main__":
    sys.exit(main())
