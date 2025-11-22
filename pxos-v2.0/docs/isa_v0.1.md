# Pixel ISA v0.1

This document specifies the v0.1 instruction set architecture for the Pixel Virtual Machine.

## Overview

The Pixel ISA is a simple, register-based architecture where instructions are encoded in the RGBA channels of individual pixels. The VM operates on a set of general-purpose registers and a data buffer.

## Pixel Format

Each pixel represents a single instruction. The RGBA channels are interpreted as follows:

- **R (Red):** Opcode (8 bits)
- **G (Green):** Argument 0 (8 bits)
- **B (Blue):** Argument 1 (8 bits)
- **A (Alpha):** Argument 2 / Flags (8 bits)

All arguments are unsigned 8-bit integers.

## Registers

The VM has the following registers:

- **ip:** Instruction Pointer (32-bit) - Points to the x-coordinate of the current instruction in the code texture.
- **regs:** 16 general-purpose 32-bit registers, addressed by indices 0-15.

## Instruction Set

| Opcode | Mnemonic | `arg0` | `arg1` | `arg2` | Description |
|---|---|---|---|---|---|
| 1 | `LOAD` | `data_idx` | `reg_dst` | - | Loads a 32-bit value from `data[data_idx]` into `regs[reg_dst]`. |
| 2 | `STORE` | `reg_src` | `data_idx` | - | Stores the 32-bit value from `regs[reg_src]` into `data[data_idx]`. |
| 3 | `ADD` | `reg_src` | `reg_dst` | - | Adds the value of `regs[reg_src]` to `regs[reg_dst]` and stores the result in `regs[reg_dst]`. |
| 4 | `JUMP` | `offset` | - | - | Adds `offset` to the `ip` register. The offset is a signed 8-bit integer. |
| 5 | `CMP` | `reg_a` | `reg_b` | - | Compares `regs[reg_a]` and `regs[reg_b]` and sets the zero flag. |
| 6 | `JNE` | `offset` | - | - | Jumps by `offset` if the zero flag is not set. |
| 255 | `HALT` | - | - | - | Halts the execution of the program. |
