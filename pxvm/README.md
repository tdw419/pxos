# pxVM Technical Reference

This document provides a detailed technical overview of the PxVM (Pixel Virtual Machine), the engine powering this digital biosphere. It is intended for developers who wish to understand the inner workings of the VM, create new organisms, or extend the capabilities of the system.

## Architecture

The PxVM is a multi-kernel, 2D virtual machine designed for simulating artificial life. Each organism, or "kernel," is an independent entity with its own memory, registers, and program counter. All kernels run in parallel and interact with each other and their environment through a shared memory space.

### Key Components

*   **VM:** The main virtual machine, responsible for managing the kernels, the shared memory layers, and the simulation loop.
*   **Kernel:** An individual organism, with a 64KB memory space, 8 general-purpose registers, and a program counter.
*   **Shared Memory Layers:**
    *   **Framebuffer:** A 1024x1024x3 numpy array of `uint8` values, representing the visual output of the world.
    *   **Pheromone Layer:** A 1024x1024 numpy array of `uint8` values, used for chemical communication.
    *   **Glyph Layer:** A 1024x1024 numpy array of `uint16` values, used for symbolic communication.

## Instruction Set

The PxVM has a simple but powerful instruction set, designed to be easily extensible.

| Opcode | Mnemonic | Description                               |
|--------|----------|-------------------------------------------|
| 0      | `HALT`   | Halts the kernel's execution.             |
| 1      | `MOV`    | Moves an immediate value into a register. |
| 2      | `PLOT`   | Plots a pixel on the framebuffer.         |
| 3      | `ADD`    | Adds the value of one register to another.|
| 4      | `ADDI`   | Adds an immediate value to a register.    |
| 255    | `NOP`    | No operation.                             |

## Syscall Reference

Syscalls are used to interact with the shared memory layers and the VM itself.

| Opcode | Mnemonic                | Description                                                                 |
|--------|-------------------------|-----------------------------------------------------------------------------|
| 100    | `SYS_EMIT_PHEROMONE`    | Emits a pheromone at a specified location.                                  |
| 101    | `SYS_SENSE_PHEROMONE`   | Senses the pheromone level at a specified location.                         |
| 102    | `SYS_WRITE_GLYPH`       | Writes a glyph to a specified location.                                     |
| 103    | `SYS_READ_GLYPH`        | Reads a glyph from a specified location.                                    |
| 104    | `SYS_SPAWN`             | Spawns a new kernel with a copy of the parent's memory.                       |
