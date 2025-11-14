#!/usr/bin/env python3
"""
pxVM Extended - Multi-process VM with Filesystem and IPC
pxOS Kernel v1.0 - Self-Hosting Edition

New syscalls:
- SYS_IPC_SEND (14): Send message to another process
- SYS_IPC_RECV (15): Receive message (blocking)
- SYS_FS_OPEN (20): Open file
- SYS_FS_CLOSE (21): Close file
- SYS_FS_WRITE (22): Write to file
- SYS_FS_READ (23): Read from file
- SYS_FORK (30): Create child process
- SYS_SPAWN (31): Spawn process from bytecode file

IMPERFECT COMPUTING MODE:
- All syscalls validate inputs and use safe defaults
- Filesystem errors return 0 (failure) instead of crashing
- IPC to invalid PID logs warning and continues
- Unknown syscalls log warnings
"""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import copy

# Import base VM opcodes
from pxvm import (
    OP_HALT, OP_NOP, OP_IMM8, OP_IMM32, OP_MOV, OP_ADD, OP_SUB, OP_SYSCALL,
    SYS_PRINT_ID, SYS_RECT_ID, SYS_TEXT_ID, SYS_LAYER_USE_ID
)

# New syscall numbers
SYS_IPC_SEND = 14
SYS_IPC_RECV = 15
SYS_FS_OPEN = 20
SYS_FS_CLOSE = 21
SYS_FS_WRITE = 22
SYS_FS_READ = 23
SYS_FORK = 30
SYS_SPAWN = 31

# New opcodes for control flow
OP_JMP = 0x40   # JMP addr
OP_JZ = 0x41    # JZ reg, addr (jump if zero)
OP_CMP = 0x42   # CMP dst, reg1, reg2 (dst = 1 if reg1==reg2, else 0)
OP_LOAD = 0x50  # LOAD dst, addr (load word from memory)
OP_STORE = 0x51 # STORE addr, src (store word to memory)


@dataclass
class Message:
    """IPC message"""
    from_pid: int
    to_pid: int
    msg_type: int
    data: List[int] = field(default_factory=list)


@dataclass
class FileHandle:
    """Open file handle"""
    file_id: int
    path: str
    mode: int  # 0=read, 1=write, 2=create/truncate
    pos: int = 0


@dataclass
class Process:
    """Process control block"""
    pid: int
    pc: int = 0
    registers: List[int] = field(default_factory=lambda: [0] * 8)
    memory: bytearray = field(default_factory=lambda: bytearray(65536))
    halted: bool = False
    waiting_for_ipc: bool = False
    file_handles: Dict[int, FileHandle] = field(default_factory=dict)
    next_handle_id: int = 1
    message_queue: List[Message] = field(default_factory=list)


class VirtualFilesystem:
    """Simple in-memory filesystem"""

    def __init__(self):
        self.files: Dict[str, bytearray] = {}
        # Pre-create some paths
        self.files["build/"] = bytearray()  # directory marker

    def open(self, path: str, mode: int) -> bool:
        """Open/create file. Returns True on success."""
        try:
            if mode == 2:  # create/truncate
                self.files[path] = bytearray()
            elif mode == 1:  # write (must exist)
                if path not in self.files:
                    return False
            elif mode == 0:  # read (must exist)
                if path not in self.files:
                    return False
            return True
        except Exception:
            return False

    def read(self, path: str, offset: int, length: int) -> Optional[bytes]:
        """Read from file. Returns None on error."""
        try:
            if path not in self.files:
                return None
            data = self.files[path]
            return bytes(data[offset:offset+length])
        except Exception:
            return None

    def write(self, path: str, offset: int, data: bytes) -> bool:
        """Write to file. Returns True on success."""
        try:
            if path not in self.files:
                return False
            file_data = self.files[path]
            # Extend if needed
            if offset + len(data) > len(file_data):
                file_data.extend(b'\x00' * (offset + len(data) - len(file_data)))
            file_data[offset:offset+len(data)] = data
            return True
        except Exception:
            return False

    def get_size(self, path: str) -> int:
        """Get file size. Returns 0 if not found."""
        if path not in self.files:
            return 0
        return len(self.files[path])


class PxVMExtended:
    """
    Extended pxVM with multi-process support, filesystem, and IPC

    Features:
    - Multiple concurrent processes
    - Round-robin scheduler
    - In-memory virtual filesystem
    - Message-based IPC
    - Process spawning (FORK, SPAWN)
    - Imperfect mode: all errors handled gracefully
    """

    def __init__(self, imperfect: bool = True):
        self.imperfect = imperfect
        self.processes: Dict[int, Process] = {}
        self.next_pid = 1
        self.current_pid: Optional[int] = None
        self.filesystem = VirtualFilesystem()
        self.sysout: List[str] = []  # Collected PXTERM output

        # Syscall lookup tables (from base pxVM)
        self.sys_messages = {
            1: "PXVM booting...",
            2: "PXVM ready.",
            3: "Task complete.",
            4: "Kernel init done.",
            5: "Process started.",
            6: "Process terminated.",
        }

        self.sys_colors = {
            1: (40, 40, 100, 255),   # window frame
            2: (20, 20, 60, 255),    # title bar
            3: (0, 0, 40, 255),      # background
            4: (255, 255, 255, 255), # white text
            5: (200, 200, 255, 255), # light blue text
        }

        self.sys_layers = {
            1: "background",
            2: "ui",
            3: "vm",
            4: "overlay",
        }

        # File ID to path mapping (for syscalls)
        self.file_paths = {
            300: "build/kernel_v2.asm",
            301: "build/kernel_v2.bin",
            302: "build/assembler.asm",
            303: "build/assembler.bin",
        }

    def create_process(self, bytecode: Optional[bytes] = None) -> int:
        """Create a new process. Returns PID."""
        pid = self.next_pid
        self.next_pid += 1
        proc = Process(pid=pid)
        if bytecode:
            proc.memory[:len(bytecode)] = bytecode
        self.processes[pid] = proc
        return pid

    def load_program(self, bytecode: bytes) -> int:
        """Load program as PID 1 (init). Returns PID."""
        pid = self.create_process(bytecode)
        self.current_pid = pid
        return pid

    def fetch_byte(self, proc: Process) -> int:
        """Fetch next byte from process memory"""
        if proc.pc >= len(proc.memory):
            raise RuntimeError(f"PC out of bounds: {proc.pc}")
        byte = proc.memory[proc.pc]
        proc.pc += 1
        return byte

    def fetch_int32(self, proc: Process) -> int:
        """Fetch 32-bit integer (little-endian)"""
        b0 = self.fetch_byte(proc)
        b1 = self.fetch_byte(proc)
        b2 = self.fetch_byte(proc)
        b3 = self.fetch_byte(proc)
        val = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
        if val >= 0x80000000:
            val -= 0x100000000
        return val

    def read_memory_int32(self, proc: Process, addr: int) -> int:
        """Read 32-bit int from memory address"""
        if addr < 0 or addr + 3 >= len(proc.memory):
            return 0  # imperfect mode: safe default
        b0 = proc.memory[addr]
        b1 = proc.memory[addr + 1]
        b2 = proc.memory[addr + 2]
        b3 = proc.memory[addr + 3]
        val = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
        if val >= 0x80000000:
            val -= 0x100000000
        return val

    def write_memory_int32(self, proc: Process, addr: int, val: int):
        """Write 32-bit int to memory address"""
        if addr < 0 or addr + 3 >= len(proc.memory):
            return  # imperfect mode: ignore out of bounds
        # Convert to unsigned
        if val < 0:
            val += 0x100000000
        proc.memory[addr] = val & 0xFF
        proc.memory[addr + 1] = (val >> 8) & 0xFF
        proc.memory[addr + 2] = (val >> 16) & 0xFF
        proc.memory[addr + 3] = (val >> 24) & 0xFF

    def handle_syscall(self, proc: Process, num: int):
        """Handle syscall - extended with filesystem and IPC"""
        r = proc.registers

        try:
            # Graphics syscalls (from base pxVM)
            if num == SYS_PRINT_ID:
                msg_id = r[1]
                msg = self.sys_messages.get(msg_id)
                if msg is None:
                    self.sysout.append(f"PRINT [vm warn] unknown message_id {msg_id}")
                else:
                    self.sysout.append(f"PRINT PXVM: {msg}")

            elif num == SYS_RECT_ID:
                x, y, w, h = r[1], r[2], r[3], r[4]
                color_id = r[5]
                rgba = self.sys_colors.get(color_id, (255, 0, 255, 255))
                if color_id not in self.sys_colors:
                    self.sysout.append(f"# WARNING: unknown color_id {color_id}")
                r_, g_, b_, a_ = rgba
                self.sysout.append(f"RECT {x} {y} {w} {h} {r_} {g_} {b_} {a_}")

            elif num == SYS_TEXT_ID:
                x, y = r[1], r[2]
                color_id = r[3]
                msg_id = r[4]
                rgba = self.sys_colors.get(color_id, (255, 255, 255, 255))
                msg = self.sys_messages.get(msg_id, f"[unknown msg {msg_id}]")
                r_, g_, b_, a_ = rgba
                self.sysout.append(f"TEXT {x} {y} {r_} {g_} {b_} {a_} {msg}")

            elif num == SYS_LAYER_USE_ID:
                layer_id = r[1]
                name = self.sys_layers.get(layer_id)
                if name:
                    self.sysout.append(f"SELECT {name}")

            # IPC syscalls
            elif num == SYS_IPC_SEND:
                # R1 = target_pid, R2 = msg_type
                target_pid = r[1]
                msg_type = r[2]
                if target_pid in self.processes:
                    msg = Message(from_pid=proc.pid, to_pid=target_pid, msg_type=msg_type)
                    self.processes[target_pid].message_queue.append(msg)
                    # Wake up target if waiting
                    self.processes[target_pid].waiting_for_ipc = False
                    r[0] = 1  # success
                else:
                    if self.imperfect:
                        self.sysout.append(f"# IPC warning: PID {target_pid} not found")
                    r[0] = 0  # failure

            elif num == SYS_IPC_RECV:
                # Blocking receive - check message queue
                if proc.message_queue:
                    msg = proc.message_queue.pop(0)
                    r[0] = msg.msg_type
                    r[1] = msg.from_pid
                    proc.waiting_for_ipc = False
                else:
                    # No messages - block this process
                    proc.waiting_for_ipc = True
                    r[0] = 0

            # Filesystem syscalls
            elif num == SYS_FS_OPEN:
                # R1 = file_id, R2 = mode (0=read, 1=write, 2=create)
                file_id = r[1]
                mode = r[2]
                path = self.file_paths.get(file_id, f"file_{file_id}")

                if self.filesystem.open(path, mode):
                    handle_id = proc.next_handle_id
                    proc.next_handle_id += 1
                    proc.file_handles[handle_id] = FileHandle(
                        file_id=file_id, path=path, mode=mode
                    )
                    r[0] = handle_id
                else:
                    r[0] = 0  # failure

            elif num == SYS_FS_CLOSE:
                # R1 = handle
                handle_id = r[1]
                if handle_id in proc.file_handles:
                    del proc.file_handles[handle_id]
                    r[0] = 1
                else:
                    r[0] = 0

            elif num == SYS_FS_READ:
                # R1 = handle, R2 = buffer_addr, R3 = max_length
                handle_id = r[1]
                buffer_addr = r[2]
                max_length = r[3]

                if handle_id not in proc.file_handles:
                    r[0] = 0
                    return

                fh = proc.file_handles[handle_id]
                data = self.filesystem.read(fh.path, fh.pos, max_length)

                if data is None:
                    r[0] = 0
                    return

                # Copy to process memory
                bytes_read = min(len(data), max_length)
                if buffer_addr >= 0 and buffer_addr + bytes_read <= len(proc.memory):
                    proc.memory[buffer_addr:buffer_addr+bytes_read] = data[:bytes_read]
                    fh.pos += bytes_read
                    r[0] = bytes_read
                else:
                    r[0] = 0  # invalid buffer

            elif num == SYS_FS_WRITE:
                # R1 = handle, R2 = buffer_addr, R3 = length
                handle_id = r[1]
                buffer_addr = r[2]
                length = r[3]

                if handle_id not in proc.file_handles:
                    r[0] = 0
                    return

                fh = proc.file_handles[handle_id]

                # Read from process memory
                if buffer_addr >= 0 and buffer_addr + length <= len(proc.memory):
                    data = bytes(proc.memory[buffer_addr:buffer_addr+length])
                    if self.filesystem.write(fh.path, fh.pos, data):
                        fh.pos += length
                        r[0] = length
                    else:
                        r[0] = 0
                else:
                    r[0] = 0  # invalid buffer

            # Process management syscalls
            elif num == SYS_FORK:
                # Create child process with copy of memory
                child_proc = copy.deepcopy(proc)
                child_pid = self.next_pid
                self.next_pid += 1
                child_proc.pid = child_pid
                child_proc.registers[0] = 0  # child returns 0
                self.processes[child_pid] = child_proc
                r[0] = child_pid  # parent returns child PID

            elif num == SYS_SPAWN:
                # R1 = file_id (load and spawn new process)
                file_id = r[1]
                path = self.file_paths.get(file_id, f"file_{file_id}")

                # Try to read entire file
                data = self.filesystem.read(path, 0, 65536)
                if data:
                    new_pid = self.create_process(bytes(data))
                    r[0] = new_pid
                else:
                    r[0] = 0  # failure

            else:
                # Unknown syscall
                if self.imperfect:
                    self.sysout.append(f"# WARNING: unknown syscall {num}")
                r[0] = 0

        except Exception as e:
            if self.imperfect:
                self.sysout.append(f"# ERROR in syscall {num}: {type(e).__name__}: {e}")
                r[0] = 0  # syscall failed
            else:
                raise

    def step(self, proc: Process) -> bool:
        """Execute one instruction. Returns True if should continue."""
        if proc.halted or proc.waiting_for_ipc:
            return False

        try:
            opcode = self.fetch_byte(proc)
            r = proc.registers

            if opcode == OP_HALT:
                proc.halted = True
                return False

            elif opcode == OP_NOP:
                pass

            elif opcode == OP_IMM8:
                reg = self.fetch_byte(proc)
                val = self.fetch_byte(proc)
                if 0 <= reg < 8:
                    r[reg] = val

            elif opcode == OP_IMM32:
                reg = self.fetch_byte(proc)
                val = self.fetch_int32(proc)
                if 0 <= reg < 8:
                    r[reg] = val

            elif opcode == OP_MOV:
                dst = self.fetch_byte(proc)
                src = self.fetch_byte(proc)
                if 0 <= dst < 8 and 0 <= src < 8:
                    r[dst] = r[src]

            elif opcode == OP_ADD:
                dst = self.fetch_byte(proc)
                src1 = self.fetch_byte(proc)
                src2 = self.fetch_byte(proc)
                if 0 <= dst < 8 and 0 <= src1 < 8 and 0 <= src2 < 8:
                    result = r[src1] + r[src2]
                    if result >= 0x80000000:
                        result -= 0x100000000
                    elif result < -0x80000000:
                        result += 0x100000000
                    r[dst] = result

            elif opcode == OP_SUB:
                dst = self.fetch_byte(proc)
                src1 = self.fetch_byte(proc)
                src2 = self.fetch_byte(proc)
                if 0 <= dst < 8 and 0 <= src1 < 8 and 0 <= src2 < 8:
                    result = r[src1] - r[src2]
                    if result >= 0x80000000:
                        result -= 0x100000000
                    elif result < -0x80000000:
                        result += 0x100000000
                    r[dst] = result

            elif opcode == OP_JMP:
                addr = self.fetch_int32(proc)
                if 0 <= addr < len(proc.memory):
                    proc.pc = addr

            elif opcode == OP_JZ:
                reg = self.fetch_byte(proc)
                addr = self.fetch_int32(proc)
                if 0 <= reg < 8 and r[reg] == 0:
                    if 0 <= addr < len(proc.memory):
                        proc.pc = addr

            elif opcode == OP_CMP:
                dst = self.fetch_byte(proc)
                reg1 = self.fetch_byte(proc)
                reg2 = self.fetch_byte(proc)
                if 0 <= dst < 8 and 0 <= reg1 < 8 and 0 <= reg2 < 8:
                    r[dst] = 1 if r[reg1] == r[reg2] else 0

            elif opcode == OP_LOAD:
                dst = self.fetch_byte(proc)
                addr = self.fetch_int32(proc)
                if 0 <= dst < 8:
                    r[dst] = self.read_memory_int32(proc, addr)

            elif opcode == OP_STORE:
                addr = self.fetch_int32(proc)
                src = self.fetch_byte(proc)
                if 0 <= src < 8:
                    self.write_memory_int32(proc, addr, r[src])

            elif opcode == OP_SYSCALL:
                num = self.fetch_byte(proc)
                self.handle_syscall(proc, num)

            else:
                if self.imperfect:
                    self.sysout.append(f"# WARNING: unknown opcode 0x{opcode:02X}")
                else:
                    raise RuntimeError(f"Unknown opcode: 0x{opcode:02X}")

            return True

        except Exception as e:
            if self.imperfect:
                self.sysout.append(f"# ERROR at PC={proc.pc}: {type(e).__name__}: {e}")
                proc.halted = True
                return False
            else:
                raise

    def run(self, max_cycles: int = 100000):
        """Run scheduler - round-robin through all processes"""
        cycles = 0

        while cycles < max_cycles:
            # Check if any process can run
            runnable = [p for p in self.processes.values() if not p.halted and not p.waiting_for_ipc]
            if not runnable:
                # All processes halted or waiting
                break

            # Round-robin: run each process for 1 instruction
            for proc in runnable:
                if not self.step(proc):
                    continue
                cycles += 1
                if cycles >= max_cycles:
                    break

        return cycles
