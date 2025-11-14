#!/usr/bin/env python3
"""
spirv_terminal.py - pxVM v0.3
A SPIR-V terminal with CPU-based pxVM interpreter + syscalls
"""

import sys
import struct
from pathlib import Path
from typing import Optional

# Import assembler if available
try:
    from pxvm_assembler import PxVmAssembler
    ASSEMBLER_AVAILABLE = True
except ImportError:
    ASSEMBLER_AVAILABLE = False

# ========== Virtual Filesystem ==========

class FileHandle:
    """Represents an open file descriptor"""
    def __init__(self, name: str, mode: int, data: bytearray):
        self.name = name
        self.mode = mode  # O_RDONLY=0, O_WRONLY=1, O_RDWR=2
        self.pos = 0
        self.data = data

class VirtualFS:
    """In-memory virtual filesystem for pxVM v0.3"""

    # Error codes
    ERR_SUCCESS = 0
    ERR_INVALID = -1
    ERR_BADFD = -2
    ERR_NOMEM = -3
    ERR_FAULT = -4
    ERR_NOTFOUND = -5
    ERR_NOSPACE = -6
    ERR_BADADDR = -7

    # File flags
    O_RDONLY = 0x00
    O_WRONLY = 0x01
    O_RDWR = 0x02
    O_CREATE = 0x40
    O_TRUNC = 0x80

    # Standard file descriptors
    FD_STDIN = 0
    FD_STDOUT = 1
    FD_STDERR = 2

    def __init__(self):
        # I/O buffers
        self.stdin_buffer = bytearray()
        self.stdout_buffer = bytearray()
        self.stderr_buffer = bytearray()

        # Files (name -> bytearray)
        self.files = {}

        # File descriptor table
        self.fd_table = [
            FileHandle("<stdin>", self.O_RDONLY, self.stdin_buffer),
            FileHandle("<stdout>", self.O_WRONLY, self.stdout_buffer),
            FileHandle("<stderr>", self.O_WRONLY, self.stderr_buffer),
        ]

    def sys_write(self, fd: int, buf_addr: int, count: int, mem: bytearray) -> int:
        """SYS_WRITE (1): Write data to file descriptor"""
        # Validate fd
        if fd < 0 or fd >= len(self.fd_table) or self.fd_table[fd] is None:
            return self.ERR_BADFD

        # Validate memory address
        if buf_addr < 0 or buf_addr + count > len(mem):
            return self.ERR_BADADDR

        # Get file handle
        fh = self.fd_table[fd]

        # Read data from memory
        data = bytes(mem[buf_addr:buf_addr + count])

        # Write to file
        if fd == self.FD_STDOUT or fd == self.FD_STDERR:
            # Write to stdout/stderr and print
            fh.data.extend(data)
            try:
                text = data.decode('utf-8')
                print(text, end='')
            except UnicodeDecodeError:
                # Print as hex if not valid UTF-8
                print(f"[binary: {data.hex()}]", end='')
        else:
            # Write to file at current position
            end_pos = fh.pos + count
            if end_pos > len(fh.data):
                fh.data.extend(b'\x00' * (end_pos - len(fh.data)))
            fh.data[fh.pos:fh.pos + count] = data
            fh.pos += count

        return count

    def sys_read(self, fd: int, buf_addr: int, count: int, mem: bytearray) -> int:
        """SYS_READ (2): Read data from file descriptor"""
        # Validate fd
        if fd < 0 or fd >= len(self.fd_table) or self.fd_table[fd] is None:
            return self.ERR_BADFD

        # Validate memory address
        if buf_addr < 0 or buf_addr + count > len(mem):
            return self.ERR_BADADDR

        # Get file handle
        fh = self.fd_table[fd]

        # Read from file
        available = len(fh.data) - fh.pos
        to_read = min(count, available)

        if to_read <= 0:
            return 0  # EOF

        # Copy to memory
        data = fh.data[fh.pos:fh.pos + to_read]
        mem[buf_addr:buf_addr + to_read] = data
        fh.pos += to_read

        return to_read

    def sys_open(self, path_addr: int, flags: int, mem: bytearray) -> int:
        """SYS_OPEN (3): Open a file and return fd"""
        # Read null-terminated string from memory
        if path_addr < 0 or path_addr >= len(mem):
            return self.ERR_BADADDR

        # Find null terminator
        path_end = path_addr
        while path_end < len(mem) and mem[path_end] != 0:
            path_end += 1

        if path_end >= len(mem):
            return self.ERR_FAULT  # No null terminator

        # Decode path
        try:
            path = mem[path_addr:path_end].decode('utf-8')
        except UnicodeDecodeError:
            return self.ERR_FAULT

        # Check if file exists
        if path not in self.files:
            if flags & self.O_CREATE:
                # Create new file
                self.files[path] = bytearray()
            else:
                return self.ERR_NOTFOUND

        # Truncate if requested
        if flags & self.O_TRUNC:
            self.files[path] = bytearray()

        # Allocate fd
        mode = flags & 0x03  # Extract O_RDONLY/O_WRONLY/O_RDWR
        fh = FileHandle(path, mode, self.files[path])

        # Find free slot in fd table
        for i in range(3, len(self.fd_table)):
            if self.fd_table[i] is None:
                self.fd_table[i] = fh
                return i

        # Append new fd
        if len(self.fd_table) >= 256:
            return self.ERR_NOMEM

        self.fd_table.append(fh)
        return len(self.fd_table) - 1

    def sys_close(self, fd: int) -> int:
        """SYS_CLOSE (4): Close a file descriptor"""
        # Cannot close stdin/stdout/stderr
        if fd < 3 or fd >= len(self.fd_table):
            return self.ERR_BADFD

        if self.fd_table[fd] is None:
            return self.ERR_BADFD

        self.fd_table[fd] = None
        return self.ERR_SUCCESS

    def feed_stdin(self, text: str):
        """Add text to stdin buffer"""
        self.stdin_buffer.extend(text.encode('utf-8'))

    def get_stdout(self) -> str:
        """Get and clear stdout buffer"""
        result = self.stdout_buffer.decode('utf-8', errors='replace')
        self.stdout_buffer.clear()
        return result

    def get_stderr(self) -> str:
        """Get and clear stderr buffer"""
        result = self.stderr_buffer.decode('utf-8', errors='replace')
        self.stderr_buffer.clear()
        return result

class SpirvTerminal:
    def __init__(self):
        # Storage buffers (name -> bytearray)
        self.buffers = {}

        # CPU kernel registry (name -> function)
        self.kernels = {
            'add1': self.kernel_add1,
            'add_const': self.kernel_add_const,
            'mul_const': self.kernel_mul_const,
            'memcpy': self.kernel_memcpy,
        }

        # Loaded modules (name -> kernel_name)
        self.modules = {}

        # Module bindings (module_name -> {binding_index -> buffer_name})
        self.bindings = {}

        # pxVM programs (name -> bytes)
        self.programs = {}

        # Virtual filesystem (v0.3)
        self.vfs = VirtualFS()

        print("spirv_terminal v0.3 (CPU backend + pxVM + syscalls)")

    # ========== pxVM v0.2 Interpreter ==========

    def run_pxvm(self, program_bytes):
        """Execute a pxVM v0.2 program"""
        # VM state
        regs = [0] * 8  # R0-R7
        mem = bytearray(4096)  # 4KB memory
        stack = []  # Call stack (stores return addresses)
        pc = 0  # Program counter
        halted = False

        # Opcode definitions
        OP_MOV = 0x10
        OP_ADD = 0x20
        OP_SUB = 0x21
        OP_MUL = 0x22
        OP_DIV = 0x23
        OP_IMM32 = 0x30
        OP_LOAD = 0x40
        OP_STORE = 0x41
        OP_PUSH = 0x50
        OP_POP = 0x51
        OP_CALL = 0x60
        OP_RET = 0x61
        OP_JMP = 0x70
        OP_JZ = 0x71
        OP_JNZ = 0x72
        OP_SYSCALL = 0x80
        OP_PRINT = 0xF0
        OP_HALT = 0xFF

        # Syscall numbers
        SYS_WRITE = 1
        SYS_READ = 2
        SYS_OPEN = 3
        SYS_CLOSE = 4

        max_cycles = 10000  # Safety limit
        cycle_count = 0

        while not halted and cycle_count < max_cycles:
            cycle_count += 1

            if pc >= len(program_bytes):
                print(f"ERR PC out of bounds: {pc}")
                break

            opcode = program_bytes[pc]
            pc += 1

            # MOV rd, rs
            if opcode == OP_MOV:
                if pc + 1 >= len(program_bytes):
                    break
                rd = program_bytes[pc]
                rs = program_bytes[pc + 1]
                pc += 2
                if rd < 8 and rs < 8:
                    regs[rd] = regs[rs]

            # ADD rd, rs, rt
            elif opcode == OP_ADD:
                if pc + 2 >= len(program_bytes):
                    break
                rd = program_bytes[pc]
                rs = program_bytes[pc + 1]
                rt = program_bytes[pc + 2]
                pc += 3
                if rd < 8 and rs < 8 and rt < 8:
                    regs[rd] = (regs[rs] + regs[rt]) & 0xFFFFFFFF

            # SUB rd, rs, rt
            elif opcode == OP_SUB:
                if pc + 2 >= len(program_bytes):
                    break
                rd = program_bytes[pc]
                rs = program_bytes[pc + 1]
                rt = program_bytes[pc + 2]
                pc += 3
                if rd < 8 and rs < 8 and rt < 8:
                    regs[rd] = (regs[rs] - regs[rt]) & 0xFFFFFFFF

            # MUL rd, rs, rt
            elif opcode == OP_MUL:
                if pc + 2 >= len(program_bytes):
                    break
                rd = program_bytes[pc]
                rs = program_bytes[pc + 1]
                rt = program_bytes[pc + 2]
                pc += 3
                if rd < 8 and rs < 8 and rt < 8:
                    regs[rd] = (regs[rs] * regs[rt]) & 0xFFFFFFFF

            # DIV rd, rs, rt
            elif opcode == OP_DIV:
                if pc + 2 >= len(program_bytes):
                    break
                rd = program_bytes[pc]
                rs = program_bytes[pc + 1]
                rt = program_bytes[pc + 2]
                pc += 3
                if rd < 8 and rs < 8 and rt < 8:
                    if regs[rt] != 0:
                        regs[rd] = (regs[rs] // regs[rt]) & 0xFFFFFFFF

            # IMM32 rd, <4-byte-value>
            elif opcode == OP_IMM32:
                if pc + 4 >= len(program_bytes):
                    break
                rd = program_bytes[pc]
                pc += 1
                val = struct.unpack('<I', program_bytes[pc:pc+4])[0]
                pc += 4
                if rd < 8:
                    regs[rd] = val

            # LOAD rd, [rs]
            elif opcode == OP_LOAD:
                if pc + 1 >= len(program_bytes):
                    break
                rd = program_bytes[pc]
                rs = program_bytes[pc + 1]
                pc += 2
                if rd < 8 and rs < 8:
                    addr = regs[rs]
                    if addr + 3 < len(mem):
                        regs[rd] = struct.unpack('<I', mem[addr:addr+4])[0]

            # STORE [rd], rs
            elif opcode == OP_STORE:
                if pc + 1 >= len(program_bytes):
                    break
                rd = program_bytes[pc]
                rs = program_bytes[pc + 1]
                pc += 2
                if rd < 8 and rs < 8:
                    addr = regs[rd]
                    if addr + 3 < len(mem):
                        struct.pack_into('<I', mem, addr, regs[rs] & 0xFFFFFFFF)

            # PUSH rs
            elif opcode == OP_PUSH:
                if pc >= len(program_bytes):
                    break
                rs = program_bytes[pc]
                pc += 1
                if rs < 8:
                    stack.append(regs[rs])

            # POP rd
            elif opcode == OP_POP:
                if pc >= len(program_bytes):
                    break
                rd = program_bytes[pc]
                pc += 1
                if rd < 8 and len(stack) > 0:
                    regs[rd] = stack.pop()

            # CALL <2-byte-addr>
            elif opcode == OP_CALL:
                if pc + 1 >= len(program_bytes):
                    break
                addr = struct.unpack('<H', program_bytes[pc:pc+2])[0]
                pc += 2
                stack.append(pc)  # Push return address
                pc = addr  # Jump to function

            # RET
            elif opcode == OP_RET:
                if len(stack) > 0:
                    pc = stack.pop()  # Pop return address
                else:
                    halted = True

            # JMP <2-byte-addr>
            elif opcode == OP_JMP:
                if pc + 1 >= len(program_bytes):
                    break
                addr = struct.unpack('<H', program_bytes[pc:pc+2])[0]
                pc = addr

            # JZ rs, <2-byte-addr>
            elif opcode == OP_JZ:
                if pc + 2 >= len(program_bytes):
                    break
                rs = program_bytes[pc]
                pc += 1
                addr = struct.unpack('<H', program_bytes[pc:pc+2])[0]
                pc += 2
                if rs < 8 and regs[rs] == 0:
                    pc = addr

            # JNZ rs, <2-byte-addr>
            elif opcode == OP_JNZ:
                if pc + 2 >= len(program_bytes):
                    break
                rs = program_bytes[pc]
                pc += 1
                addr = struct.unpack('<H', program_bytes[pc:pc+2])[0]
                pc += 2
                if rs < 8 and regs[rs] != 0:
                    pc = addr

            # SYSCALL
            elif opcode == OP_SYSCALL:
                syscall_num = regs[0] & 0xFFFFFFFF
                if syscall_num == SYS_WRITE:
                    # sys_write(fd, buf_addr, count)
                    fd = regs[1] & 0xFFFFFFFF
                    buf_addr = regs[2] & 0xFFFFFFFF
                    count = regs[3] & 0xFFFFFFFF
                    result = self.vfs.sys_write(fd, buf_addr, count, mem)
                    regs[0] = result & 0xFFFFFFFF
                elif syscall_num == SYS_READ:
                    # sys_read(fd, buf_addr, count)
                    fd = regs[1] & 0xFFFFFFFF
                    buf_addr = regs[2] & 0xFFFFFFFF
                    count = regs[3] & 0xFFFFFFFF
                    result = self.vfs.sys_read(fd, buf_addr, count, mem)
                    regs[0] = result & 0xFFFFFFFF
                elif syscall_num == SYS_OPEN:
                    # sys_open(path_addr, flags)
                    path_addr = regs[1] & 0xFFFFFFFF
                    flags = regs[2] & 0xFFFFFFFF
                    result = self.vfs.sys_open(path_addr, flags, mem)
                    regs[0] = result & 0xFFFFFFFF
                elif syscall_num == SYS_CLOSE:
                    # sys_close(fd)
                    fd = regs[1] & 0xFFFFFFFF
                    result = self.vfs.sys_close(fd)
                    regs[0] = result & 0xFFFFFFFF
                else:
                    # Unknown syscall
                    regs[0] = VirtualFS.ERR_INVALID

            # PRINT rd
            elif opcode == OP_PRINT:
                if pc >= len(program_bytes):
                    break
                rd = program_bytes[pc]
                pc += 1
                if rd < 8:
                    print(f"PRINT R{rd} = {regs[rd]}")

            # HALT
            elif opcode == OP_HALT:
                halted = True

            else:
                print(f"ERR unknown opcode 0x{opcode:02X} at PC {pc-1}")
                halted = True

        if cycle_count >= max_cycles:
            print(f"ERR max cycles ({max_cycles}) exceeded")

    # ========== CPU Kernel Implementations ==========

    def kernel_add1(self, in_buf, out_buf, count):
        """Add 1 to each u32 element"""
        for i in range(count):
            val = struct.unpack_from('<I', in_buf, i*4)[0]
            struct.pack_into('<I', out_buf, i*4, (val + 1) & 0xFFFFFFFF)

    def kernel_add_const(self, in_buf, out_buf, count, const):
        """Add a constant to each u32 element"""
        for i in range(count):
            val = struct.unpack_from('<I', in_buf, i*4)[0]
            struct.pack_into('<I', out_buf, i*4, (val + const) & 0xFFFFFFFF)

    def kernel_mul_const(self, in_buf, out_buf, count, const):
        """Multiply each u32 element by a constant"""
        for i in range(count):
            val = struct.unpack_from('<I', in_buf, i*4)[0]
            struct.pack_into('<I', out_buf, i*4, (val * const) & 0xFFFFFFFF)

    def kernel_memcpy(self, src_buf, dst_buf, count):
        """Copy bytes from src to dst"""
        dst_buf[:count] = src_buf[:count]

    # ========== Terminal Commands ==========

    def cmd_NEWBUF(self, name, size_str):
        """Create a new buffer: NEWBUF <name> <size_bytes>"""
        size = int(size_str)
        self.buffers[name] = bytearray(size)
        print(f"OK NEWBUF {name} ({size} bytes)")

    def cmd_WRITE(self, name, offset_str, *hex_bytes):
        """Write hex data to buffer: WRITE <name> <offset> <hex>..."""
        if name not in self.buffers:
            print(f"ERR buffer {name} not found")
            return
        offset = int(offset_str)
        data = bytes.fromhex(' '.join(hex_bytes))
        self.buffers[name][offset:offset+len(data)] = data
        print(f"OK WRITE {name}")

    def cmd_READ(self, name, offset_str, count_str):
        """Read hex data from buffer: READ <name> <offset> <count>"""
        if name not in self.buffers:
            print(f"ERR buffer {name} not found")
            return
        offset = int(offset_str)
        count = int(count_str)
        data = self.buffers[name][offset:offset+count]
        hex_str = ' '.join(f"{b:02X}" for b in data)
        print(f"DATA {hex_str}")

    def cmd_LOADMOD(self, name, path):
        """Load a WGSL module: LOADMOD <name> <path>"""
        # For CPU backend, just map name to kernel
        kernel_name = name.split('.')[0]  # e.g., "add1.wgsl" -> "add1"
        if kernel_name in self.kernels:
            self.modules[name] = kernel_name
            print(f"OK LOADMOD {name}")
        else:
            print(f"ERR kernel {kernel_name} not in registry")

    def cmd_BIND(self, module, binding_str, buffer):
        """Bind a buffer to a module: BIND <module> <binding> <buffer>"""
        if module not in self.modules:
            print(f"ERR module {module} not loaded")
            return
        if buffer not in self.buffers:
            print(f"ERR buffer {buffer} not found")
            return
        if module not in self.bindings:
            self.bindings[module] = {}
        binding = int(binding_str)
        self.bindings[module][binding] = buffer
        print(f"OK BIND {module}.{binding} -> {buffer}")

    def cmd_DISPATCH(self, module, count_str, *args):
        """Dispatch a kernel: DISPATCH <module> <count> [args...]"""
        if module not in self.modules:
            print(f"ERR module {module} not loaded")
            return
        kernel_name = self.modules[module]
        kernel_func = self.kernels[kernel_name]

        # Get bound buffers
        bindings = self.bindings.get(module, {})
        bound_buffers = [self.buffers[bindings[i]] for i in sorted(bindings.keys())]

        # Convert args to integers
        int_args = [int(a) for a in args]

        # Execute kernel
        try:
            kernel_func(*bound_buffers, int(count_str), *int_args)
            print(f"OK DISPATCH {module}")
        except Exception as e:
            print(f"ERR DISPATCH {module}: {e}")

    def cmd_SCRIPT(self, path):
        """Execute a script file: SCRIPT <path>"""
        try:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self.execute_line(line)
            print(f"OK SCRIPT {path}")
        except FileNotFoundError:
            print(f"ERR script {path} not found")
        except Exception as e:
            print(f"ERR SCRIPT {path}: {e}")

    def cmd_WRITEFILE(self, path, *hex_bytes):
        """Write hex data to a file: WRITEFILE <path> <hex>..."""
        try:
            data = bytes.fromhex(' '.join(hex_bytes))
            with open(path, 'wb') as f:
                f.write(data)
            print(f"OK WRITEFILE {path} ({len(data)} bytes)")
        except Exception as e:
            print(f"ERR WRITEFILE {path}: {e}")

    def cmd_LOADIMG(self, name, path):
        """Load a pxVM program: LOADIMG <name> <path>"""
        try:
            with open(path, 'rb') as f:
                data = f.read()
            self.programs[name] = data
            print(f"OK LOADIMG {name} ({len(data)} bytes)")
        except FileNotFoundError:
            print(f"ERR image {path} not found")
        except Exception as e:
            print(f"ERR LOADIMG {name}: {e}")

    def cmd_RUN(self, name):
        """Run a pxVM program: RUN <name>"""
        if name not in self.programs:
            print(f"ERR program {name} not loaded")
            return
        print(f"OK RUN {name}")
        self.run_pxvm(self.programs[name])

    def cmd_ASM(self, name, path):
        """Assemble .pxasm to .pxvm: ASM <name> <path.pxasm>"""
        if not ASSEMBLER_AVAILABLE:
            print("ERR assembler not available (pxvm_assembler.py missing)")
            return

        try:
            with open(path, 'r') as f:
                source = f.read()

            assembler = PxVmAssembler()
            bytecode = assembler.assemble(source)

            # Auto-generate .pxvm filename
            pxvm_path = Path(path).with_suffix('.pxvm')

            with open(pxvm_path, 'wb') as f:
                f.write(bytecode)

            # Load into programs
            self.programs[name] = bytecode

            print(f"OK ASM {name} ({len(bytecode)} bytes → {pxvm_path})")

            # Show label addresses
            if assembler.labels:
                for label, addr in sorted(assembler.labels.items(), key=lambda x: x[1]):
                    print(f"  {label:20s} = 0x{addr:04X}")

        except FileNotFoundError:
            print(f"ERR file not found: {path}")
        except Exception as e:
            print(f"ERR ASM: {e}")

    def cmd_DISASM(self, name):
        """Disassemble a program: DISASM <name>"""
        if name not in self.programs:
            print(f"ERR program {name} not loaded")
            return

        if not ASSEMBLER_AVAILABLE:
            print("ERR assembler not available (pxvm_assembler.py missing)")
            return

        try:
            assembler = PxVmAssembler()
            asm = assembler.disassemble(self.programs[name])
            print(asm)
        except Exception as e:
            print(f"ERR DISASM: {e}")

    # ========== Virtual Filesystem Commands (v0.3) ==========

    def cmd_FEED(self, *text_parts):
        """Add text to stdin buffer: FEED <text>"""
        text = ' '.join(text_parts)
        self.vfs.feed_stdin(text)
        print(f"OK FEED ({len(text)} bytes to stdin)")

    def cmd_STDOUT(self):
        """Print and clear stdout buffer: STDOUT"""
        output = self.vfs.get_stdout()
        if output:
            print(output, end='')
        print(f"OK STDOUT ({len(output)} bytes)")

    def cmd_STDERR(self):
        """Print and clear stderr buffer: STDERR"""
        output = self.vfs.get_stderr()
        if output:
            print(output, end='')
        print(f"OK STDERR ({len(output)} bytes)")

    def cmd_LS(self):
        """List files in virtual filesystem: LS"""
        if not self.vfs.files:
            print("(no files)")
        else:
            for name, data in self.vfs.files.items():
                print(f"{name:30s} {len(data):8d} bytes")
        print(f"OK LS ({len(self.vfs.files)} files)")

    def cmd_CAT(self, filename):
        """Print file contents: CAT <filename>"""
        if filename not in self.vfs.files:
            print(f"ERR file not found: {filename}")
            return
        data = self.vfs.files[filename]
        try:
            text = data.decode('utf-8')
            print(text)
        except UnicodeDecodeError:
            # Print as hex if not valid UTF-8
            print(f"[binary file, {len(data)} bytes]")
            print(' '.join(f"{b:02X}" for b in data))
        print(f"OK CAT {filename}")

    def execute_line(self, line):
        """Execute a single command line"""
        line = line.strip()
        if not line or line.startswith('#'):
            return
        parts = line.split()
        cmd = parts[0].upper()
        args = parts[1:]
        handler = getattr(self, f"cmd_{cmd}", None)
        if handler is None:
            print(f"ERR unknown cmd {cmd}")
        else:
            try:
                handler(*args)
            except TypeError as e:
                print(f"ERR bad args for {cmd}: {e}")

    def repl(self):
        """Read-Eval-Print Loop"""
        while True:
            try:
                line = input("> ")
                self.execute_line(line)
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nBye!")
                break

def main():
    term = SpirvTerminal()
    term.repl()

if __name__ == "__main__":
    main()
