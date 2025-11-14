#!/usr/bin/env python3
"""
spirv_terminal.py - pxVM v0.2
A SPIR-V terminal with CPU-based pxVM interpreter
"""

import sys
import struct

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

        print("spirv_terminal v0.2 (CPU backend + pxVM)")

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
        OP_PRINT = 0xF0
        OP_HALT = 0xFF

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
