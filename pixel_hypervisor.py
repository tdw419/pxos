#!/usr/bin/env python3
"""
pixel_hypervisor.py - Execute guest code on the pixel substrate

The hypervisor sits between PXICPU (pixel execution) and guest runtimes (Python, future languages).

Architecture:
  Host OS
    └─ CPython
        └─ Pixel Hypervisor ← YOU ARE HERE
            ├─ PXICPU (pixel-native code)
            └─ Guest runtimes (Python, PXLang, etc.)

What the hypervisor does:
  1. Runs PXICPU instructions (pixel-native code)
  2. Watches for trap flags (syscalls from guests)
  3. Loads and executes guest code (Python modules from pixels)
  4. Provides PixelBridge API to guests
  5. Manages imperfect computing (noise, probabilistic ops)

Trap Interface:
  Memory layout for traps:
    TRAP_FLAG_ADDR (0xF000):     0 = none, >0 = trap type
    TRAP_ARG0_ADDR (0xF001-F004): First argument (32-bit)
    TRAP_ARG1_ADDR (0xF005-F008): Second argument (32-bit)
    TRAP_ARG2_ADDR (0xF009-F00C): Third argument (32-bit)
    TRAP_RESULT_ADDR (0xF00D):    Result/status from hypervisor

Trap Types:
    0x01: RUN_PYTHON - Execute Python guest module
    0x02: CALL_LLM - Call LLM via PXDigest
    0x03: LOAD_FILE - Load file via SYS_BLOB
    0x04: SAMPLE_NOISE - Get random/noisy value
    0x05: ORACLE - Ask LLM for probabilistic decision

Usage:
    python3 pixel_hypervisor.py --demo
    python3 pixel_hypervisor.py --run-guest /apps/my_module.py
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional, Dict, Any
import sys

# Import pxOS components
from pxi_cpu import PXICPU, query_local_llm_via_digest
from PIL import Image

# Trap memory addresses
TRAP_FLAG_ADDR = 0xF000
TRAP_ARG0_ADDR = 0xF001
TRAP_ARG1_ADDR = 0xF005
TRAP_ARG2_ADDR = 0xF009
TRAP_RESULT_ADDR = 0xF00D

# Trap types
TRAP_RUN_PYTHON = 0x01
TRAP_CALL_LLM = 0x02
TRAP_LOAD_FILE = 0x03
TRAP_SAMPLE_NOISE = 0x04
TRAP_ORACLE = 0x05

# Guest memory region
GUEST_MEMORY_START = 0x5000
GUEST_MEMORY_END = 0xEFFF
GUEST_MEMORY_SIZE = GUEST_MEMORY_END - GUEST_MEMORY_START


class PixelBridge:
    """
    API bridge for guest code to interact with the pixel substrate

    Guest Python modules use this to:
    - Read/write pixel memory
    - Call LLMs
    - Load files from PixelFS
    - Sample noise (imperfect computing)
    """

    def __init__(self, hypervisor):
        self.hypervisor = hypervisor
        self.cpu = hypervisor.cpu

    def read_pixel(self, addr: int) -> tuple:
        """Read RGBA pixel at address"""
        if addr >= self.cpu.width * self.cpu.height:
            raise ValueError(f"Address {addr} out of bounds")

        x, y = self.cpu._pc_to_xy(addr)
        return self.cpu.pxi_image.getpixel((x, y))

    def write_pixel(self, addr: int, rgba: tuple):
        """Write RGBA pixel at address"""
        if addr >= self.cpu.width * self.cpu.height:
            raise ValueError(f"Address {addr} out of bounds")

        x, y = self.cpu._pc_to_xy(addr)
        self.cpu.pxi_image.putpixel((x, y), rgba)

    def read_bytes(self, addr: int, length: int) -> bytes:
        """Read bytes from pixel memory (uses G channel)"""
        data = []
        for i in range(length):
            if addr + i >= self.cpu.width * self.cpu.height:
                break
            r, g, b, a = self.read_pixel(addr + i)
            data.append(g)
        return bytes(data)

    def write_bytes(self, addr: int, data: bytes):
        """Write bytes to pixel memory (uses G channel)"""
        for i, byte_val in enumerate(data):
            if addr + i >= self.cpu.width * self.cpu.height:
                break
            self.write_pixel(addr + i, (byte_val, byte_val, byte_val, 255))

    def read_string(self, addr: int, max_len: int = 1024) -> str:
        """Read null-terminated string"""
        data = self.read_bytes(addr, max_len)
        # Find null terminator
        try:
            null_idx = data.index(0)
            data = data[:null_idx]
        except ValueError:
            pass
        return data.decode('utf-8', errors='replace')

    def write_string(self, addr: int, text: str):
        """Write null-terminated string"""
        data = text.encode('utf-8') + b'\x00'
        self.write_bytes(addr, data)

    def call_llm(self, prompt: str, model_id: int = 0, max_tokens: int = 512) -> str:
        """Call LLM via PXDigest"""
        if model_id == 0:
            # Use default LLM
            from pxi_cpu import query_local_llm
            return query_local_llm(prompt)
        else:
            return query_local_llm_via_digest(prompt, model_id, max_tokens)

    def load_file(self, file_id: int) -> bytes:
        """Load file from PixelFS via registry"""
        registry_path = Path("file_boot_registry.json")
        if not registry_path.exists():
            raise FileNotFoundError("file_boot_registry.json not found")

        with open(registry_path, 'r') as f:
            registry = json.load(f)

        entry = registry.get(str(file_id))
        if not entry:
            raise ValueError(f"File ID {file_id} not found in registry")

        # Load from blob
        blob_path = Path(entry['blob_file'])
        import zlib
        compressed = blob_path.read_bytes()
        return zlib.decompress(compressed)

    def sample_noise(self, mean: float = 0.0, std: float = 1.0) -> float:
        """Sample from Gaussian noise (imperfect computing)"""
        return random.gauss(mean, std)

    def sample_uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """Sample from uniform distribution"""
        return random.uniform(low, high)

    def oracle(self, question: str, model_id: int = 0) -> bool:
        """
        Ask LLM a yes/no question (probabilistic decision)

        This is "imperfect computing" at its finest:
        The answer depends on the model's interpretation.
        """
        prompt = f"{question}\n\nAnswer with exactly 'yes' or 'no':"
        response = self.call_llm(prompt, model_id, max_tokens=5).strip().lower()
        return 'yes' in response

    def log(self, message: str):
        """Log message from guest"""
        print(f"[GUEST] {message}")


class PixelHypervisor:
    """
    Hypervisor for executing guest code on the pixel substrate
    """

    def __init__(self, pxi_image: Image.Image, debug: bool = False):
        self.cpu = PXICPU(pxi_image)
        self.debug = debug
        self.running = True
        self.guest_modules = {}

        # Stats
        self.cycles = 0
        self.traps_handled = 0

    def log(self, message: str):
        """Hypervisor log"""
        if self.debug:
            print(f"[HYP] {message}")

    def read_trap_flag(self) -> int:
        """Read trap flag from memory"""
        x, y = self.cpu._pc_to_xy(TRAP_FLAG_ADDR)
        r, g, b, a = self.cpu.pxi_image.getpixel((x, y))
        return r

    def write_trap_flag(self, value: int):
        """Clear trap flag"""
        x, y = self.cpu._pc_to_xy(TRAP_FLAG_ADDR)
        self.cpu.pxi_image.putpixel((x, y), (value, 0, 0, 255))

    def read_trap_arg(self, arg_num: int) -> int:
        """Read 32-bit trap argument"""
        # Simplified: just read from first byte for now
        # In full implementation, would read 4 bytes
        addr = TRAP_ARG0_ADDR + (arg_num * 4)
        x, y = self.cpu._pc_to_xy(addr)
        r, g, b, a = self.cpu.pxi_image.getpixel((x, y))
        return r  # Simplified to 8-bit

    def write_trap_result(self, value: int):
        """Write result back to trap result address"""
        x, y = self.cpu._pc_to_xy(TRAP_RESULT_ADDR)
        self.cpu.pxi_image.putpixel((x, y), (value, 0, 0, 255))

    def handle_trap(self, trap_type: int):
        """Handle a trap/syscall from guest or PXI code"""
        self.traps_handled += 1

        if trap_type == TRAP_RUN_PYTHON:
            self.handle_run_python()

        elif trap_type == TRAP_CALL_LLM:
            self.handle_call_llm()

        elif trap_type == TRAP_LOAD_FILE:
            self.handle_load_file()

        elif trap_type == TRAP_SAMPLE_NOISE:
            self.handle_sample_noise()

        elif trap_type == TRAP_ORACLE:
            self.handle_oracle()

        else:
            self.log(f"Unknown trap type: 0x{trap_type:02X}")
            self.write_trap_result(0xFF)  # Error

    def handle_run_python(self):
        """Execute Python guest module"""
        # ARG0 = file_id or path hash
        file_id = self.read_trap_arg(0)

        self.log(f"TRAP_RUN_PYTHON: file_id=0x{file_id:02X}")

        try:
            # For demo, we'll look for a file in current directory
            # In real impl, would load from PixelFS via file_id

            # Simplified: assume file_id maps to a demo module
            if file_id == 1:
                code = """
# Demo guest module
pxos.log("Hello from guest Python!")
pxos.log(f"Memory region: 0x{pxos.GUEST_MEMORY_START:04X} - 0x{pxos.GUEST_MEMORY_END:04X}")

# Write a message into pixel memory
pxos.write_string(0x5000, "Guest was here!")

# Read it back
msg = pxos.read_string(0x5000)
pxos.log(f"Read back: {msg}")
                """
            else:
                self.log(f"Unknown file_id: {file_id}")
                self.write_trap_result(0xFF)
                return

            # Execute guest code
            self.execute_python_guest(code)
            self.write_trap_result(0)  # Success

        except Exception as e:
            self.log(f"Error executing Python guest: {e}")
            self.write_trap_result(0xFF)

    def handle_call_llm(self):
        """Call LLM on behalf of guest"""
        # ARG0 = prompt address
        # ARG1 = model_id
        # ARG2 = response address

        prompt_addr = self.read_trap_arg(0)
        model_id = self.read_trap_arg(1)

        self.log(f"TRAP_CALL_LLM: prompt_addr=0x{prompt_addr:04X}, model_id={model_id}")

        # Read prompt from pixel memory
        prompt = self.cpu.read_string(prompt_addr)

        # Call LLM
        if model_id > 0:
            response = query_local_llm_via_digest(prompt, model_id, 256)
        else:
            from pxi_cpu import query_local_llm
            response = query_local_llm(prompt)

        # Write response back (simplified - write to fixed response buffer)
        RESPONSE_BUFFER = 0x6000
        self.cpu.write_string(RESPONSE_BUFFER, response, 512)

        self.write_trap_result(0)  # Success

    def handle_load_file(self):
        """Load file from PixelFS"""
        file_id = self.read_trap_arg(0)
        dest_addr = self.read_trap_arg(1)

        self.log(f"TRAP_LOAD_FILE: file_id=0x{file_id:02X}, dest=0x{dest_addr:04X}")

        # Would use SYS_BLOB or load from registry
        # For now, stub
        self.write_trap_result(0)

    def handle_sample_noise(self):
        """Sample noise for imperfect computing"""
        # ARG0 = noise type (0=gaussian, 1=uniform)
        noise_type = self.read_trap_arg(0)

        if noise_type == 0:
            value = int(random.gauss(128, 32))
        else:
            value = random.randint(0, 255)

        # Write to result
        self.write_trap_result(value & 0xFF)

    def handle_oracle(self):
        """Probabilistic LLM decision"""
        # ARG0 = question address
        question_addr = self.read_trap_arg(0)

        question = self.cpu.read_string(question_addr)

        # Ask LLM yes/no
        bridge = PixelBridge(self)
        result = bridge.oracle(question)

        self.write_trap_result(1 if result else 0)

    def execute_python_guest(self, code: str):
        """Execute Python code with PixelBridge API"""
        bridge = PixelBridge(self)

        # Guest namespace
        guest_globals = {
            "pxos": bridge,
            # Expose constants
            "GUEST_MEMORY_START": GUEST_MEMORY_START,
            "GUEST_MEMORY_END": GUEST_MEMORY_END,
        }

        try:
            exec(code, guest_globals, guest_globals)
        except Exception as e:
            self.log(f"Guest Python error: {e}")
            raise

    def run(self, max_cycles: int = 100000, pxi_steps_per_check: int = 100):
        """
        Main hypervisor loop

        Args:
            max_cycles: Maximum cycles to run
            pxi_steps_per_check: How many PXI instructions between trap checks
        """
        self.log("Starting hypervisor...")

        while self.running and self.cycles < max_cycles:
            # Run some PXI instructions
            for _ in range(pxi_steps_per_check):
                if not self.cpu.step():
                    self.log("PXICPU halted")
                    self.running = False
                    break

            self.cycles += pxi_steps_per_check

            # Check for traps
            trap_flag = self.read_trap_flag()
            if trap_flag != 0:
                self.log(f"Trap detected: 0x{trap_flag:02X}")
                self.handle_trap(trap_flag)
                self.write_trap_flag(0)  # Clear trap

        self.log(f"Hypervisor stopped. Cycles: {self.cycles}, Traps: {self.traps_handled}")


def create_demo_image():
    """Create demo PXI image with trap trigger"""
    from pxi_cpu import OP_LOAD, OP_HALT

    img = Image.new("RGBA", (256, 256), (0, 0, 0, 255))

    def emit(pc, opcode, arg1=0, arg2=0, arg3=0):
        x = pc % 256
        y = pc // 256
        img.putpixel((x, y), (opcode, arg1, arg2, arg3))

    pc = 0

    # Simple program: trigger TRAP_RUN_PYTHON
    # Load trap type into R0
    emit(pc, OP_LOAD, 0, TRAP_RUN_PYTHON, 0); pc += 1

    # Write to TRAP_FLAG_ADDR
    # (Simplified - in real code would need proper memory write)

    # Load file_id into trap arg
    emit(pc, OP_LOAD, 1, 1, 0); pc += 1  # file_id = 1

    # Halt
    emit(pc, OP_HALT, 0, 0, 0)

    # Manually set trap flag for demo
    x, y = TRAP_FLAG_ADDR % 256, TRAP_FLAG_ADDR // 256
    img.putpixel((x, y), (TRAP_RUN_PYTHON, 0, 0, 255))

    # Set ARG0 = 1 (file_id)
    x, y = TRAP_ARG0_ADDR % 256, TRAP_ARG0_ADDR // 256
    img.putpixel((x, y), (1, 0, 0, 255))

    return img


def main():
    parser = argparse.ArgumentParser(description="Pixel Hypervisor")
    parser.add_argument('--demo', action='store_true',
                       help='Run demo with guest Python')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')

    args = parser.parse_args()

    if args.demo:
        print("="*60)
        print("Pixel Hypervisor Demo")
        print("="*60)
        print("\nCreating demo PXI image with trap...")

        img = create_demo_image()

        print("Starting hypervisor...")
        hyp = PixelHypervisor(img, debug=args.debug or True)
        hyp.run(max_cycles=1000, pxi_steps_per_check=10)

        print("\n" + "="*60)
        print("Demo complete!")
        print(f"Cycles executed: {hyp.cycles}")
        print(f"Traps handled: {hyp.traps_handled}")
        print("="*60)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
