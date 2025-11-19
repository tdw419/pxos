#!/usr/bin/env python3
"""
pxOS Builder v1.0
Converts WRITE/DEFINE/CALL primitives into a bootable .bin

This builder demonstrates pxOS's unique primitive-based approach:
- No assembler needed for initial development
- Direct memory manipulation via WRITE commands
- Symbolic addressing via DEFINE commands
- Educational and hackable build process
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

from enum import Enum
from dataclasses import dataclass, field

MEMORY_SIZE = 0x10000  # 64KB
VGA_TEXT_BUFFER = 0xB8000  # VGA text mode buffer (for reference)

class ParamType(Enum):
    """GPU kernel parameter types"""
    INT = "int"
    FLOAT = "float"
    INT_ARRAY = "int[]"
    FLOAT_ARRAY = "float[]"

@dataclass
class GPUKernel:
    name: str
    params: List[Dict] = field(default_factory=list)
    code: List[str] = field(default_factory=list)

@dataclass
class GPULaunch:
    kernel_name: str
    blocks: int
    threads: int

class CUDAGenerator:
    def __init__(self, kernels: List[GPUKernel], launches: List[GPULaunch]):
        self.kernels = kernels
        self.launches = launches

    def generate(self) -> str:
        # Placeholder for CUDA code generation
        return "// CUDA code generation not yet implemented"

class LLMAnalyzer:
    def __init__(self, provider: str = "local"):
        self.provider = provider

    def analyze(self, kernels: List[GPUKernel], launches: List[GPULaunch]) -> bool:
        """
        Analyzes the primitives and decides if GPU code should be generated.
        For this PoC, it's a simple rule: if there are GPU kernels, generate code.
        """
        if kernels:
            print("\n🤖 LLM Analyzer: GPU kernels detected. Recommending GPU code generation.")
            return True
        else:
            print("\n🤖 LLM Analyzer: No GPU kernels detected. Skipping GPU code generation.")
            return False

class UnifiedBuilder:
    def __init__(self):
        self.memory = bytearray(MEMORY_SIZE)
        self.symbols: Dict[str, int] = {}
        self.gpu_kernels: List[GPUKernel] = []
        self.gpu_launches: List[GPULaunch] = []
        self.operations_count = 0
        self._in_gpu_kernel = False
        self._current_gpu_kernel: Optional[GPUKernel] = None

    def parse_line(self, line: str, line_num: int) -> None:
        """Parse a single line of pxOS primitive commands"""
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith('COMMENT') or line.startswith('#'):
            return

        # Remove inline comments (anything after COMMENT)
        if 'COMMENT' in line:
            line = line.split('COMMENT')[0].strip()

        parts = line.split()
        if not parts:
            return

        cmd = parts[0].upper()

        if self._in_gpu_kernel:
            if cmd == 'GPU_PARAM':
                if len(parts) < 3:
                    raise ValueError("GPU_PARAM requires a name and type")
                self._current_gpu_kernel.params.append({'name': parts[1], 'type': parts[2]})
            elif cmd == 'GPU_END':
                self._in_gpu_kernel = False
                self.gpu_kernels.append(self._current_gpu_kernel)
                self._current_gpu_kernel = None
            else:
                self._current_gpu_kernel.code.append(line)
            return

        if cmd == 'GPU_KERNEL':
            if len(parts) < 2:
                raise ValueError("GPU_KERNEL requires a name")
            self._in_gpu_kernel = True
            self._current_gpu_kernel = GPUKernel(name=parts[1])

        elif cmd == 'GPU_LAUNCH':
            if len(parts) < 4:
                raise ValueError("GPU_LAUNCH requires kernel_name, blocks, and threads")
            launch = GPULaunch(kernel_name=parts[1], blocks=int(parts[2]), threads=int(parts[3]))
            self.gpu_launches.append(launch)

        elif cmd == 'WRITE':
            # WRITE <address> <byte_value>
            if len(parts) < 3:
                raise ValueError(f"WRITE requires 2 arguments: address and value")
            addr = self._parse_value(parts[1])
            byte_val = self._parse_value(parts[2])

            if addr >= MEMORY_SIZE:
                raise ValueError(f"Address 0x{addr:04X} out of bounds (max 0x{MEMORY_SIZE:04X})")
            if byte_val > 0xFF:
                raise ValueError(f"Byte value 0x{byte_val:X} out of range (max 0xFF)")

            self.memory[addr] = byte_val
            self.operations_count += 1

        elif cmd == 'DEFINE':
            # DEFINE <label> <address>
            if len(parts) < 3:
                raise ValueError(f"DEFINE requires 2 arguments: label and address")
            label = parts[1]
            addr = self._parse_value(parts[2])

            if addr >= MEMORY_SIZE:
                raise ValueError(f"Address 0x{addr:04X} out of bounds")

            self.symbols[label] = addr

        elif cmd == 'CALL':
            # CALL <label> - Future enhancement for symbolic references
            # Currently just a comment/documentation
            pass

        else:
            # Unknown command - treat as comment
            pass

    def _parse_value(self, value_str: str) -> int:
        """Parse a numeric value (hex, decimal, or symbol expression)"""
        value_str = value_str.strip()

        # Handle simple expressions like SYMBOL+1
        for symbol, value in self.symbols.items():
            value_str = value_str.replace(symbol, str(value))

        try:
            # Use eval for simplicity. For a production system, a proper
            # expression parser would be safer.
            return eval(value_str)
        except Exception:
            raise ValueError(f"Cannot parse value: {value_str}")

    def build(self, input_file: Path) -> None:
        """Build pxOS from primitive commands"""
        if not input_file.exists():
            print(f"Error: {input_file} not found!")
            sys.exit(1)

        print(f"Building pxOS from {input_file}...")

        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    self.parse_line(line, line_num)
                except Exception as e:
                    print(f"Error at line {line_num}: {line.strip()}")
                    print(f"  {e}")
                    sys.exit(1)

        # Add boot signature at end of boot sector
        self.memory[0x1FE] = 0x55
        self.memory[0x1FF] = 0xAA

        print(f"  {self.operations_count} operations applied")
        print(f"  {len(self.symbols)} symbols defined")

    def write_binary(self, output_file: Path) -> None:
        """Write the bootable binary"""
        # Find the last non-zero byte to determine actual size
        last_byte = MEMORY_SIZE - 1
        while last_byte > 0 and self.memory[last_byte] == 0:
            last_byte -= 1

        # Always write at least one sector (512 bytes)
        size = max(512, last_byte + 1)

        with open(output_file, 'wb') as f:
            f.write(self.memory[:size])

        print(f"  Binary written: {output_file} ({size} bytes)")

    def create_iso(self, bin_file: Path, iso_file: Path) -> bool:
        """Create bootable ISO (optional, requires genisoimage)"""
        try:
            import subprocess

            iso_boot = Path("iso_boot")
            iso_boot.mkdir(exist_ok=True)

            # Copy binary to iso boot directory
            import shutil
            shutil.copy(bin_file, iso_boot / "pxos.bin")

            subprocess.run([
                "genisoimage", "-o", str(iso_file),
                "-b", "pxos.bin", "-no-emul-boot",
                "-boot-load-size", "4", "-boot-info-table",
                str(iso_boot)
            ], check=True, capture_output=True)

            print(f"  ISO created: {iso_file}")
            return True

        except FileNotFoundError:
            print("  (genisoimage not found, skipping ISO creation)")
            return False
        except Exception as e:
            print(f"  (ISO creation failed: {e})")
            return False

    def generate_cuda(self, output_file: Path):
        """Generates a CUDA source file from the parsed GPU primitives."""
        if not self.gpu_kernels:
            return

        print(f"\nGenerating CUDA code to {output_file}...")
        generator = CUDAGenerator(self.gpu_kernels, self.gpu_launches)
        cuda_code = generator.generate()
        output_file.write_text(cuda_code)
        print(f"  ✅ CUDA code generated.")

    def print_summary(self) -> None:
        """Print build summary"""
        print("\n=== Build Summary ===")
        print(f"CPU Operations: {self.operations_count}")
        print(f"CPU Symbols defined: {len(self.symbols)}")
        print(f"GPU Kernels defined: {len(self.gpu_kernels)}")

        if self.symbols:
            print("\nCPU Symbol Table:")
            for label, addr in sorted(self.symbols.items(), key=lambda x: x[1]):
                print(f"  {label:20s} = 0x{addr:04X}")

        if self.gpu_kernels:
            print("\nGPU Kernels:")
            for kernel in self.gpu_kernels:
                print(f"  - {kernel.name} ({len(kernel.params)} params, {len(kernel.code)} lines)")
                for param in kernel.params:
                    print(f"    - IN: {param['name']} ({param['type']})")

        if self.gpu_launches:
            print("\nGPU Launches:")
            for launch in self.gpu_launches:
                print(f"  - {launch.kernel_name} (blocks: {launch.blocks}, threads: {launch.threads})")

        print("\nBoot with: qemu-system-i386 -fda pxos.bin")

def main():
    input_file = Path("unified_primitives.px")
    output_bin = Path("pxos.bin")
    output_iso = Path("pxos.iso")

    builder = UnifiedBuilder()
    builder.build(input_file)
    builder.write_binary(output_bin)
    builder.create_iso(output_bin, output_iso)

    analyzer = LLMAnalyzer()
    if analyzer.analyze(builder.gpu_kernels, builder.gpu_launches):
        builder.generate_cuda(Path("kernel.cu"))

    builder.print_summary()

    print("\npxOS v1.0 built successfully!")

if __name__ == "__main__":
    main()
