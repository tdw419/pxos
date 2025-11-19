#!/usr/bin/env python3
"""
pxOS Semantic Builder
=====================

Extended pxOS builder that supports both:
1. Traditional primitives (WRITE, DEFINE, CALL)
2. High-level semantic intents

This demonstrates how the semantic abstraction layer integrates with
the existing pxOS build system.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
from build_pxos import PxOSBuilder
from semantic_layer import SemanticPipeline


# ============================================================================
# x86 INSTRUCTION ENCODING
# ============================================================================

X86_OPCODES = {
    # Control flow
    'cli': [0xFA],                  # Clear interrupt flag
    'sti': [0xFB],                  # Set interrupt flag
    'hlt': [0xF4],                  # Halt
    'nop': [0x90],                  # No operation

    # Stack operations
    'pusha': [0x60],                # Push all general registers
    'popa': [0x61],                 # Pop all general registers
    'pushf': [0x9C],                # Push flags
    'popf': [0x9D],                 # Pop flags

    # Jumps (relative, need offset calculation)
    'jmp_short': [0xEB],            # Short jump (1 byte offset)
    'jmp_near': [0xE9],             # Near jump (2/4 byte offset)

    # Calls (relative, need offset calculation)
    'call_near': [0xE8],            # Near call

    # Returns
    'ret': [0xC3],                  # Near return
    'retf': [0xCB],                 # Far return
    'iret': [0xCF],                 # Interrupt return

    # Register moves (simplified - actual encoding is more complex)
    # These are placeholders - real encoding requires ModR/M bytes
}


def assemble_x86_instruction(instruction: str, address: int, labels: Dict[str, int]) -> List[int]:
    """
    Convert x86 assembly instruction to machine code bytes.

    Args:
        instruction: Assembly instruction string (e.g., "cli", "call foo")
        address: Current address where instruction will be placed
        labels: Symbol table for resolving labels

    Returns:
        List of bytes representing the instruction
    """
    instruction = instruction.strip().lower()

    # Handle comments
    if instruction.startswith('#') or instruction.startswith('@') or instruction.startswith(';'):
        return []  # Skip comments

    # Skip empty lines
    if not instruction:
        return []

    parts = instruction.split()
    if not parts:
        return []

    mnemonic = parts[0]

    # Simple instructions (no operands)
    if mnemonic in X86_OPCODES:
        return X86_OPCODES[mnemonic]

    # Instructions with operands (simplified handling)
    if mnemonic in ['call', 'jmp']:
        # For now, just emit NOP as placeholder
        # Real implementation would need label resolution
        return [0x90]  # NOP placeholder

    # Unknown instruction - emit NOP as placeholder
    print(f"Warning: Unknown instruction '{instruction}', using NOP placeholder")
    return [0x90]


# ============================================================================
# SEMANTIC COMMAND PARSER
# ============================================================================

class SemanticPxOSBuilder(PxOSBuilder):
    """
    Extended pxOS builder with semantic intent support.

    Supports both traditional primitives and high-level semantic intents:
        - WRITE <addr> <value>     # Traditional primitive
        - INTENT critical_section  # High-level semantic intent
    """

    def __init__(self, target_platform: str = 'x86_64'):
        super().__init__()
        self.pipeline = SemanticPipeline(target_platform=target_platform)
        self.target_platform = target_platform
        self.current_address = 0x7C00  # Default: boot sector start

    def parse_line(self, line: str, line_num: int) -> None:
        """Parse both traditional primitives and semantic intents"""
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith('COMMENT') or line.startswith('#'):
            return

        # Remove inline comments
        if 'COMMENT' in line:
            line = line.split('COMMENT')[0].strip()

        parts = line.split()
        if not parts:
            return

        cmd = parts[0].upper()

        # ----------------------------------------------------------------
        # SEMANTIC INTENT COMMANDS
        # ----------------------------------------------------------------
        if cmd == 'INTENT':
            # INTENT <goal> [key=value ...]
            # Example: INTENT critical_section reason="modifying shared data"
            self._process_intent(parts[1:], line_num)

        elif cmd == 'ASM':
            # ASM <instruction>
            # Example: ASM cli
            # Directly assemble x86 instruction
            asm_code = ' '.join(parts[1:])
            self._assemble_instruction(asm_code)

        elif cmd == 'ADDRESS':
            # ADDRESS <addr>
            # Set current assembly address
            self.current_address = self._parse_value(parts[1])

        # ----------------------------------------------------------------
        # TRADITIONAL PRIMITIVE COMMANDS (from parent class)
        # ----------------------------------------------------------------
        else:
            # Delegate to parent class for WRITE, DEFINE, CALL
            super().parse_line(line, line_num)

    def _process_intent(self, args: List[str], line_num: int) -> None:
        """Process a semantic intent command"""
        if not args:
            raise ValueError(f"INTENT requires a goal argument")

        goal = args[0]

        # Parse optional key=value context parameters
        context = {}
        constraints = {}

        for arg in args[1:]:
            if '=' in arg:
                key, value = arg.split('=', 1)
                # Remove quotes if present
                value = value.strip('"').strip("'")

                # Try to parse as number
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # Keep as string

                # Determine if this is context or constraint
                if key.startswith('max_') or key.startswith('must_') or key.endswith('_ms') or key.endswith('_us'):
                    constraints[key] = value
                else:
                    context[key] = value

        # Build intent dictionary
        intent = {
            'goal': goal,
            'context': context,
            'constraints': constraints
        }

        # Process through semantic pipeline
        result = self.pipeline.process(intent)

        print(f"  [Line {line_num}] INTENT {goal}")
        print(f"    Semantic concepts: {len(result['concepts'])}")
        print(f"    Pixels: {result['pixels']}")
        print(f"    Generated code: {len(result['code'])} instructions")

        # Assemble generated code
        for instruction in result['code']:
            self._assemble_instruction(instruction)

    def _assemble_instruction(self, instruction: str) -> None:
        """Assemble a single instruction and write to memory"""
        bytes_list = assemble_x86_instruction(instruction, self.current_address, self.symbols)

        for byte in bytes_list:
            if self.current_address < len(self.memory):
                self.memory[self.current_address] = byte
                self.current_address += 1
                self.operations_count += 1
            else:
                raise ValueError(f"Address {self.current_address:04X} out of bounds")


# ============================================================================
# EXAMPLE SEMANTIC BUILD FILES
# ============================================================================

def create_example_semantic_commands():
    """Create an example semantic build file"""
    example = """
# pxOS Semantic Build Example
# ===========================
# This demonstrates mixing traditional primitives with semantic intents

COMMENT Boot sector setup using SEMANTIC intents
ADDRESS 0x7C00

COMMENT Entry point: disable interrupts for safety
INTENT critical_section reason="boot_initialization"

COMMENT Setup stack using traditional primitives
WRITE 0x7C01 0xB8    COMMENT mov ax, 0x9000
WRITE 0x7C02 0x00
WRITE 0x7C03 0x90
WRITE 0x7C04 0x8E    COMMENT mov ss, ax
WRITE 0x7C05 0xD0
WRITE 0x7C06 0xBC    COMMENT mov sp, 0xFFFF
WRITE 0x7C07 0xFF
WRITE 0x7C08 0xFF

COMMENT Re-enable interrupts after stack setup
ASM sti

COMMENT Memory allocation intent (generates platform-specific code)
INTENT memory_allocation size=4096 purpose=kernel alignment=page_boundary

COMMENT I/O operation intent
INTENT io_operation device=VGA operation=write

COMMENT Boot signature
WRITE 0x7DFE 0x55
WRITE 0x7DFF 0xAA
"""

    with open('pxos_semantic_example.txt', 'w') as f:
        f.write(example)

    print("Created example file: pxos_semantic_example.txt")


# ============================================================================
# MAIN BUILD PROCESS
# ============================================================================

def main():
    """Main build process with semantic support"""

    print("=" * 80)
    print("pxOS SEMANTIC BUILDER")
    print("=" * 80)
    print()

    # Check if we should create an example
    if len(sys.argv) > 1 and sys.argv[1] == '--create-example':
        create_example_semantic_commands()
        return

    # Determine input file
    input_file = Path("pxos_semantic_commands.txt")
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])

    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        print()
        print("Usage:")
        print(f"  {sys.argv[0]} [input_file]")
        print(f"  {sys.argv[0]} --create-example")
        print()
        print("To create an example semantic build file:")
        print(f"  python3 {sys.argv[0]} --create-example")
        sys.exit(1)

    output_bin = Path(input_file.stem + "_semantic.bin")

    # Determine target platform
    target_platform = 'x86_64'
    if len(sys.argv) > 2:
        target_platform = sys.argv[2]

    print(f"Input file: {input_file}")
    print(f"Target platform: {target_platform}")
    print(f"Output binary: {output_bin}")
    print()

    # Build with semantic support
    builder = SemanticPxOSBuilder(target_platform=target_platform)

    print("Building...")
    print()
    builder.build(input_file)

    print()
    builder.write_binary(output_bin)
    builder.print_summary()

    print()
    print("=" * 80)
    print("SEMANTIC LAYER FEATURES USED:")
    print("=" * 80)
    print("✅ High-level intent specification")
    print("✅ Automatic semantic analysis")
    print("✅ Platform-independent pixel encoding")
    print("✅ Platform-specific code generation")
    print("✅ Integration with traditional primitives")
    print()
    print(f"Build complete! Binary: {output_bin}")
    print("=" * 80)


if __name__ == "__main__":
    main()
