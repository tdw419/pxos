#!/usr/bin/env python3
"""
AI-Powered Primitive Generator for pxOS

This tool uses LM Studio to automatically generate WRITE/DEFINE/CALL primitives
for pxOS features. It bridges natural language descriptions to executable
bootloader code.

The system:
1. Takes a feature description (e.g., "add backspace support")
2. Queries LM Studio with pxOS context
3. Extracts primitive commands from LLM response
4. Validates and formats them
5. Appends to pxos_commands.txt
6. Learns from successful builds
"""

from pathlib import Path
import sys
import re
import requests
from typing import List, Dict, Optional, Tuple

# Add pxOS root to path
pxos_root = Path(__file__).resolve().parents[1]
sys.path.append(str(pxos_root))

from pxvm.integration.lm_studio_bridge import LMStudioPixelBridge


class PrimitiveGenerator:
    """
    Generates pxOS primitive commands using LM Studio.

    This is the AI brain that writes assembly code as primitives!
    """

    def __init__(
        self,
        network_path: Path,
        lm_studio_url: str = "http://localhost:1234/v1",
        knowledge_base: Optional[str] = None
    ):
        self.bridge = LMStudioPixelBridge(network_path, lm_studio_url)
        self.knowledge_base = knowledge_base or self._load_default_knowledge()

        # Seed the network with pxOS-specific knowledge
        if knowledge_base:
            self._seed_knowledge()

    def _load_default_knowledge(self) -> str:
        """Load default pxOS knowledge base."""
        return """
=== pxOS PRIMITIVE SYSTEM KNOWLEDGE BASE ===

## PRIMITIVE COMMANDS

WRITE <addr> <byte>  - Write a byte to memory
DEFINE <label> <addr> - Create a symbolic label
CALL <label>         - Documentation only (not executed)
COMMENT <text>       - Comment line

## MEMORY MAP

0x7C00-0x7DFF : Boot sector (512 bytes)
0x7E00-0x7FFF : Extended boot code
0x0050        : Cursor position storage
0xB800:0000   : VGA text mode buffer (80x25)

## COMMON x86 OPCODES

# Data movement
0xB8 + reg    : MOV reg16, imm16
0xB0 + reg    : MOV reg8, imm8
0x8E          : MOV segment, reg
0xA0/A1       : MOV AL/AX, [addr]

# Stack
0x50-0x57     : PUSH reg16
0x58-0x5F     : POP reg16
0xFA          : CLI (disable interrupts)
0xFB          : STI (enable interrupts)

# Arithmetic/Logic
0x31          : XOR reg, reg
0xB9          : MOV CX, imm16
0xF3 0xAB     : REP STOSW

# Control flow
0xCD          : INT (software interrupt)
0xE8          : CALL rel16
0xC3          : RET
0xEB          : JMP short
0x74          : JZ (jump if zero)
0x75          : JNZ (jump if not zero)

## BIOS INTERRUPTS

INT 0x10 - Video Services
  AH=0x00 : Set video mode
  AH=0x0E : Teletype output (AL=char)
  AH=0x02 : Set cursor position

INT 0x16 - Keyboard Services
  AH=0x00 : Wait for keypress (returns AL=ASCII)
  AH=0x01 : Check for keypress

INT 0x13 - Disk Services
  AH=0x02 : Read sectors
  AH=0x03 : Write sectors

## BOOT SECTOR REQUIREMENTS

- First instruction at 0x7C00
- Boot signature 0x55AA at bytes 510-511 (0x1FE-0x1FF)
- Total size: exactly 512 bytes for sector 1

## EXAMPLE PRIMITIVE SEQUENCES

Print character 'A':
WRITE 0x7C00 0xB4    COMMENT mov ah, 0x0E
WRITE 0x7C01 0x0E
WRITE 0x7C02 0xB0    COMMENT mov al, 'A'
WRITE 0x7C03 0x41
WRITE 0x7C04 0xCD    COMMENT int 0x10
WRITE 0x7C05 0x10

Wait for keypress:
WRITE 0x7C10 0xB4    COMMENT mov ah, 0x00
WRITE 0x7C11 0x00
WRITE 0x7C12 0xCD    COMMENT int 0x16
WRITE 0x7C13 0x16

Set up stack:
WRITE 0x7C00 0xFA    COMMENT cli
WRITE 0x7C01 0xB8    COMMENT mov ax, 0x9000
WRITE 0x7C02 0x00
WRITE 0x7C03 0x90
WRITE 0x7C04 0x8E    COMMENT mov ss, ax
WRITE 0x7C05 0xD0
WRITE 0x7C06 0xBC    COMMENT mov sp, 0xFFFF
WRITE 0x7C07 0xFF
WRITE 0x7C08 0xFF
WRITE 0x7C09 0xFB    COMMENT sti
"""

    def _seed_knowledge(self):
        """Seed the pixel network with pxOS knowledge."""
        print("🌱 Seeding pixel network with pxOS knowledge...")
        self.bridge.append_interaction(
            "What is the pxOS primitive system?",
            self.knowledge_base
        )

    def generate_primitives(
        self,
        feature_description: str,
        start_address: Optional[int] = None,
        max_attempts: int = 3
    ) -> Tuple[bool, List[str], str]:
        """
        Generate primitive commands for a feature.

        Returns:
            (success, primitive_lines, explanation)
        """
        print(f"\n🤖 Generating primitives for: {feature_description}")

        # Build detailed prompt
        prompt = self._build_generation_prompt(feature_description, start_address)

        # Query LM Studio
        response = self.bridge.ask_lm_studio(prompt, use_context=True)

        # Extract primitives from response
        primitives = self._extract_primitives(response)

        if not primitives:
            print("⚠️  No valid primitives found in LLM response")
            return False, [], response

        # Validate primitives
        valid, errors = self._validate_primitives(primitives)

        if not valid:
            print(f"❌ Validation errors: {errors}")
            return False, primitives, response

        print(f"✅ Generated {len(primitives)} primitive commands")

        # Learn from successful generation
        self.bridge.append_interaction(feature_description, response)

        return True, primitives, response

    def _build_generation_prompt(
        self,
        feature: str,
        start_addr: Optional[int]
    ) -> str:
        """Build a detailed prompt for primitive generation."""

        addr_hint = f"Start at address 0x{start_addr:04X}." if start_addr else ""

        prompt = f"""Generate pxOS primitive commands (WRITE/DEFINE/COMMENT) for this feature:

FEATURE: {feature}

REQUIREMENTS:
1. Use only WRITE, DEFINE, COMMENT, and CALL commands
2. Include COMMENT lines explaining each instruction
3. Use proper x86 opcodes and BIOS interrupts
4. Follow pxOS memory map conventions
5. {addr_hint}
6. Output ONLY the primitive commands, no extra explanation

EXAMPLE FORMAT:
COMMENT Add backspace support
DEFINE backspace_handler 0x7C50
WRITE 0x7C50 0xB4    COMMENT mov ah, 0x0E
WRITE 0x7C51 0x0E
WRITE 0x7C52 0xB0    COMMENT mov al, 0x08 (backspace)
WRITE 0x7C53 0x08
...

Now generate the primitives:
"""
        return prompt

    def _extract_primitives(self, llm_response: str) -> List[str]:
        """Extract primitive command lines from LLM response."""
        primitives = []

        lines = llm_response.split('\n')
        for line in lines:
            line = line.strip()

            # Look for lines starting with primitive commands
            if line.startswith(('WRITE ', 'DEFINE ', 'COMMENT ', 'CALL ')):
                primitives.append(line)

            # Also check for lines that might have markdown code formatting
            elif line.startswith('WRITE ') or line.startswith('DEFINE '):
                primitives.append(line)

        return primitives

    def _validate_primitives(self, primitives: List[str]) -> Tuple[bool, List[str]]:
        """Validate primitive commands."""
        errors = []

        for i, line in enumerate(primitives):
            parts = line.split()
            if not parts:
                continue

            cmd = parts[0].upper()

            if cmd == 'WRITE':
                if len(parts) < 3:
                    errors.append(f"Line {i+1}: WRITE needs address and value")
                    continue

                # Validate address
                try:
                    addr_str = parts[1]
                    if addr_str.startswith('0x'):
                        addr = int(addr_str, 16)
                    else:
                        addr = int(addr_str, 0)

                    if addr >= 0x10000:
                        errors.append(f"Line {i+1}: Address 0x{addr:X} out of bounds")
                except ValueError:
                    errors.append(f"Line {i+1}: Invalid address format")

                # Validate byte value
                try:
                    val_str = parts[2]
                    if val_str.startswith('0x'):
                        val = int(val_str, 16)
                    else:
                        val = int(val_str, 0)

                    if val > 0xFF:
                        errors.append(f"Line {i+1}: Byte value 0x{val:X} > 0xFF")
                except ValueError:
                    errors.append(f"Line {i+1}: Invalid byte value")

            elif cmd == 'DEFINE':
                if len(parts) < 3:
                    errors.append(f"Line {i+1}: DEFINE needs label and address")

        return len(errors) == 0, errors

    def append_to_commands_file(
        self,
        primitives: List[str],
        commands_file: Path,
        section_name: str
    ):
        """Append generated primitives to pxos_commands.txt."""

        with open(commands_file, 'a') as f:
            f.write(f"\n\nCOMMENT {'='*70}\n")
            f.write(f"COMMENT AI-GENERATED: {section_name}\n")
            f.write(f"COMMENT {'='*70}\n")

            for line in primitives:
                f.write(line + '\n')

        print(f"✅ Appended {len(primitives)} lines to {commands_file}")


def main():
    """Interactive AI primitive generator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AI-Powered pxOS Primitive Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate primitives for a feature
  python3 tools/ai_primitive_generator.py --feature "Add backspace support"

  # Interactive mode
  python3 tools/ai_primitive_generator.py --interactive

  # Specify starting address
  python3 tools/ai_primitive_generator.py --feature "Help command" --addr 0x7E00
"""
    )

    parser.add_argument(
        "--network",
        default="pxvm/networks/pxos_dev.png",
        help="Pixel network for AI learning"
    )
    parser.add_argument(
        "--feature",
        help="Feature description to generate"
    )
    parser.add_argument(
        "--addr",
        help="Starting address (hex, e.g., 0x7E00)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive generation mode"
    )
    parser.add_argument(
        "--output",
        default="pxos-v1.0/pxos_commands.txt",
        help="Commands file to append to"
    )

    args = parser.parse_args()

    # Parse start address if provided
    start_addr = None
    if args.addr:
        start_addr = int(args.addr, 16) if args.addr.startswith('0x') else int(args.addr, 0)

    # Initialize generator
    print("🚀 Initializing AI Primitive Generator...")
    generator = PrimitiveGenerator(
        network_path=Path(args.network),
        knowledge_base=generator._load_default_knowledge() if hasattr(PrimitiveGenerator, '_load_default_knowledge') else None
    )

    if args.interactive:
        # Interactive loop
        print("\n" + "="*70)
        print("🎨 INTERACTIVE PRIMITIVE GENERATION")
        print("="*70)
        print("Describe features and I'll generate the primitives!")
        print("Type 'exit' to quit.\n")

        while True:
            try:
                feature = input("\n💡 Feature to implement: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nExiting...")
                break

            if not feature or feature.lower() == 'exit':
                break

            success, primitives, explanation = generator.generate_primitives(
                feature,
                start_addr
            )

            if success:
                print("\n📝 Generated Primitives:")
                print("-" * 70)
                for line in primitives:
                    print(line)
                print("-" * 70)

                save = input("\n💾 Append to commands file? (y/n): ").strip().lower()
                if save == 'y':
                    generator.append_to_commands_file(
                        primitives,
                        Path(args.output),
                        feature
                    )
            else:
                print("\n❌ Generation failed")
                print(explanation[:500])

    elif args.feature:
        # Single feature generation
        success, primitives, explanation = generator.generate_primitives(
            args.feature,
            start_addr
        )

        if success:
            print("\n📝 Generated Primitives:")
            for line in primitives:
                print(line)

            generator.append_to_commands_file(
                primitives,
                Path(args.output),
                args.feature
            )
        else:
            print("\n❌ Generation failed")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
