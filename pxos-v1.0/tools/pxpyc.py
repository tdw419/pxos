#!/usr/bin/env python3
"""
pxOS Python Compiler (pxpyc) v0.1

Compiles a subset of Python into pxOS primitives (WRITE/DEFINE commands).

This allows the Pixel LLM to write pxOS programs in Python, which are then
cross-compiled into the primitive format that pxOS understands.

Usage:
    python3 pxpyc.py input.py -o output.txt
    python3 pxpyc.py input.py --run  # Compile and build pxOS

Example input (hello.py):
    from pxos import clear_screen, print_text

    def main():
        clear_screen()
        print_text("Hello from Python!", row=10, col=20)

    if __name__ == "__main__":
        main()

Output: pxOS primitive commands that implement the above logic.
"""

import ast
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field


# ============================================================================
# Memory Layout Configuration
# ============================================================================

BOOT_SECTOR_START = 0x7C00
BOOT_SECTOR_END = 0x7DFE  # Last 2 bytes reserved for boot signature
DATA_SECTION_START = 0x7E00  # Right after boot sector
DATA_SECTION_END = 0x9000  # Before stack
CODE_SECTION_START = 0x7C00  # Code starts at boot sector
MAX_CODE_SIZE = 512 - 2  # Boot sector minus signature


# ============================================================================
# x86 Opcode Templates
# ============================================================================

class X86Opcodes:
    """x86 16-bit real mode opcode sequences"""

    # Basic instructions
    CLI = [0xFA]
    STI = [0xFB]
    RET = [0xC3]
    NOP = [0x90]
    HLT = [0xF4]

    # Jump instructions
    JMP_SHORT_REL = [0xEB]  # Followed by signed byte offset
    JMP_SELF = [0xEB, 0xFE]  # JMP $ (infinite loop)

    # INT instructions
    @staticmethod
    def INT(num: int) -> List[int]:
        return [0xCD, num]

    # MOV instructions
    @staticmethod
    def MOV_AX_IMM(value: int) -> List[int]:
        """MOV AX, immediate"""
        return [0xB8, value & 0xFF, (value >> 8) & 0xFF]

    @staticmethod
    def MOV_AH_IMM(value: int) -> List[int]:
        """MOV AH, immediate"""
        return [0xB4, value & 0xFF]

    @staticmethod
    def MOV_AL_IMM(value: int) -> List[int]:
        """MOV AL, immediate"""
        return [0xB0, value & 0xFF]

    @staticmethod
    def MOV_CX_IMM(value: int) -> List[int]:
        """MOV CX, immediate"""
        return [0xB9, value & 0xFF, (value >> 8) & 0xFF]

    @staticmethod
    def MOV_DI_IMM(value: int) -> List[int]:
        """MOV DI, immediate"""
        return [0xBF, value & 0xFF, (value >> 8) & 0xFF]

    @staticmethod
    def MOV_SI_IMM(value: int) -> List[int]:
        """MOV SI, immediate"""
        return [0xBE, value & 0xFF, (value >> 8) & 0xFF]

    # Segment register moves
    @staticmethod
    def MOV_ES_AX() -> List[int]:
        return [0x8E, 0xC0]

    # String operations
    STOSW = [0xAB]  # Store AX at ES:DI, increment DI
    LODSB = [0xAC]  # Load byte at DS:SI to AL, increment SI
    REP = [0xF3]  # Repeat prefix

    # Logical operations
    XOR_DI_DI = [0x31, 0xFF]  # XOR DI, DI (zero DI)
    OR_AL_AL = [0x08, 0xC0]  # OR AL, AL (test if zero)

    # Conditional jumps
    @staticmethod
    def JZ_REL(offset: int) -> List[int]:
        """JZ relative (jump if zero)"""
        return [0x74, offset & 0xFF]

    @staticmethod
    def CALL_REL(offset: int) -> List[int]:
        """CALL relative"""
        # E8 followed by 16-bit signed offset
        return [0xE8, offset & 0xFF, (offset >> 8) & 0xFF]


# ============================================================================
# Compiler State
# ============================================================================

@dataclass
class CompilerContext:
    """Tracks compiler state during code generation"""
    primitives: List[str] = field(default_factory=list)
    symbols: Dict[str, int] = field(default_factory=dict)
    data_address: int = DATA_SECTION_START
    code_address: int = CODE_SECTION_START
    string_counter: int = 0
    label_counter: int = 0
    functions: Dict[str, int] = field(default_factory=dict)  # function name -> address

    def emit_comment(self, text: str) -> None:
        """Emit a comment line"""
        self.primitives.append(f"COMMENT {text}")

    def emit_separator(self) -> None:
        """Emit a visual separator"""
        self.primitives.append("COMMENT " + "=" * 60)

    def emit_write(self, addr: int, value: int, comment: str = "") -> None:
        """Emit a WRITE command"""
        comment_part = f"COMMENT {comment}" if comment else ""
        self.primitives.append(f"WRITE 0x{addr:04X} 0x{value:02X}    {comment_part}".strip())

    def emit_define(self, label: str, addr: int, comment: str = "") -> None:
        """Emit a DEFINE command and track symbol"""
        comment_part = f"COMMENT {comment}" if comment else ""
        self.primitives.append(f"DEFINE {label} 0x{addr:04X}    {comment_part}".strip())
        self.symbols[label] = addr

    def emit_bytes(self, addr: int, bytes_list: List[int], comment: str = "") -> int:
        """Emit a sequence of bytes and return next address"""
        for i, byte in enumerate(bytes_list):
            byte_comment = comment if i == 0 else ""
            self.emit_write(addr + i, byte, byte_comment)
        return addr + len(bytes_list)

    def allocate_data(self, size: int, label: Optional[str] = None) -> int:
        """Allocate space in data section"""
        addr = self.data_address
        if label:
            self.emit_define(label, addr)
        self.data_address += size
        if self.data_address > DATA_SECTION_END:
            raise MemoryError(f"Data section overflow: {self.data_address:04X} > {DATA_SECTION_END:04X}")
        return addr

    def allocate_code(self, size: int, label: Optional[str] = None) -> int:
        """Allocate space in code section"""
        addr = self.code_address
        if label:
            self.emit_define(label, addr)
        self.code_address += size
        if self.code_address > BOOT_SECTOR_END:
            raise MemoryError(f"Boot sector overflow: {self.code_address:04X} > {BOOT_SECTOR_END:04X}")
        return addr

    def get_label(self, prefix: str = "label") -> str:
        """Generate a unique label"""
        label = f"{prefix}_{self.label_counter}"
        self.label_counter += 1
        return label


# ============================================================================
# Code Generator for pxos API Functions
# ============================================================================

class PxOSCodeGen:
    """Generates x86 code for pxos module functions"""

    def __init__(self, ctx: CompilerContext):
        self.ctx = ctx

    def gen_clear_screen(self, color: int = 0x07) -> None:
        """Generate code to clear screen"""
        addr = self.ctx.code_address

        # MOV AX, 0xB800
        addr = self.ctx.emit_bytes(addr, X86Opcodes.MOV_AX_IMM(0xB800), "Set ES to VGA buffer")
        # MOV ES, AX
        addr = self.ctx.emit_bytes(addr, X86Opcodes.MOV_ES_AX())
        # XOR DI, DI
        addr = self.ctx.emit_bytes(addr, X86Opcodes.XOR_DI_DI, "DI = 0 (start of screen)")
        # MOV CX, 2000
        addr = self.ctx.emit_bytes(addr, X86Opcodes.MOV_CX_IMM(2000), "2000 characters")
        # MOV AX, 0x0720 (space + attribute)
        space_char = 0x20 | (color << 8)
        addr = self.ctx.emit_bytes(addr, X86Opcodes.MOV_AX_IMM(space_char), "Space + color attr")
        # REP STOSW
        addr = self.ctx.emit_bytes(addr, [X86Opcodes.REP[0]] + X86Opcodes.STOSW, "Fill screen")

        self.ctx.code_address = addr

    def gen_print_string_func(self) -> int:
        """Generate print_string function, return its address"""
        if "print_string" in self.ctx.functions:
            return self.ctx.functions["print_string"]

        func_addr = self.ctx.code_address
        self.ctx.emit_define("print_string", func_addr, "Print null-terminated string at SI")

        addr = func_addr
        # .loop:
        loop_addr = addr
        # LODSB
        addr = self.ctx.emit_bytes(addr, X86Opcodes.LODSB, "Load byte from SI")
        # OR AL, AL
        addr = self.ctx.emit_bytes(addr, X86Opcodes.OR_AL_AL, "Check if zero")
        # JZ .done (skip 6 bytes ahead)
        addr = self.ctx.emit_bytes(addr, X86Opcodes.JZ_REL(6), "Jump if null terminator")
        # MOV AH, 0x0E
        addr = self.ctx.emit_bytes(addr, X86Opcodes.MOV_AH_IMM(0x0E), "BIOS teletype")
        # INT 0x10
        addr = self.ctx.emit_bytes(addr, X86Opcodes.INT(0x10), "Print character")
        # JMP .loop
        jump_offset = loop_addr - (addr + 2)  # Relative to next instruction
        addr = self.ctx.emit_bytes(addr, X86Opcodes.JMP_SHORT_REL + [jump_offset & 0xFF], "Loop back")
        # .done:
        # RET
        addr = self.ctx.emit_bytes(addr, X86Opcodes.RET, "Return")

        self.ctx.code_address = addr
        self.ctx.functions["print_string"] = func_addr
        return func_addr

    def gen_print_text(self, text: str, row: Optional[int] = None,
                      col: Optional[int] = None, attr: int = 0x07) -> None:
        """Generate code to print text"""
        # First, allocate string in data section
        string_label = f"str_{self.ctx.string_counter}"
        self.ctx.string_counter += 1
        str_addr = self.ctx.allocate_data(len(text) + 1, string_label)

        # Write string bytes
        self.ctx.emit_comment(f"String data: \"{text}\"")
        for i, char in enumerate(text):
            self.ctx.emit_write(str_addr + i, ord(char), f"'{char}'")
        self.ctx.emit_write(str_addr + len(text), 0, "Null terminator")

        if row is not None and col is not None:
            # TODO: Set cursor position first
            # For now, just print at current position
            pass

        # Ensure print_string function exists
        self.gen_print_string_func()

        # Reload address after function generation
        addr = self.ctx.code_address

        # MOV SI, string_label
        addr = self.ctx.emit_bytes(addr, X86Opcodes.MOV_SI_IMM(str_addr), f"Load string address")

        # CALL print_string
        print_str_addr = self.ctx.functions["print_string"]
        call_offset = print_str_addr - (addr + 3)  # Relative to next instruction after CALL
        addr = self.ctx.emit_bytes(addr, X86Opcodes.CALL_REL(call_offset), "Call print_string")

        self.ctx.code_address = addr

    def gen_print_char(self, char: str, row: Optional[int] = None,
                      col: Optional[int] = None, attr: int = 0x07) -> None:
        """Generate code to print a single character"""
        addr = self.ctx.code_address

        # MOV AH, 0x0E
        addr = self.ctx.emit_bytes(addr, X86Opcodes.MOV_AH_IMM(0x0E), "BIOS teletype")
        # MOV AL, char
        addr = self.ctx.emit_bytes(addr, X86Opcodes.MOV_AL_IMM(ord(char)), f"Char: '{char}'")
        # INT 0x10
        addr = self.ctx.emit_bytes(addr, X86Opcodes.INT(0x10), "Print character")

        self.ctx.code_address = addr

    def gen_read_key(self) -> None:
        """Generate code to read a key (result in AL)"""
        addr = self.ctx.code_address

        # MOV AH, 0x00
        addr = self.ctx.emit_bytes(addr, X86Opcodes.MOV_AH_IMM(0x00), "BIOS read key")
        # INT 0x16
        addr = self.ctx.emit_bytes(addr, X86Opcodes.INT(0x16), "Read key -> AL")

        self.ctx.code_address = addr

    def gen_loop_forever(self) -> None:
        """Generate infinite loop"""
        addr = self.ctx.code_address
        addr = self.ctx.emit_bytes(addr, X86Opcodes.JMP_SELF, "Infinite loop")
        self.ctx.code_address = addr


# ============================================================================
# Python AST Visitor
# ============================================================================

class PxOSCompiler(ast.NodeVisitor):
    """Compiles Python AST to pxOS primitives"""

    def __init__(self):
        self.ctx = CompilerContext()
        self.codegen = PxOSCodeGen(self.ctx)
        self.in_main = False

    def compile(self, source_file: Path) -> str:
        """Compile Python source to primitives"""
        with open(source_file) as f:
            source = f.read()

        tree = ast.parse(source, filename=str(source_file))

        self.ctx.emit_separator()
        self.ctx.emit_comment(f"pxOS Python Compiler Output")
        self.ctx.emit_comment(f"Source: {source_file.name}")
        self.ctx.emit_separator()
        self.ctx.emit_comment("")

        # Reserve space for boot sector initialization
        self.ctx.code_address = BOOT_SECTOR_START

        # Visit the AST
        self.visit(tree)

        # Add boot signature
        self.ctx.emit_comment("")
        self.ctx.emit_separator()
        self.ctx.emit_comment("Boot sector signature (added automatically)")
        self.ctx.emit_separator()
        # Note: build_pxos.py adds this automatically, but we document it

        return "\n".join(self.ctx.primitives)

    def visit_Module(self, node: ast.Module) -> None:
        """Visit module level"""
        for stmt in node.body:
            self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition"""
        if node.name == "main":
            self.in_main = True
            self.ctx.emit_comment(f"Function: {node.name}")

            # Process main function body
            for stmt in node.body:
                self.visit(stmt)

            self.in_main = False

    def visit_If(self, node: ast.If) -> None:
        """Visit if statement (only handle if __name__ == "__main__")"""
        # Check if this is: if __name__ == "__main__":
        if (isinstance(node.test, ast.Compare) and
            isinstance(node.test.left, ast.Name) and
            node.test.left.id == "__name__"):

            # Execute the body (typically calls main())
            for stmt in node.body:
                self.visit(stmt)

    def visit_Expr(self, node: ast.Expr) -> None:
        """Visit expression statement"""
        self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call"""
        # Handle pxos module functions
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "clear_screen":
                self._handle_clear_screen(node)
            elif node.func.attr == "print_text":
                self._handle_print_text(node)
            elif node.func.attr == "print_char":
                self._handle_print_char(node)
            elif node.func.attr == "loop_forever":
                self._handle_loop_forever(node)
        elif isinstance(node.func, ast.Name):
            # Direct function call (from import)
            func_name = node.func.id
            if func_name == "clear_screen":
                self._handle_clear_screen(node)
            elif func_name == "print_text":
                self._handle_print_text(node)
            elif func_name == "print_char":
                self._handle_print_char(node)
            elif func_name == "loop_forever":
                self._handle_loop_forever(node)
            elif func_name == "main":
                # Calling main() - just continue, we handle it in FunctionDef
                pass

    def _handle_clear_screen(self, node: ast.Call) -> None:
        """Handle clear_screen() call"""
        color = 0x07
        if node.args:
            if isinstance(node.args[0], ast.Constant):
                color = node.args[0].value

        self.ctx.emit_comment("clear_screen()")
        self.codegen.gen_clear_screen(color)

    def _handle_print_text(self, node: ast.Call) -> None:
        """Handle print_text() call"""
        if not node.args:
            return

        text = None
        row = None
        col = None
        attr = 0x07

        # Get positional argument (text)
        if isinstance(node.args[0], ast.Constant):
            text = node.args[0].value

        # Get keyword arguments
        for keyword in node.keywords:
            if keyword.arg == "row" and isinstance(keyword.value, ast.Constant):
                row = keyword.value.value
            elif keyword.arg == "col" and isinstance(keyword.value, ast.Constant):
                col = keyword.value.value
            elif keyword.arg == "attr" and isinstance(keyword.value, ast.Constant):
                attr = keyword.value.value

        if text:
            self.ctx.emit_comment(f'print_text("{text}")')
            self.codegen.gen_print_text(text, row, col, attr)

    def _handle_print_char(self, node: ast.Call) -> None:
        """Handle print_char() call"""
        if not node.args:
            return

        if isinstance(node.args[0], ast.Constant):
            char = node.args[0].value
            self.ctx.emit_comment(f"print_char('{char}')")
            self.codegen.gen_print_char(char)

    def _handle_loop_forever(self, node: ast.Call) -> None:
        """Handle loop_forever() call"""
        self.ctx.emit_comment("loop_forever()")
        self.codegen.gen_loop_forever()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="pxOS Python Compiler - Compile Python to pxOS primitives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compile Python to primitives
  pxpyc.py hello.py -o hello_primitives.txt

  # Compile and show output
  pxpyc.py hello.py

  # Compile and build pxOS image
  pxpyc.py hello.py --build
        """
    )

    parser.add_argument("input", type=Path, help="Input Python file")
    parser.add_argument("-o", "--output", type=Path, help="Output primitives file")
    parser.add_argument("--build", action="store_true", help="Build pxOS binary after compiling")
    parser.add_argument("--run", action="store_true", help="Compile, build, and run in QEMU")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    # Compile
    try:
        compiler = PxOSCompiler()
        primitives = compiler.compile(args.input)

        # Output
        if args.output:
            args.output.write_text(primitives)
            print(f"Compiled: {args.input} -> {args.output}")
        else:
            print(primitives)

        # Build if requested
        if args.build or args.run:
            output_file = args.output or Path("pxos_python_output.txt")
            if not args.output:
                output_file.write_text(primitives)

            # Call build_pxos.py
            import subprocess
            build_script = Path(__file__).parent.parent / "build_pxos.py"
            subprocess.run([sys.executable, str(build_script)], check=True)

        # Run if requested
        if args.run:
            import subprocess
            print("\nLaunching QEMU...")
            subprocess.run(["qemu-system-i386", "-fda", "pxos.bin"])

    except SyntaxError as e:
        print(f"Python syntax error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Compilation error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
