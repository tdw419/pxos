#!/usr/bin/env python3
"""
pxOS Python Compiler v2.0 (IR-based)

Compiles Python to PXI Assembly IR, which is then compiled to primitives.

New Architecture:
    Python → PXI Assembly IR (JSON) → ir_compiler.py → Primitives → Binary

This version is MUCH simpler than v1 because:
- No opcode knowledge needed
- No address calculation needed
- Just emits semantic PXI Assembly instructions
- Debuggable: can inspect IR between steps

Usage:
    pxpyc2.py input.py -o output.pxi.json
    pxpyc2.py input.py --build  # Compile + build + run
"""

import ast
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict


# ============================================================================
# IR Data Structures
# ============================================================================

@dataclass
class IRInstruction:
    """Single PXI Assembly instruction"""
    op: str
    operands: Dict[str, Any] = field(default_factory=dict)
    comment: str = ""

    def to_dict(self):
        """Convert to JSON-serializable dict"""
        return {
            "op": self.op,
            "operands": self.operands,
            "comment": self.comment
        }


@dataclass
class IRData:
    """Data definition"""
    label: str
    type: str  # 'string', 'byte', 'word'
    value: Any

    def to_dict(self):
        return {
            "label": self.label,
            "type": self.type,
            "value": self.value
        }


@dataclass
class IRProgram:
    """Complete IR program"""
    source_file: str
    origin: int = 0x7C00
    instructions: List[IRInstruction] = field(default_factory=list)
    data: List[IRData] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps({
            "source": self.source_file,
            "ir_version": "1.0",
            "origin": self.origin,
            "instructions": [instr.to_dict() for instr in self.instructions],
            "data": [d.to_dict() for d in self.data]
        }, indent=2)


# ============================================================================
# IR Code Generator (Python → PXI Assembly)
# ============================================================================

class IRGenerator:
    """Generate PXI Assembly IR from Python code"""

    def __init__(self, source_file: str):
        self.program = IRProgram(source_file=source_file)
        self.string_counter = 0
        self.label_counter = 0

    def emit_instr(self, op: str, **operands) -> None:
        """Emit an IR instruction"""
        comment = operands.pop('comment', '')
        instr = IRInstruction(op=op, operands=operands, comment=comment)
        self.program.instructions.append(instr)

    def emit_data(self, label: str, data_type: str, value: Any) -> None:
        """Emit a data definition"""
        data = IRData(label=label, type=data_type, value=value)
        self.program.data.append(data)

    def get_string_label(self) -> str:
        """Generate unique string label"""
        label = f"str_{self.string_counter}"
        self.string_counter += 1
        return label

    # ========================================================================
    # High-level pxos API → PXI Assembly
    # ========================================================================

    def gen_clear_screen(self, color: int = 0x07) -> None:
        """Generate IR for clear_screen()"""
        self.emit_instr("COMMENT", comment="clear_screen()")

        # Set ES to VGA buffer (0xB800)
        self.emit_instr("MOV", dst="AX", src=0xB800, comment="Set ES to VGA buffer")
        self.emit_instr("MOV", dst="ES", src="AX")

        # Zero DI (start of screen)
        self.emit_instr("XOR", dst="DI", src="DI", comment="DI = 0 (start of screen)")

        # Set CX = 2000 (80x25 characters)
        self.emit_instr("MOV", dst="CX", src=2000, comment="2000 characters")

        # Set AX = space character with color attribute
        space_val = 0x20 | (color << 8)
        self.emit_instr("MOV", dst="AX", src=space_val, comment="Space + color attr")

        # Fill screen: REP STOSW
        self.emit_instr("REP", comment="Fill screen")
        self.emit_instr("STOSW")

    def gen_print_text(self, text: str, row: Optional[int] = None,
                      col: Optional[int] = None, attr: int = 0x07) -> None:
        """Generate IR for print_text()"""
        self.emit_instr("COMMENT", comment=f'print_text("{text}")')

        # Define string data
        str_label = self.get_string_label()
        self.emit_data(str_label, "string", text)

        # TODO: If row/col specified, call move_cursor first
        if row is not None and col is not None:
            # For now, ignore positioning
            pass

        # Load string address and call print_string
        self.emit_instr("MOV", dst="SI", src=str_label, comment="Load string address")
        self.emit_instr("CALL", target="print_string", comment="Call print_string")

    def gen_print_char(self, char: str, row: Optional[int] = None,
                      col: Optional[int] = None, attr: int = 0x07) -> None:
        """Generate IR for print_char()"""
        self.emit_instr("COMMENT", comment=f"print_char('{char}')")

        # BIOS teletype: AH=0x0E, AL=char, INT 0x10
        self.emit_instr("MOV", dst="AH", src=0x0E, comment="BIOS teletype")
        self.emit_instr("MOV", dst="AL", src=ord(char), comment=f"Char: '{char}'")
        self.emit_instr("INT", num=0x10, comment="Print character")

    def gen_loop_forever(self) -> None:
        """Generate IR for loop_forever()"""
        self.emit_instr("COMMENT", comment="loop_forever()")
        self.emit_instr("JMP", target="$", comment="Infinite loop")

    def gen_print_string_function(self) -> None:
        """Generate the print_string library function"""
        self.emit_instr("LABEL", name="print_string",
                       comment="Print null-terminated string at SI")

        # .loop:
        loop_label = ".print_loop"
        self.emit_instr("LABEL", name=loop_label)

        # Load byte from string
        self.emit_instr("LODSB", comment="Load byte from SI")

        # Check if null terminator
        self.emit_instr("OR", dst="AL", src="AL", comment="Check if zero")
        self.emit_instr("JZ", target=".print_done", comment="Jump if null terminator")

        # Print character via BIOS
        self.emit_instr("MOV", dst="AH", src=0x0E, comment="BIOS teletype")
        self.emit_instr("INT", num=0x10, comment="Print character")

        # Loop back
        self.emit_instr("JMP", target=loop_label, comment="Loop back")

        # .done:
        self.emit_instr("LABEL", name=".print_done")
        self.emit_instr("RET", comment="Return")


# ============================================================================
# Python AST Visitor
# ============================================================================

class PythonToIR(ast.NodeVisitor):
    """Compile Python AST to PXI Assembly IR"""

    def __init__(self, source_file: Path):
        self.source_file = source_file
        self.ir_gen = IRGenerator(source_file.name)
        self.in_main = False
        self.need_print_string = False

    def compile(self) -> IRProgram:
        """Compile Python source to IR"""
        with open(self.source_file) as f:
            source = f.read()

        tree = ast.parse(source, filename=str(self.source_file))

        # Visit AST
        self.visit(tree)

        # Add print_string function if needed
        if self.need_print_string:
            self.ir_gen.gen_print_string_function()

        return self.ir_gen.program

    def visit_Module(self, node: ast.Module) -> None:
        for stmt in node.body:
            self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == "main":
            self.in_main = True

            # Process main function body
            for stmt in node.body:
                self.visit(stmt)

            self.in_main = False

    def visit_If(self, node: ast.If) -> None:
        # Handle: if __name__ == "__main__":
        if (isinstance(node.test, ast.Compare) and
            isinstance(node.test.left, ast.Name) and
            node.test.left.id == "__name__"):

            for stmt in node.body:
                self.visit(stmt)

    def visit_Expr(self, node: ast.Expr) -> None:
        self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        # Get function name
        func_name = None
        if isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id

        # Handle pxos API calls
        if func_name == "clear_screen":
            self._handle_clear_screen(node)
        elif func_name == "print_text":
            self._handle_print_text(node)
        elif func_name == "print_char":
            self._handle_print_char(node)
        elif func_name == "loop_forever":
            self._handle_loop_forever(node)
        elif func_name == "main":
            # Calling main() - just continue
            pass

    def _handle_clear_screen(self, node: ast.Call) -> None:
        color = 0x07
        if node.args and isinstance(node.args[0], ast.Constant):
            color = node.args[0].value

        self.ir_gen.gen_clear_screen(color)

    def _handle_print_text(self, node: ast.Call) -> None:
        if not node.args:
            return

        text = None
        row = None
        col = None
        attr = 0x07

        # Get positional argument
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
            self.ir_gen.gen_print_text(text, row, col, attr)
            self.need_print_string = True

    def _handle_print_char(self, node: ast.Call) -> None:
        if not node.args:
            return

        if isinstance(node.args[0], ast.Constant):
            char = node.args[0].value
            self.ir_gen.gen_print_char(char)

    def _handle_loop_forever(self, node: ast.Call) -> None:
        self.ir_gen.gen_loop_forever()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="pxOS Python Compiler v2 - Compile Python to PXI Assembly IR",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("input", type=Path, help="Input Python file")
    parser.add_argument("-o", "--output", type=Path, help="Output IR JSON file")
    parser.add_argument("--build", action="store_true", help="Compile IR to primitives and build")
    parser.add_argument("--run", action="store_true", help="Compile, build, and run in QEMU")
    parser.add_argument("--show-ir", action="store_true", help="Print IR (don't compile)")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        return 1

    try:
        # Step 1: Python → IR
        print(f"Compiling {args.input} to PXI Assembly IR...")
        compiler = PythonToIR(args.input)
        ir_program = compiler.compile()

        # Save or display IR
        ir_json = ir_program.to_json()
        ir_file = args.output or args.input.with_suffix('.pxi.json')
        ir_file.write_text(ir_json)
        print(f"  Generated IR: {ir_file}")

        if args.show_ir:
            print("\n" + "="*60)
            print("PXI ASSEMBLY IR:")
            print("="*60)
            print(ir_json)
            return 0

        # Step 2: IR → Primitives (if building)
        if args.build or args.run:
            print(f"Compiling IR to primitives...")
            primitives_file = Path("pxos_commands.txt")

            # Call ir_compiler.py
            ir_compiler = Path(__file__).parent / "ir_compiler.py"
            result = subprocess.run(
                [sys.executable, str(ir_compiler), str(ir_file), "-o", str(primitives_file)],
                capture_output=True, text=True
            )

            if result.returncode != 0:
                print(f"IR compilation failed:", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
                return 1

            print(f"  Generated primitives: {primitives_file}")

            # Step 3: Primitives → Binary
            print("Building pxOS binary...")
            build_script = Path(__file__).parent.parent / "build_pxos.py"
            result = subprocess.run(
                [sys.executable, str(build_script)],
                capture_output=True, text=True
            )

            if result.returncode != 0:
                print("Build failed:", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
                return 1

            print(result.stdout)

        # Step 4: Run in QEMU
        if args.run:
            print("\nLaunching QEMU...")
            subprocess.run(["qemu-system-i386", "-fda", "pxos.bin"])

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
