#!/usr/bin/env python3
"""
Python → PXI Translator

Compile simple Python functions to pixel-native PXI modules.

This is the first step toward self-hosting: Python code becomes pixels.

Usage:
    python_to_pxi.py my_function.py output.pxi.png
"""

import ast
import sys
from PIL import Image
from pathlib import Path
from typing import List, Tuple
from pxi_cpu import *


class PXITranslator:
    """Translate Python AST to PXI bytecode"""

    def __init__(self):
        self.code = []  # List of (opcode, arg1, arg2, arg3) tuples
        self.variables = {}  # var_name -> register mapping
        self.next_reg = 0
        self.label_counter = 0
        self.labels = {}  # label_name -> address

    def allocate_register(self, var_name: str = None) -> int:
        """Allocate a register for a variable"""
        if var_name and var_name in self.variables:
            return self.variables[var_name]

        if self.next_reg >= 16:
            raise RuntimeError("Out of registers (max 16)")

        reg = self.next_reg
        self.next_reg += 1

        if var_name:
            self.variables[var_name] = reg

        return reg

    def emit(self, opcode, arg1=0, arg2=0, arg3=0):
        """Emit a PXI instruction"""
        self.code.append((opcode, arg1, arg2, arg3))
        return len(self.code) - 1  # Return address

    def get_label(self, name: str) -> str:
        """Create a unique label"""
        label = f"{name}_{self.label_counter}"
        self.label_counter += 1
        return label

    def mark_label(self, label: str):
        """Mark current position with a label"""
        self.labels[label] = len(self.code)

    def compile_function(self, func_def: ast.FunctionDef) -> List[Tuple[int, int, int, int]]:
        """Compile a Python function to PXI"""

        print(f"Compiling function: {func_def.name}")

        # Function entry point
        entry_point = len(self.code)

        # Map arguments to registers
        for i, arg in enumerate(func_def.args.args):
            arg_name = arg.arg
            self.variables[arg_name] = i  # R0, R1, R2, ...
            print(f"  Arg: {arg_name} → R{i}")

        # Compile function body
        for stmt in func_def.body:
            self.compile_stmt(stmt)

        # If no explicit return, add RET
        if not self.code or self.code[-1][0] != OP_RET:
            self.emit(OP_RET)

        print(f"  Generated {len(self.code) - entry_point} instructions")

        return entry_point

    def compile_stmt(self, stmt):
        """Compile a statement"""

        if isinstance(stmt, ast.Return):
            # Return statement
            if stmt.value:
                result_reg = self.compile_expr(stmt.value)
                if result_reg != 0:  # Move result to R0
                    self.emit(OP_LOAD, 0, result_reg, 0)
            self.emit(OP_RET)

        elif isinstance(stmt, ast.Assign):
            # Assignment: var = expr
            target = stmt.targets[0]
            if isinstance(target, ast.Name):
                var_name = target.id
                value_reg = self.compile_expr(stmt.value)

                # Allocate register for this variable
                var_reg = self.allocate_register(var_name)

                # Move value to variable's register
                if value_reg != var_reg:
                    self.emit(OP_LOAD, var_reg, value_reg, 0)

        elif isinstance(stmt, ast.Expr):
            # Expression statement (e.g., function call)
            self.compile_expr(stmt.value)

        elif isinstance(stmt, ast.If):
            # If statement
            cond_reg = self.compile_expr(stmt.test)

            # Jump if false
            else_label = self.get_label("else")
            end_label = self.get_label("endif")

            # JNZ cond_reg, else_label (inverted logic)
            jnz_addr = self.emit(OP_JNZ, cond_reg, 0, 0)  # Patch later

            # Then branch
            for s in stmt.body:
                self.compile_stmt(s)

            # Jump to end
            jmp_addr = self.emit(OP_JMP, 0, 0, 0)  # Patch later

            # Else branch
            self.mark_label(else_label)
            if stmt.orelse:
                for s in stmt.orelse:
                    self.compile_stmt(s)

            # End
            self.mark_label(end_label)

            # Patch jump addresses
            # (Simplified - would need proper address calculation)

        elif isinstance(stmt, ast.While):
            # While loop
            loop_start = self.get_label("loop_start")
            loop_end = self.get_label("loop_end")

            self.mark_label(loop_start)

            # Compile condition
            cond_reg = self.compile_expr(stmt.test)

            # Jump if false
            jnz_addr = self.emit(OP_JNZ, cond_reg, 0, 0)  # Patch later

            # Loop body
            for s in stmt.body:
                self.compile_stmt(s)

            # Jump back to start
            loop_start_addr = self.labels[loop_start]
            self.emit(OP_JMP, 0, loop_start_addr >> 8, loop_start_addr & 0xFF)

            self.mark_label(loop_end)

        else:
            print(f"  Warning: Unsupported statement {type(stmt).__name__}")

    def compile_expr(self, expr) -> int:
        """Compile an expression, return register containing result"""

        if isinstance(expr, ast.Constant):
            # Constant value
            reg = self.allocate_register()
            value = expr.value
            if isinstance(value, int):
                self.emit(OP_LOAD, reg, value & 0xFF, 0)
            return reg

        elif isinstance(expr, ast.Name):
            # Variable reference
            var_name = expr.id
            if var_name in self.variables:
                return self.variables[var_name]
            else:
                raise RuntimeError(f"Undefined variable: {var_name}")

        elif isinstance(expr, ast.BinOp):
            # Binary operation
            left_reg = self.compile_expr(expr.left)
            right_reg = self.compile_expr(expr.right)

            result_reg = self.allocate_register()

            if isinstance(expr.op, ast.Add):
                self.emit(OP_ADD, result_reg, left_reg, right_reg)
            elif isinstance(expr.op, ast.Sub):
                self.emit(OP_SUB, result_reg, left_reg, right_reg)
            else:
                raise RuntimeError(f"Unsupported operator: {type(expr.op).__name__}")

            return result_reg

        elif isinstance(expr, ast.Call):
            # Function call (simplified - no args yet)
            # Would need to push args, CALL, pop result
            print("  Warning: Function calls not yet fully supported")
            return 0

        else:
            print(f"  Warning: Unsupported expression {type(expr).__name__}")
            return 0

    def create_module_image(self, func_name: str = "module") -> Image.Image:
        """Create a PXI module image from compiled code"""

        # Calculate module size (must be power of 2 or reasonable size)
        code_size = len(self.code)
        img_size = 64  # Start with 64x64
        while img_size * img_size < code_size + 256:  # Reserve header space
            img_size *= 2

        print(f"Creating module image: {img_size}x{img_size}")

        img = Image.new("RGBA", (img_size, img_size), (0, 0, 0, 255))

        def set_pixel(pc, r, g, b, a=0):
            x = pc % img_size
            y = pc // img_size
            img.putpixel((x, y), (r, g, b, a))

        # Header
        set_pixel(0, 0x50, 0x58, 0x49, 0x4D)  # "PXIM" magic
        set_pixel(1, 1, 0, 0, 0)               # Version 1.0.0
        set_pixel(2, 1, 0, 0, 0)               # Entry point = 256
        set_pixel(3, code_size >> 8, code_size & 0xFF, 0, 0)

        # Module table (one function for now)
        set_pixel(64, 0, 0, 0, 0)              # Function ID = 0
        set_pixel(65, 0, 0, 0, 0)              # Name hash (TODO)
        set_pixel(66, 1, 0, 0, 0)              # Entry = 256

        # Code
        for i, (opcode, arg1, arg2, arg3) in enumerate(self.code):
            set_pixel(256 + i, opcode, arg1, arg2, arg3)

        return img


def compile_python_to_pxi(python_source: str, output_path: str):
    """Compile Python source to PXI module"""

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║          PYTHON → PXI TRANSLATOR                          ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

    # Parse Python source
    tree = ast.parse(python_source)

    # Find function definitions
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    if not functions:
        print("Error: No functions found in source")
        return

    print(f"Found {len(functions)} function(s)")
    print()

    # Compile first function
    translator = PXITranslator()
    func = functions[0]

    entry_point = translator.compile_function(func)

    print()
    print("Compilation summary:")
    print(f"  Instructions: {len(translator.code)}")
    print(f"  Registers used: {translator.next_reg}/16")
    print(f"  Entry point: {entry_point}")
    print()

    # Create module image
    img = translator.create_module_image(func.name)
    img.save(output_path)

    print(f"✓ Module saved: {output_path}")
    print()
    print("To run:")
    print(f"  pxos_boot.py {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compile Python to PXI pixel modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python_to_pxi.py my_func.py output.pxi.png
  python_to_pxi.py --source "def add(a, b): return a + b" add.pxi.png
        """
    )

    parser.add_argument("input", nargs="?", help="Python source file")
    parser.add_argument("output", nargs="?", default="output.pxi.png",
                        help="Output PXI module (PNG)")
    parser.add_argument("--source", "-s", help="Python source code (inline)")

    args = parser.parse_args()

    if args.source:
        source = args.source
        output = args.output if args.output else "output.pxi.png"
    elif args.input:
        source = Path(args.input).read_text()
        output = args.output if args.output else args.input.replace(".py", ".pxi.png")
    else:
        parser.print_help()
        sys.exit(1)

    compile_python_to_pxi(source, output)


if __name__ == "__main__":
    main()
