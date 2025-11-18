#!/usr/bin/env python3
"""
tools/pxpyc.py - pxOS Python Compiler

Compiles a subset of Python into PXI Assembly IR in JSON format.
"""
import argparse
import json
import ast

class IRGenerator(ast.NodeVisitor):
    def __init__(self):
        self.ir = {
            "source": "",
            "ir_version": "1.0",
            "functions": {},
            "data": {}
        }
        self.current_function = None
        self.string_counter = 0

    def visit_FunctionDef(self, node):
        self.current_function = node.name
        self.ir["functions"][self.current_function] = {"instructions": []}
        self.generic_visit(node)
        self.current_function = None

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
            if function_name == 'print_text':
                self.gen_print_text(node.args[0].s)
            elif function_name == 'clear_screen':
                self.gen_clear_screen()
        self.generic_visit(node)

    def gen_clear_screen(self):
        """Generate IR for clear screen."""
        if self.current_function:
            self.ir["functions"][self.current_function]["instructions"].extend([
                {"op": "MOV", "dst": "AH", "src": "0x06"},
                {"op": "MOV", "dst": "AL", "src": "0x00"},
                {"op": "MOV", "dst": "BH", "src": "0x07"},
                {"op": "MOV", "dst": "CX", "src": "0x0000"},
                {"op": "MOV", "dst": "DX", "src": "0x184F"},
                {"op": "INT", "num": "0x10"},
            ])

    def gen_print_text(self, text):
        """Generate IR for print text."""
        if self.current_function:
            string_label = f"str_{self.string_counter}"
            self.string_counter += 1
            self.ir["data"][string_label] = text

            self.ir["functions"][self.current_function]["instructions"].extend([
                {"op": "MOV", "dst": "SI", "src": string_label},
                {"op": "CALL", "target": "print_string"},
            ])

def main():
    parser = argparse.ArgumentParser(description="Compile a subset of Python into PXI Assembly IR.")
    parser.add_argument("input_file", type=Path, help="The Python file to compile.")
    parser.add_argument("--output_file", type=Path, help="The output JSON file for the IR.")
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        source_code = f.read()

    tree = ast.parse(source_code)
    generator = IRGenerator()
    generator.ir['source'] = args.input_file.name
    generator.visit(tree)

    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(generator.ir, f, indent=2)
        print(f"✅ IR compiled to {args.output_file}")
    else:
        print(json.dumps(generator.ir, indent=2))

if __name__ == "__main__":
    from pathlib import Path
    main()
