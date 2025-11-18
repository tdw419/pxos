#!/usr/bin/env python3
"""
tools/ir_compiler.py

PXI Assembly IR to Primitives Compiler.

Mechanically translates IR JSON into validated primitives.
This is where x86 knowledge lives - NOT in language compilers.
"""
import json
import argparse
from pathlib import Path

class IRCompiler:
    def __init__(self):
        self.primitives = []
        self.address = 0x7C00
        self.symbols = {}

    def compile(self, ir_json):
        """Main compilation pipeline"""
        # Pass 1: Allocate addresses for all functions/labels
        self.allocate_addresses(ir_json)

        # Pass 2: Generate primitives
        for func_name, func_body in ir_json.get('functions', {}).items():
            self.compile_function(func_name, func_body)

        return self.primitives

    def allocate_addresses(self, ir_json):
        """Allocate addresses for all functions and labels."""
        # This is a simplified allocation. A real implementation would be more robust.
        for func_name, func_body in ir_json.get('functions', {}).items():
            self.symbols[func_name] = self.address
            # A real implementation would calculate the size of the function.
            self.address += 0x10 # Placeholder size

    def compile_function(self, func_name, func_body):
        """Compile a single function from IR to primitives."""
        for instr in func_body.get('instructions', []):
            self.compile_instruction(instr)

    def compile_instruction(self, instr):
        """Translate single IR instruction into primitives."""
        op = instr.get('op')
        if op == 'MOV':
            self.emit_mov(instr.get('dst'), instr.get('src'))
        # A real implementation would handle all other opcodes.

    def emit_mov(self, dst, src):
        """Generate x86 MOV opcodes."""
        # This is a highly simplified MOV implementation.
        # A real implementation would handle registers, memory, and immediate values.
        if dst == "AH" and src.startswith("0x"):
            self.primitives.append({
                "type": "WRITE",
                "addr": hex(self.address),
                "byte": "0xB4"
            })
            self.address += 1
            self.primitives.append({
                "type": "WRITE",
                "addr": hex(self.address),
                "byte": src
            })
            self.address += 1

def main():
    parser = argparse.ArgumentParser(description="Compile PXI Assembly IR into pxOS primitives.")
    parser.add_argument("input_file", type=Path, help="The IR JSON file to compile.")
    parser.add_argument("--output_file", type=Path, help="The output JSON file for the primitives.")
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        ir_json = json.load(f)

    compiler = IRCompiler()
    primitives = compiler.compile(ir_json)

    output = {
        "feature": f"Compiled from {args.input_file.name}",
        "primitives": primitives
    }

    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"✅ Primitives compiled to {args.output_file}")
    else:
        print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
