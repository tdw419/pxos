"""
pxIR to PXI Assembly Code Generator

Compiles high-level pxIR to low-level PXI Assembly IR.

This backend lowers:
- Typed SSA operations → assembly instructions
- Matrix operations → library calls or inline assembly
- Block structure → labels and jumps
- Memory operations → load/store instructions

The output is PXI Assembly JSON that can be compiled by ir_compiler.py.
"""

import json
from typing import Dict, List, Optional
from pathlib import Path

from .ir import Program, Block, Op, Value, Type, TypeKind


class PxIRToPXI:
    """Lower pxIR to PXI Assembly"""

    def __init__(self):
        self.pxi_instructions = []
        self.pxi_data = []
        self.value_locations: Dict[str, str] = {}  # SSA value → register/memory
        self.string_counter = 0

    def generate(self, program: Program) -> dict:
        """Generate PXI Assembly from pxIR program"""
        self.pxi_instructions = []
        self.pxi_data = []
        self.value_locations = {}

        # Process each block
        for block in program.blocks:
            self._generate_block(block)

        # Return PXI Assembly JSON
        return {
            "source": "pxir_generated",
            "ir_version": "1.0",
            "origin": 0x7C00,
            "instructions": self.pxi_instructions,
            "data": self.pxi_data
        }

    def generate_to_file(self, program: Program, output_path: Path):
        """Generate PXI Assembly and write to file"""
        pxi_json = self.generate(program)
        output_path.write_text(json.dumps(pxi_json, indent=2))

    def _generate_block(self, block: Block):
        """Generate code for a basic block"""
        # Emit block label
        self._emit_label(block.name)

        # Generate code for each operation
        for op in block.ops:
            self._generate_op(op)

        # Generate terminator
        if block.terminator:
            self._generate_op(block.terminator)

    def _generate_op(self, op: Op):
        """Generate PXI Assembly for a single operation"""
        kind = op.kind

        if kind == "ADD":
            self._gen_add(op)
        elif kind == "SUB":
            self._gen_sub(op)
        elif kind == "MUL":
            self._gen_mul(op)
        elif kind == "DIV":
            self._gen_div(op)
        elif kind == "MATMUL":
            self._gen_matmul(op)
        elif kind == "RELU":
            self._gen_relu(op)
        elif kind == "LOAD":
            self._gen_load(op)
        elif kind == "STORE":
            self._gen_store(op)
        elif kind == "PRINT_STR":
            self._gen_print_str(op)
        elif kind == "DRAW_GLYPH":
            self._gen_draw_glyph(op)
        elif kind == "CALL":
            self._gen_call(op)
        elif kind == "BR":
            self._gen_br(op)
        elif kind == "JMP":
            self._gen_jmp(op)
        elif kind == "RET":
            self._gen_ret(op)
        else:
            raise NotImplementedError(f"Operation {kind} not yet supported in codegen")

    def _gen_add(self, op: Op):
        """Generate ADD"""
        # For now, use registers AX, BX
        # TODO: Proper register allocation
        lhs, rhs = op.args
        result = op.result

        # Load LHS to AX
        self._load_value_to_reg(lhs, "AX")

        # Load RHS to BX
        self._load_value_to_reg(rhs, "BX")

        # ADD AX, BX
        self._emit_instr("ADD", dst="AX", src="BX", comment=f"{result} = ADD {lhs} {rhs}")

        # Store result
        self._store_reg_to_value("AX", result)

    def _gen_sub(self, op: Op):
        """Generate SUB"""
        lhs, rhs = op.args
        result = op.result

        self._load_value_to_reg(lhs, "AX")
        self._load_value_to_reg(rhs, "BX")
        self._emit_instr("SUB", dst="AX", src="BX", comment=f"{result} = SUB {lhs} {rhs}")
        self._store_reg_to_value("AX", result)

    def _gen_mul(self, op: Op):
        """Generate MUL"""
        lhs, rhs = op.args
        result = op.result

        self._load_value_to_reg(lhs, "AX")
        self._load_value_to_reg(rhs, "BX")
        self._emit_instr("MUL", src="BX", comment=f"{result} = MUL {lhs} {rhs}")
        self._store_reg_to_value("AX", result)

    def _gen_div(self, op: Op):
        """Generate DIV"""
        lhs, rhs = op.args
        result = op.result

        self._load_value_to_reg(lhs, "AX")
        self._load_value_to_reg(rhs, "BX")
        # Zero DX for division
        self._emit_instr("XOR", dst="DX", src="DX")
        self._emit_instr("DIV", src="BX", comment=f"{result} = DIV {lhs} {rhs}")
        self._store_reg_to_value("AX", result)

    def _gen_matmul(self, op: Op):
        """Generate MATMUL (call library function)"""
        lhs, rhs = op.args
        result = op.result

        # For now, call a matmul library function
        # TODO: Inline matrix multiply for small matrices
        self._emit_comment(f"MATMUL {lhs} x {rhs} -> {result}")
        self._emit_instr("CALL", target="matmul_f32", comment="Matrix multiply")

        # Result is in some known location (convention)
        self._store_reg_to_value("AX", result)

    def _gen_relu(self, op: Op):
        """Generate RELU"""
        x = op.args[0]
        result = op.result

        # Load value
        self._load_value_to_reg(x, "AX")

        # ReLU: max(0, x)
        # Compare with 0, jump if positive
        self._emit_instr("CMP", op1="AX", op2=0, comment="ReLU: check if positive")
        # TODO: Proper conditional
        self._emit_comment("TODO: ReLU implementation")

        self._store_reg_to_value("AX", result)

    def _gen_load(self, op: Op):
        """Generate LOAD"""
        ptr = op.args[0]
        result = op.result

        # Load from memory pointed to by ptr
        # For now, assume ptr is in SI
        self._load_value_to_reg(ptr, "SI")
        self._emit_instr("MOV", dst="AL", src="[SI]", comment=f"LOAD {result} from {ptr}")

        self._store_reg_to_value("AL", result)

    def _gen_store(self, op: Op):
        """Generate STORE"""
        ptr, value = op.args

        self._load_value_to_reg(ptr, "SI")
        self._load_value_to_reg(value, "AL")
        self._emit_instr("MOV", dst="[SI]", src="AL", comment=f"STORE {value} to {ptr}")

    def _gen_print_str(self, op: Op):
        """Generate PRINT_STR"""
        # Check if first arg is a string constant
        arg = op.args[0]

        if isinstance(arg, str):
            # String literal - create data and print
            str_label = f"str_{self.string_counter}"
            self.string_counter += 1

            # Add string to data section
            self.pxi_data.append({
                "label": str_label,
                "type": "string",
                "value": arg
            })

            # Load string address and call print
            self._emit_instr("MOV", dst="SI", src=str_label, comment="Load string address")
            self._emit_instr("CALL", target="print_string", comment="Print string")

        else:
            # Runtime value - load and print
            self._load_value_to_reg(arg, "SI")
            self._emit_instr("CALL", target="print_string", comment="Print string")

    def _gen_draw_glyph(self, op: Op):
        """Generate DRAW_GLYPH"""
        glyph_id, x, y = op.args

        # Load arguments
        if isinstance(glyph_id, int):
            self._emit_instr("MOV", dst="AL", src=glyph_id, comment="Glyph ID")
        else:
            self._load_value_to_reg(glyph_id, "AL")

        if isinstance(x, int):
            self._emit_instr("MOV", dst="BX", src=x, comment="X position")
        else:
            self._load_value_to_reg(x, "BX")

        if isinstance(y, int):
            self._emit_instr("MOV", dst="CX", src=y, comment="Y position")
        else:
            self._load_value_to_reg(y, "CX")

        # Call draw function
        self._emit_instr("CALL", target="draw_glyph", comment="Draw glyph")

    def _gen_call(self, op: Op):
        """Generate CALL"""
        func_name = op.args[0]
        args = op.args[1:]
        result = op.result

        # For now, simple calling convention
        # TODO: Proper argument passing
        self._emit_instr("CALL", target=func_name, comment=f"Call {func_name}")

        if result:
            self._store_reg_to_value("AX", result)

    def _gen_br(self, op: Op):
        """Generate conditional branch"""
        cond, then_block, else_block = op.args

        # Load condition
        self._load_value_to_reg(cond, "AL")

        # Test and jump
        self._emit_instr("OR", dst="AL", src="AL", comment="Test condition")
        self._emit_instr("JNZ", target=then_block, comment=f"Branch to {then_block}")
        self._emit_instr("JMP", target=else_block, comment=f"Fallthrough to {else_block}")

    def _gen_jmp(self, op: Op):
        """Generate unconditional jump"""
        target = op.args[0]
        self._emit_instr("JMP", target=target, comment=f"Jump to {target}")

    def _gen_ret(self, op: Op):
        """Generate return"""
        if op.args:
            # Return value in AX
            value = op.args[0]
            self._load_value_to_reg(value, "AX")

        self._emit_instr("RET", comment="Return")

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _emit_label(self, name: str):
        """Emit a label"""
        self.pxi_instructions.append({
            "op": "LABEL",
            "operands": {"name": name},
            "comment": ""
        })

    def _emit_instr(self, op: str, comment: str = "", **operands):
        """Emit a PXI Assembly instruction"""
        self.pxi_instructions.append({
            "op": op,
            "operands": operands,
            "comment": comment
        })

    def _emit_comment(self, text: str):
        """Emit a comment"""
        self.pxi_instructions.append({
            "op": "COMMENT",
            "operands": {},
            "comment": text
        })

    def _load_value_to_reg(self, value: Value, reg: str):
        """Load SSA value to register"""
        # Check if value is already in a register
        if value.name in self.value_locations:
            loc = self.value_locations[value.name]
            if loc != reg:
                # Move from loc to reg
                self._emit_instr("MOV", dst=reg, src=loc, comment=f"Load {value.name}")
        else:
            # Value is a constant or needs to be loaded
            # Check metadata for constants
            if hasattr(value, 'metadata') and 'constant' in value.metadata:
                const_val = value.metadata['constant']
                self._emit_instr("MOV", dst=reg, src=const_val, comment=f"Load constant {const_val}")
            else:
                # Assume value is in memory - load it
                # TODO: Proper spilling/reloading
                self._emit_comment(f"TODO: Load {value.name} to {reg}")

    def _store_reg_to_value(self, reg: str, value: Value):
        """Store register to SSA value location"""
        self.value_locations[value.name] = reg


# ============================================================================
# Convenience Functions
# ============================================================================

def generate_pxi(program: Program) -> dict:
    """Generate PXI Assembly JSON from pxIR program"""
    codegen = PxIRToPXI()
    return codegen.generate(program)


def generate_pxi_file(program: Program, output_path: Path):
    """Generate PXI Assembly and write to file"""
    codegen = PxIRToPXI()
    codegen.generate_to_file(program, output_path)
