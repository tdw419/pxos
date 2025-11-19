"""
GLSL Backend Emitter
Converts an IRKernel object into a GLSL source string.
"""

from .ir import IRKernel, IRParameter, IRInstruction, IROperand, IROperation, IRType

class GLSLEmitter:
    def emit(self, kernel: IRKernel) -> str:
        """Converts an IRKernel into a GLSL source string."""
        lines = []
        lines.append("#version 430\n")
        lines.append(self._emit_workgroup_layout(kernel))
        lines.extend(self._emit_parameters(kernel.parameters))
        lines.append("\nvoid main() {")
        lines.extend([f"    {self._emit_instruction(inst)}" for inst in kernel.instructions])
        lines.append("}")
        return "\n".join(lines)

    def _emit_workgroup_layout(self, kernel: IRKernel) -> str:
        x, y, z = kernel.workgroup_size
        return f"layout(local_size_x = {x}, local_size_y = {y}, local_size_z = {z}) in;"

    def _emit_parameters(self, params: list[IRParameter]) -> list[str]:
        param_lines = []
        for p in params:
            qualifiers = " ".join(p.qualifiers)
            param_lines.append(
                f"layout(binding = {p.binding}, rgba32f) uniform {qualifiers} {p.ir_type.value} {p.name};"
            )
        return param_lines

    def _emit_operand(self, operand: IROperand) -> str:
        return operand.name

    def _emit_instruction(self, instr: IRInstruction) -> str:
        if instr.op == IROperation.LOAD_IMAGE:
            dest = self._emit_operand(instr.result)
            src_image = self._emit_operand(instr.operands[0])
            coord = self._emit_operand(instr.operands[1])
            return f"{instr.result.ir_type.value} {dest} = imageLoad({src_image}, {coord});"

        if instr.op == IROperation.STORE_IMAGE:
            dest_image = self._emit_operand(instr.operands[0])
            coord = self._emit_operand(instr.operands[1])
            value = self._emit_operand(instr.operands[2])
            return f"imageStore({dest_image}, {coord}, {value});"

        if instr.op == IROperation.SUBTRACT:
            dest = self._emit_operand(instr.result)
            op1 = self._emit_operand(instr.operands[0])
            op2 = self._emit_operand(instr.operands[1])
            return f"vec3 {dest} = {op1} - {op2};"

        if instr.op == IROperation.CONSTRUCT_FLOAT4:
            dest = self._emit_operand(instr.result)
            op1 = self._emit_operand(instr.operands[0])
            op2 = self._emit_operand(instr.operands[1])
            return f"{instr.result.ir_type.value} {dest} = vec4({op1}, {op2});"

        raise NotImplementedError(f"GLSL emission for {instr.op} not implemented.")
