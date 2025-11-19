#!/usr/bin/env python3
"""
GPU Primitive Parser and Data Structures
Part of pxOS Heterogeneous Computing System

Parses GPU primitive syntax into structured data
that can be compiled to CUDA, PTX, or OpenCL.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class ParamType(Enum):
    """GPU kernel parameter types"""
    INT = "int"
    FLOAT = "float"
    INT_ARRAY = "int[]"
    FLOAT_ARRAY = "float[]"
    INT_PTR = "int*"
    FLOAT_PTR = "float*"


class OpType(Enum):
    """GPU operation types"""
    THREAD_ID = "THREAD_ID"
    BLOCK_ID = "BLOCK_ID"
    LOCAL_ID = "LOCAL_ID"
    LOAD = "LOAD"
    STORE = "STORE"
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    MOD = "MOD"
    COMPARE = "COMPARE"
    IF = "IF"
    ELSE = "ELSE"
    ENDIF = "ENDIF"
    SYNC_THREADS = "SYNC_THREADS"


@dataclass
class GPUParameter:
    """Represents a GPU kernel parameter"""
    name: str
    param_type: ParamType

    def to_cuda_param(self) -> str:
        """Convert to CUDA function parameter"""
        type_map = {
            ParamType.INT: "int",
            ParamType.FLOAT: "float",
            ParamType.INT_ARRAY: "int*",
            ParamType.FLOAT_ARRAY: "float*",
            ParamType.INT_PTR: "int*",
            ParamType.FLOAT_PTR: "float*",
        }
        return f"{type_map[self.param_type]} {self.name}"


@dataclass
class GPUOperation:
    """Represents a single GPU operation"""
    op_type: OpType
    operands: List[str] = field(default_factory=list)
    result: Optional[str] = None
    indent: int = 0

    def to_cuda_code(self) -> str:
        """Convert operation to CUDA C code"""
        indent_str = "    " * self.indent

        if self.op_type == OpType.THREAD_ID:
            return f"{indent_str}int {self.result} = blockIdx.x * blockDim.x + threadIdx.x;"

        elif self.op_type == OpType.BLOCK_ID:
            return f"{indent_str}int {self.result} = blockIdx.x;"

        elif self.op_type == OpType.LOCAL_ID:
            return f"{indent_str}int {self.result} = threadIdx.x;"

        elif self.op_type == OpType.LOAD:
            # operands[0] is source like "array[tid]"
            return f"{indent_str}auto {self.result} = {self.operands[0]};"

        elif self.op_type == OpType.STORE:
            # operands[0] is value, result is destination like "array[tid]"
            return f"{indent_str}{self.result} = {self.operands[0]};"

        elif self.op_type == OpType.ADD:
            return f"{indent_str}auto {self.result} = {self.operands[0]} + {self.operands[1]};"

        elif self.op_type == OpType.SUB:
            return f"{indent_str}auto {self.result} = {self.operands[0]} - {self.operands[1]};"

        elif self.op_type == OpType.MUL:
            return f"{indent_str}auto {self.result} = {self.operands[0]} * {self.operands[1]};"

        elif self.op_type == OpType.DIV:
            return f"{indent_str}auto {self.result} = {self.operands[0]} / {self.operands[1]};"

        elif self.op_type == OpType.COMPARE:
            # operands: [left, operator, right]
            return f"{indent_str}bool {self.result} = ({self.operands[0]} {self.operands[1]} {self.operands[2]});"

        elif self.op_type == OpType.IF:
            return f"{indent_str}if ({self.operands[0]}) {{"

        elif self.op_type == OpType.ELSE:
            return f"{indent_str}}} else {{"

        elif self.op_type == OpType.ENDIF:
            return f"{indent_str}}}"

        elif self.op_type == OpType.SYNC_THREADS:
            return f"{indent_str}__syncthreads();"

        return f"{indent_str}// Unknown operation: {self.op_type}"


@dataclass
class GPUKernel:
    """Represents a complete GPU kernel"""
    name: str
    parameters: List[GPUParameter] = field(default_factory=list)
    operations: List[GPUOperation] = field(default_factory=list)
    shared_memory: Dict[str, tuple] = field(default_factory=dict)  # name -> (type, size)

    def to_cuda_code(self) -> str:
        """Generate complete CUDA kernel code"""
        # Build parameter list
        params = ", ".join(p.to_cuda_param() for p in self.parameters)

        # Start kernel definition
        code = f"__global__ void {self.name}({params}) {{\n"

        # Add shared memory declarations
        for name, (stype, size) in self.shared_memory.items():
            code += f"    __shared__ {stype} {name}[{size}];\n"

        # Add operations
        for op in self.operations:
            code += op.to_cuda_code() + "\n"

        code += "}\n"
        return code


@dataclass
class GPULaunch:
    """Represents a GPU kernel launch"""
    kernel_name: str
    blocks: int
    threads: int

    def to_cuda_code(self, params: List[str]) -> str:
        """Generate CUDA kernel launch code"""
        param_str = ", ".join(params)
        return f"{self.kernel_name}<<<{self.blocks}, {self.threads}>>>({param_str});"


class GPUPrimitiveParser:
    """Parses GPU primitive syntax"""

    def __init__(self):
        self.kernels: Dict[str, GPUKernel] = {}
        self.launches: List[GPULaunch] = []
        self.current_kernel: Optional[GPUKernel] = None
        self.indent_level = 0

    def parse_line(self, line: str, line_num: int) -> None:
        """Parse a single line of GPU primitives"""
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith('#'):
            return

        parts = line.split()
        if not parts:
            return

        cmd = parts[0].upper()

        try:
            if cmd == "GPU_KERNEL":
                self._parse_kernel(parts)

            elif cmd == "GPU_PARAM":
                self._parse_param(parts)

            elif cmd == "GPU_SHARED":
                self._parse_shared(parts)

            elif cmd == "GPU_THREAD_CODE:":
                # Just a marker, operations follow
                pass

            elif cmd == "GPU_END":
                self._end_kernel()

            elif cmd == "GPU_LAUNCH":
                self._parse_launch(parts)

            elif cmd == "GPU_SYNC":
                # Will add cudaDeviceSynchronize() later
                pass

            elif cmd in ["THREAD_ID", "BLOCK_ID", "LOCAL_ID"]:
                self._parse_id_operation(cmd, parts)

            elif cmd == "LOAD":
                self._parse_load(parts)

            elif cmd == "STORE":
                self._parse_store(parts)

            elif cmd in ["ADD", "SUB", "MUL", "DIV", "MOD"]:
                self._parse_arithmetic(cmd, parts)

            elif cmd == "COMPARE":
                self._parse_compare(parts)

            elif cmd == "IF":
                self._parse_if(parts)

            elif cmd == "ELSE:":
                self._parse_else()

            elif cmd == "ENDIF":
                self._parse_endif()

            elif cmd == "SYNC_THREADS":
                self._parse_sync_threads()

        except Exception as e:
            raise ValueError(f"Line {line_num}: {line}\nError: {e}")

    def _parse_kernel(self, parts: List[str]) -> None:
        """Parse GPU_KERNEL command"""
        if len(parts) < 2:
            raise ValueError("GPU_KERNEL requires kernel name")

        kernel_name = parts[1]
        self.current_kernel = GPUKernel(name=kernel_name)
        self.indent_level = 0

    def _parse_param(self, parts: List[str]) -> None:
        """Parse GPU_PARAM command"""
        if not self.current_kernel:
            raise ValueError("GPU_PARAM outside of kernel definition")

        if len(parts) < 3:
            raise ValueError("GPU_PARAM requires name and type")

        name = parts[1]
        type_str = parts[2]

        # Map string to ParamType
        type_map = {
            "int": ParamType.INT,
            "float": ParamType.FLOAT,
            "int[]": ParamType.INT_ARRAY,
            "float[]": ParamType.FLOAT_ARRAY,
            "int*": ParamType.INT_PTR,
            "float*": ParamType.FLOAT_PTR,
        }

        if type_str not in type_map:
            raise ValueError(f"Unknown parameter type: {type_str}")

        param = GPUParameter(name=name, param_type=type_map[type_str])
        self.current_kernel.parameters.append(param)

    def _parse_shared(self, parts: List[str]) -> None:
        """Parse GPU_SHARED command"""
        if not self.current_kernel:
            raise ValueError("GPU_SHARED outside of kernel definition")

        if len(parts) < 4:
            raise ValueError("GPU_SHARED requires name, type, and size")

        name = parts[1]
        stype = parts[2]
        size = int(parts[3])

        self.current_kernel.shared_memory[name] = (stype, size)

    def _parse_id_operation(self, cmd: str, parts: List[str]) -> None:
        """Parse THREAD_ID, BLOCK_ID, LOCAL_ID operations"""
        if not self.current_kernel:
            raise ValueError(f"{cmd} outside of kernel definition")

        # Format: THREAD_ID → var
        if len(parts) < 3 or parts[1] != '→':
            raise ValueError(f"{cmd} requires: {cmd} → variable")

        result_var = parts[2]
        op_type = OpType[cmd]

        op = GPUOperation(op_type=op_type, result=result_var, indent=self.indent_level)
        self.current_kernel.operations.append(op)

    def _parse_load(self, parts: List[str]) -> None:
        """Parse LOAD operation"""
        if not self.current_kernel:
            raise ValueError("LOAD outside of kernel definition")

        # Format: LOAD source → dest
        if len(parts) < 4 or parts[2] != '→':
            raise ValueError("LOAD requires: LOAD source → dest")

        source = parts[1]
        dest = parts[3]

        op = GPUOperation(
            op_type=OpType.LOAD,
            operands=[source],
            result=dest,
            indent=self.indent_level
        )
        self.current_kernel.operations.append(op)

    def _parse_store(self, parts: List[str]) -> None:
        """Parse STORE operation"""
        if not self.current_kernel:
            raise ValueError("STORE outside of kernel definition")

        # Format: STORE source → dest
        if len(parts) < 4 or parts[2] != '→':
            raise ValueError("STORE requires: STORE source → dest")

        source = parts[1]
        dest = parts[3]

        op = GPUOperation(
            op_type=OpType.STORE,
            operands=[source],
            result=dest,
            indent=self.indent_level
        )
        self.current_kernel.operations.append(op)

    def _parse_arithmetic(self, cmd: str, parts: List[str]) -> None:
        """Parse arithmetic operations"""
        if not self.current_kernel:
            raise ValueError(f"{cmd} outside of kernel definition")

        # Format: ADD a b → result
        if len(parts) < 5 or parts[3] != '→':
            raise ValueError(f"{cmd} requires: {cmd} a b → result")

        op_a = parts[1]
        op_b = parts[2]
        result = parts[4]

        op = GPUOperation(
            op_type=OpType[cmd],
            operands=[op_a, op_b],
            result=result,
            indent=self.indent_level
        )
        self.current_kernel.operations.append(op)

    def _parse_compare(self, parts: List[str]) -> None:
        """Parse COMPARE operation"""
        if not self.current_kernel:
            raise ValueError("COMPARE outside of kernel definition")

        # Format: COMPARE a < b → result
        if len(parts) < 6 or parts[4] != '→':
            raise ValueError("COMPARE requires: COMPARE a op b → result")

        op_a = parts[1]
        operator = parts[2]
        op_b = parts[3]
        result = parts[5]

        op = GPUOperation(
            op_type=OpType.COMPARE,
            operands=[op_a, operator, op_b],
            result=result,
            indent=self.indent_level
        )
        self.current_kernel.operations.append(op)

    def _parse_if(self, parts: List[str]) -> None:
        """Parse IF statement"""
        if not self.current_kernel:
            raise ValueError("IF outside of kernel definition")

        # Format: IF condition:
        if len(parts) < 2:
            raise ValueError("IF requires condition")

        condition = parts[1].rstrip(':')

        op = GPUOperation(
            op_type=OpType.IF,
            operands=[condition],
            indent=self.indent_level
        )
        self.current_kernel.operations.append(op)
        self.indent_level += 1

    def _parse_else(self) -> None:
        """Parse ELSE statement"""
        if not self.current_kernel:
            raise ValueError("ELSE outside of kernel definition")

        self.indent_level -= 1
        op = GPUOperation(op_type=OpType.ELSE, indent=self.indent_level)
        self.current_kernel.operations.append(op)
        self.indent_level += 1

    def _parse_endif(self) -> None:
        """Parse ENDIF statement"""
        if not self.current_kernel:
            raise ValueError("ENDIF outside of kernel definition")

        self.indent_level -= 1
        op = GPUOperation(op_type=OpType.ENDIF, indent=self.indent_level)
        self.current_kernel.operations.append(op)

    def _parse_sync_threads(self) -> None:
        """Parse SYNC_THREADS operation"""
        if not self.current_kernel:
            raise ValueError("SYNC_THREADS outside of kernel definition")

        op = GPUOperation(op_type=OpType.SYNC_THREADS, indent=self.indent_level)
        self.current_kernel.operations.append(op)

    def _end_kernel(self) -> None:
        """Finish current kernel definition"""
        if not self.current_kernel:
            raise ValueError("GPU_END without matching GPU_KERNEL")

        self.kernels[self.current_kernel.name] = self.current_kernel
        self.current_kernel = None
        self.indent_level = 0

    def _parse_launch(self, parts: List[str]) -> None:
        """Parse GPU_LAUNCH command"""
        # Format: GPU_LAUNCH kernel_name BLOCKS n THREADS m
        if len(parts) < 6:
            raise ValueError("GPU_LAUNCH requires: GPU_LAUNCH name BLOCKS n THREADS m")

        kernel_name = parts[1]
        blocks = int(parts[3])
        threads = int(parts[5])

        launch = GPULaunch(kernel_name=kernel_name, blocks=blocks, threads=threads)
        self.launches.append(launch)

    def parse_file(self, filepath: str) -> None:
        """Parse entire file of GPU primitives"""
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                self.parse_line(line, line_num)


def main():
    """Test the parser"""
    test_code = """
GPU_KERNEL vector_add
GPU_PARAM a float[]
GPU_PARAM b float[]
GPU_PARAM c float[]
GPU_PARAM n int

GPU_THREAD_CODE:
    THREAD_ID → tid
    COMPARE tid < n → in_bounds

    IF in_bounds:
        LOAD a[tid] → val_a
        LOAD b[tid] → val_b
        ADD val_a val_b → sum
        STORE sum → c[tid]
    ENDIF
GPU_END

GPU_LAUNCH vector_add BLOCKS 256 THREADS 256
"""

    parser = GPUPrimitiveParser()

    for line_num, line in enumerate(test_code.split('\n'), 1):
        parser.parse_line(line, line_num)

    # Print generated CUDA code
    for kernel_name, kernel in parser.kernels.items():
        print(kernel.to_cuda_code())


if __name__ == "__main__":
    main()
