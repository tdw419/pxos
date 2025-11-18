"""
pxIR: Core IR Data Structures

Inspired by LLVM, MLIR, TVM, and SPIR-V, but minimal and focused.

Key concepts:
- SSA (Static Single Assignment) form
- Strongly typed operations
- First-class matrix and vector types
- Address spaces for different memory regions
- Support for ML operations (MATMUL, RELU) and graphics operations (DRAW_GLYPH, BLIT)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Union, Any
from enum import Enum


# -----------------------------------------------------------------------------
# Type System
# -----------------------------------------------------------------------------

class TypeKind(Enum):
    """Type categories in pxIR."""
    VOID = "void"
    I8 = "i8"
    I16 = "i16"
    I32 = "i32"
    I64 = "i64"
    U8 = "u8"
    U16 = "u16"
    U32 = "u32"
    U64 = "u64"
    F32 = "f32"
    F64 = "f64"
    VECTOR = "vector"     # vec4<f32>, vec16<i32>
    MATRIX = "matrix"     # mat128x256<f32>
    POINTER = "pointer"   # ptr<i32, framebuffer>
    FUNCTION = "function"


class AddressSpace(Enum):
    """Memory address spaces (SPIR-V inspired)."""
    MEM = "mem"                     # Main memory (cached)
    FRAMEBUFFER = "framebuffer"     # Video memory (write-combined)
    IO = "io"                       # I/O ports (uncached)
    CONSTANT = "constant"           # ROM (read-only)


@dataclass
class Type:
    """Type representation."""
    kind: TypeKind

    # For vectors: element type and count
    elem_type: Optional[Type] = None
    vec_size: Optional[int] = None

    # For matrices: element type and dimensions
    rows: Optional[int] = None
    cols: Optional[int] = None

    # For pointers: pointee type and address space
    pointee_type: Optional[Type] = None
    addr_space: Optional[AddressSpace] = None

    # For quantized types
    is_quantized: bool = False
    scale: Optional[float] = None
    zero_point: Optional[int] = None

    def __str__(self) -> str:
        if self.kind == TypeKind.VECTOR:
            return f"vec{self.vec_size}<{self.elem_type}>"
        elif self.kind == TypeKind.MATRIX:
            q = f", quantized=True, scale={self.scale}" if self.is_quantized else ""
            return f"mat{self.rows}x{self.cols}<{self.elem_type}{q}>"
        elif self.kind == TypeKind.POINTER:
            return f"ptr<{self.pointee_type}, {self.addr_space.value}>"
        else:
            return self.kind.value

    @staticmethod
    def i8() -> Type:
        return Type(TypeKind.I8)

    @staticmethod
    def i16() -> Type:
        return Type(TypeKind.I16)

    @staticmethod
    def i32() -> Type:
        return Type(TypeKind.I32)

    @staticmethod
    def i64() -> Type:
        return Type(TypeKind.I64)

    @staticmethod
    def u8() -> Type:
        return Type(TypeKind.U8)

    @staticmethod
    def u32() -> Type:
        return Type(TypeKind.U32)

    @staticmethod
    def f32() -> Type:
        return Type(TypeKind.F32)

    @staticmethod
    def f64() -> Type:
        return Type(TypeKind.F64)

    @staticmethod
    def void() -> Type:
        return Type(TypeKind.VOID)

    @staticmethod
    def vector(elem_type: Type, size: int) -> Type:
        return Type(TypeKind.VECTOR, elem_type=elem_type, vec_size=size)

    @staticmethod
    def matrix(elem_type: Type, rows: int, cols: int, is_quantized: bool = False, scale: Optional[float] = None) -> Type:
        return Type(TypeKind.MATRIX, elem_type=elem_type, rows=rows, cols=cols,
                   is_quantized=is_quantized, scale=scale)

    @staticmethod
    def pointer(pointee_type: Type, addr_space: AddressSpace = AddressSpace.MEM) -> Type:
        return Type(TypeKind.POINTER, pointee_type=pointee_type, addr_space=addr_space)


# -----------------------------------------------------------------------------
# Values (SSA)
# -----------------------------------------------------------------------------

@dataclass
class Value:
    """
    An SSA value - produced by an operation, used by other operations.

    Each Value has:
    - A unique name (e.g., %0, %1, %temp)
    - A type
    - Optional constant value (for constant propagation)
    """
    name: str
    ty: Type
    is_const: bool = False
    const_value: Any = None

    def __str__(self) -> str:
        if self.is_const:
            return f"{self.name}: {self.ty} = {self.const_value}"
        return f"{self.name}: {self.ty}"

    def __repr__(self) -> str:
        return f"Value({self.name}, {self.ty})"

    def __hash__(self) -> int:
        """Make Value hashable for use in sets/dicts."""
        return hash(self.name)

    def __eq__(self, other) -> bool:
        """Compare values by name (identity)."""
        if not isinstance(other, Value):
            return False
        return self.name == other.name


# -----------------------------------------------------------------------------
# Operations
# -----------------------------------------------------------------------------

@dataclass
class Op:
    """
    An operation in the IR.

    Each operation:
    - Has an opcode (ADD, MATMUL, DRAW_GLYPH, etc.)
    - Takes operands (Values or immediates)
    - Produces a result (Value) or has side effects
    - May have attributes (metadata)
    """
    op: str                                    # Opcode
    operands: List[Union[Value, int, str]]    # Operands (Values or immediates)
    result: Optional[Value] = None            # Result value (if any)
    attributes: dict = field(default_factory=dict)  # Extra metadata
    has_side_effects: bool = False            # True for STORE, PRINT_STR, DRAW_GLYPH, etc.

    def __str__(self) -> str:
        result_str = f"{self.result.name} = " if self.result else ""
        operands_str = ", ".join(str(o) if isinstance(o, Value) else repr(o) for o in self.operands)
        attrs_str = f" {self.attributes}" if self.attributes else ""
        return f"{result_str}{self.op}({operands_str}){attrs_str}"


# -----------------------------------------------------------------------------
# Basic Blocks
# -----------------------------------------------------------------------------

@dataclass
class Block:
    """
    A basic block - a sequence of operations with a single entry and exit.
    """
    name: str
    ops: List[Op] = field(default_factory=list)

    def add_op(self, op: Op) -> None:
        """Add an operation to this block."""
        self.ops.append(op)

    def __str__(self) -> str:
        ops_str = "\n  ".join(str(op) for op in self.ops)
        return f"{self.name}:\n  {ops_str}"


# -----------------------------------------------------------------------------
# Program (Top-level)
# -----------------------------------------------------------------------------

@dataclass
class Program:
    """
    A complete pxIR program.

    Contains:
    - Multiple basic blocks
    - Global values (constants, globals)
    - Type definitions
    """
    name: str
    blocks: List[Block] = field(default_factory=list)
    globals: List[Value] = field(default_factory=list)

    def add_block(self, block: Block) -> None:
        """Add a basic block to the program."""
        self.blocks.append(block)

    def add_global(self, value: Value) -> None:
        """Add a global value."""
        self.globals.append(value)

    def replace_all_uses(self, old: Value, new: Value) -> None:
        """
        Replace all uses of Value `old` with `new` throughout the program.
        This is the LLVM-style RAUW (Replace All Uses With).
        """
        for block in self.blocks:
            for op in block.ops:
                for i, operand in enumerate(op.operands):
                    if operand is old:
                        op.operands[i] = new

    def get_return_values(self) -> List[Value]:
        """Collect values that are returned by functions/entry blocks."""
        rets = []
        for block in self.blocks:
            for op in block.ops:
                if op.op == "RET" and op.operands:
                    v = op.operands[0]
                    if isinstance(v, Value):
                        rets.append(v)
        return rets

    def pretty(self) -> str:
        """Pretty-print the program."""
        lines = [f"program {self.name} {{"]

        if self.globals:
            lines.append("\n  // Globals")
            for g in self.globals:
                lines.append(f"  {g}")

        for block in self.blocks:
            lines.append(f"\n  {block}")

        lines.append("}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.pretty()


# -----------------------------------------------------------------------------
# IR Builder (Helper for constructing IR)
# -----------------------------------------------------------------------------

class IRBuilder:
    """
    Helper for building pxIR programs.
    Maintains current block and SSA value counter.
    """

    def __init__(self, program: Program):
        self.program = program
        self.current_block: Optional[Block] = None
        self.value_counter = 0

    def create_block(self, name: str) -> Block:
        """Create a new basic block."""
        block = Block(name)
        self.program.add_block(block)
        return block

    def set_insert_point(self, block: Block) -> None:
        """Set the current block for inserting operations."""
        self.current_block = block

    def fresh_value(self, ty: Type, prefix: str = "v") -> Value:
        """Generate a fresh SSA value."""
        name = f"%{prefix}{self.value_counter}"
        self.value_counter += 1
        return Value(name, ty)

    def const_value(self, ty: Type, value: Any, prefix: str = "c") -> Value:
        """Create a constant value."""
        name = f"%{prefix}{self.value_counter}"
        self.value_counter += 1
        return Value(name, ty, is_const=True, const_value=value)

    def emit(self, op: Op) -> Optional[Value]:
        """Emit an operation to the current block."""
        if self.current_block is None:
            raise ValueError("No insert point set!")
        self.current_block.add_op(op)
        return op.result

    # Convenience methods for common operations

    def add(self, lhs: Value, rhs: Value) -> Value:
        """Emit ADD operation."""
        result = self.fresh_value(lhs.ty, "add")
        self.emit(Op("ADD", [lhs, rhs], result))
        return result

    def sub(self, lhs: Value, rhs: Value) -> Value:
        """Emit SUB operation."""
        result = self.fresh_value(lhs.ty, "sub")
        self.emit(Op("SUB", [lhs, rhs], result))
        return result

    def mul(self, lhs: Value, rhs: Value) -> Value:
        """Emit MUL operation."""
        result = self.fresh_value(lhs.ty, "mul")
        self.emit(Op("MUL", [lhs, rhs], result))
        return result

    def div(self, lhs: Value, rhs: Value) -> Value:
        """Emit DIV operation."""
        result = self.fresh_value(lhs.ty, "div")
        self.emit(Op("DIV", [lhs, rhs], result))
        return result

    def matmul(self, lhs: Value, rhs: Value, result_type: Type) -> Value:
        """Emit MATMUL operation."""
        result = self.fresh_value(result_type, "matmul")
        self.emit(Op("MATMUL", [lhs, rhs], result))
        return result

    def relu(self, x: Value) -> Value:
        """Emit RELU operation."""
        result = self.fresh_value(x.ty, "relu")
        self.emit(Op("RELU", [x], result))
        return result

    def draw_glyph(self, glyph_id: int, x: int, y: int) -> None:
        """Emit DRAW_GLYPH operation (side effect)."""
        self.emit(Op("DRAW_GLYPH", [glyph_id, x, y], has_side_effects=True))

    def print_str(self, s: str) -> None:
        """Emit PRINT_STR operation (side effect)."""
        self.emit(Op("PRINT_STR", [s], has_side_effects=True))

    def ret(self, value: Optional[Value] = None) -> None:
        """Emit RET operation."""
        operands = [value] if value else []
        self.emit(Op("RET", operands, has_side_effects=True))
