"""
pxIR Core Data Structures

Defines the IR representation: Type, Value, Op, Block, Program
"""

from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json


# ============================================================================
# Type System
# ============================================================================

class TypeKind(Enum):
    """IR type kinds"""
    I8 = "i8"
    I16 = "i16"
    I32 = "i32"
    F32 = "f32"
    BOOL = "bool"
    VEC = "vec"
    MAT = "mat"
    PTR = "ptr"


@dataclass
class Type:
    """IR type"""
    kind: TypeKind
    # For vec: element type and length
    elem_type: Optional['Type'] = None
    length: Optional[int] = None
    # For mat: element type, rows, cols, quantization
    rows: Optional[int] = None
    cols: Optional[int] = None
    quant_scheme: Optional[str] = None
    quant_scale: Optional[float] = None
    quant_offset: Optional[int] = None
    # For ptr: address space and pointed-to type
    addr_space: Optional[str] = None
    pointee_type: Optional['Type'] = None

    def __str__(self):
        if self.kind == TypeKind.VEC:
            return f"vec<{self.elem_type},{self.length}>"
        elif self.kind == TypeKind.MAT:
            quant = f",{self.quant_scheme}" if self.quant_scheme else ""
            return f"mat<{self.elem_type},{self.rows},{self.cols}{quant}>"
        elif self.kind == TypeKind.PTR:
            return f"ptr<{self.addr_space},{self.pointee_type}>"
        else:
            return self.kind.value

    @staticmethod
    def i8():
        return Type(TypeKind.I8)

    @staticmethod
    def i16():
        return Type(TypeKind.I16)

    @staticmethod
    def i32():
        return Type(TypeKind.I32)

    @staticmethod
    def f32():
        return Type(TypeKind.F32)

    @staticmethod
    def bool_type():
        return Type(TypeKind.BOOL)

    @staticmethod
    def vec(elem_type: 'Type', length: int):
        return Type(TypeKind.VEC, elem_type=elem_type, length=length)

    @staticmethod
    def mat(elem_type: 'Type', rows: int, cols: int,
            quant_scheme: Optional[str] = None,
            quant_scale: Optional[float] = None,
            quant_offset: Optional[int] = None):
        return Type(TypeKind.MAT, elem_type=elem_type, rows=rows, cols=cols,
                   quant_scheme=quant_scheme, quant_scale=quant_scale,
                   quant_offset=quant_offset)

    @staticmethod
    def ptr(addr_space: str, pointee_type: 'Type'):
        return Type(TypeKind.PTR, addr_space=addr_space, pointee_type=pointee_type)


# ============================================================================
# SSA Values
# ============================================================================

@dataclass
class Value:
    """SSA value (result of an operation)"""
    name: str
    type: Type

    def __str__(self):
        return f"%{self.name}:{self.type}"


# ============================================================================
# Operations
# ============================================================================

@dataclass
class Op:
    """IR operation"""
    kind: str  # Operation kind (ADD, MATMUL, LOAD, etc.)
    args: List[Union[Value, int, float, str]]  # Arguments
    result: Optional[Value] = None  # Result value (if produces value)
    has_side_effects: bool = False  # True for STORE, PRINT, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def __str__(self):
        if self.result:
            args_str = ", ".join(str(a) for a in self.args)
            return f"{self.result} = {self.kind}({args_str})"
        else:
            args_str = ", ".join(str(a) for a in self.args)
            return f"{self.kind}({args_str})"

    def to_dict(self):
        """Serialize to JSON-compatible dict"""
        return {
            "kind": self.kind,
            "args": [self._serialize_arg(a) for a in self.args],
            "result": str(self.result) if self.result else None,
            "has_side_effects": self.has_side_effects,
            "metadata": self.metadata
        }

    @staticmethod
    def _serialize_arg(arg):
        """Serialize argument for JSON"""
        if isinstance(arg, Value):
            return str(arg)
        else:
            return arg


# ============================================================================
# Control Flow
# ============================================================================

@dataclass
class Block:
    """Basic block (straight-line code with single entry/exit)"""
    name: str
    ops: List[Op] = field(default_factory=list)
    terminator: Optional[Op] = None  # BR, JMP, or RET

    def __str__(self):
        ops_str = "\n  ".join(str(op) for op in self.ops)
        term_str = f"\n  {self.terminator}" if self.terminator else ""
        return f"{self.name}:\n  {ops_str}{term_str}"

    def add_op(self, op: Op):
        """Add operation to block"""
        if self.terminator:
            raise ValueError(f"Cannot add op to block with terminator: {self.name}")
        self.ops.append(op)

    def set_terminator(self, op: Op):
        """Set block terminator"""
        if self.terminator:
            raise ValueError(f"Block {self.name} already has terminator")
        if op.kind not in ["BR", "JMP", "RET"]:
            raise ValueError(f"Invalid terminator kind: {op.kind}")
        self.terminator = op

    def to_dict(self):
        """Serialize to JSON-compatible dict"""
        return {
            "name": self.name,
            "ops": [op.to_dict() for op in self.ops],
            "terminator": self.terminator.to_dict() if self.terminator else None
        }


# ============================================================================
# Program
# ============================================================================

@dataclass
class Program:
    """Complete IR program"""
    blocks: List[Block] = field(default_factory=list)
    globals: Dict[str, Value] = field(default_factory=dict)
    entry_block: str = "entry"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        blocks_str = "\n\n".join(str(block) for block in self.blocks)
        globals_str = "\n".join(f"global {name} = {val}" for name, val in self.globals.items())
        return f"// Globals\n{globals_str}\n\n// Program\n{blocks_str}"

    def add_block(self, block: Block):
        """Add block to program"""
        if any(b.name == block.name for b in self.blocks):
            raise ValueError(f"Block {block.name} already exists")
        self.blocks.append(block)

    def get_block(self, name: str) -> Optional[Block]:
        """Get block by name"""
        for block in self.blocks:
            if block.name == name:
                return block
        return None

    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps({
            "blocks": [block.to_dict() for block in self.blocks],
            "globals": {name: str(val) for name, val in self.globals.items()},
            "entry_block": self.entry_block,
            "metadata": self.metadata
        }, indent=2)


# ============================================================================
# Builder Helpers
# ============================================================================

class IRBuilder:
    """Helper for building IR programs"""

    def __init__(self):
        self.program = Program()
        self.current_block: Optional[Block] = None
        self.value_counter = 0

    def create_block(self, name: str) -> Block:
        """Create and add a new block"""
        block = Block(name)
        self.program.add_block(block)
        return block

    def set_insert_point(self, block: Block):
        """Set current block for inserting ops"""
        self.current_block = block

    def fresh_value(self, type: Type, prefix: str = "v") -> Value:
        """Generate fresh SSA value"""
        name = f"{prefix}{self.value_counter}"
        self.value_counter += 1
        return Value(name, type)

    def emit_op(self, kind: str, args: List, result_type: Optional[Type] = None,
                has_side_effects: bool = False, **metadata) -> Optional[Value]:
        """Emit operation to current block"""
        if not self.current_block:
            raise ValueError("No current block set")

        result = self.fresh_value(result_type) if result_type else None
        op = Op(kind, args, result, has_side_effects, metadata)
        self.current_block.add_op(op)
        return result

    def emit_terminator(self, kind: str, args: List, **metadata):
        """Emit terminator to current block"""
        if not self.current_block:
            raise ValueError("No current block set")

        op = Op(kind, args, None, False, metadata)
        self.current_block.set_terminator(op)

    # Convenience methods for common operations
    def add(self, lhs: Value, rhs: Value) -> Value:
        return self.emit_op("ADD", [lhs, rhs], result_type=lhs.type)

    def mul(self, lhs: Value, rhs: Value) -> Value:
        return self.emit_op("MUL", [lhs, rhs], result_type=lhs.type)

    def matmul(self, lhs: Value, rhs: Value, result_type: Type) -> Value:
        return self.emit_op("MATMUL", [lhs, rhs], result_type=result_type)

    def relu(self, x: Value) -> Value:
        return self.emit_op("RELU", [x], result_type=x.type)

    def load(self, ptr: Value, elem_type: Type) -> Value:
        return self.emit_op("LOAD", [ptr], result_type=elem_type)

    def store(self, ptr: Value, value: Value):
        self.emit_op("STORE", [ptr, value], has_side_effects=True)

    def print_str(self, ptr: Value, length: Union[Value, int]):
        self.emit_op("PRINT_STR", [ptr, length], has_side_effects=True)

    def draw_glyph(self, glyph_id: Union[Value, int], x: Union[Value, int], y: Union[Value, int]):
        self.emit_op("DRAW_GLYPH", [glyph_id, x, y], has_side_effects=True)

    def br(self, cond: Value, then_block: str, else_block: str):
        self.emit_terminator("BR", [cond, then_block, else_block])

    def jmp(self, target: str):
        self.emit_terminator("JMP", [target])

    def ret(self, value: Optional[Value] = None):
        args = [value] if value else []
        self.emit_terminator("RET", args)
