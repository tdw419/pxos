"""
pxIR - Pixel Intermediate Representation
The stable abstraction between frontend and backend
"""

from dataclasses import dataclass
from typing import List, Union, Dict, Optional
from enum import Enum

class IRType(Enum):
    FLOAT4 = "vec4"
    INT2 = "ivec2"
    IMAGE2D = "image2D"

class IROperation(Enum):
    LOAD_IMAGE = "imageLoad"
    STORE_IMAGE = "imageStore"
    SUBTRACT = "subtract"
    CONSTRUCT_FLOAT4 = "construct_vec4"

@dataclass
class IROperand:
    name: str
    ir_type: IRType
    is_literal: bool = False

@dataclass
class IRInstruction:
    op: IROperation
    operands: List[IROperand]
    result: Optional[IROperand] = None
    metadata: Dict[str, any] = None

@dataclass
class IRParameter:
    name: str
    ir_type: IRType
    binding: int
    qualifiers: List[str]

@dataclass
class IRKernel:
    name: str
    parameters: List[IRParameter]
    instructions: List[IRInstruction]
    workgroup_size: tuple = (16, 16, 1)
