"""
Shader Virtual Machine - Core instruction set and bytecode compiler

This is the heart of pxOS: a simple VM that runs ON GPU shaders,
allowing multiple language frontends to compile to the same bytecode.

Architecture:
- Simple stack-based VM
- Graphics-optimized instruction set
- Runs entirely in WGSL compute shaders
- Supports hot-reloading and interactive development
"""

import struct
from enum import IntEnum
from typing import List, Tuple, Any
from dataclasses import dataclass


class Opcode(IntEnum):
    """Shader VM instruction set - optimized for graphics"""

    # Stack operations
    PUSH = 0        # Push constant to stack
    POP = 1         # Pop from stack
    DUP = 2         # Duplicate top of stack
    SWAP = 3        # Swap top two stack values

    # Arithmetic operations
    ADD = 10        # Add top two values
    SUB = 11        # Subtract
    MUL = 12        # Multiply
    DIV = 13        # Divide
    MOD = 14        # Modulo
    NEG = 15        # Negate

    # Math functions
    SIN = 20        # Sine
    COS = 21        # Cosine
    TAN = 22        # Tangent
    ASIN = 23       # Arcsine
    ACOS = 24       # Arccosine
    ATAN = 25       # Arctangent
    ATAN2 = 26      # Two-argument arctangent
    EXP = 27        # e^x
    LOG = 28        # Natural log
    POW = 29        # Power
    SQRT = 30       # Square root
    ABS = 31        # Absolute value
    FLOOR = 32      # Floor
    CEIL = 33       # Ceiling
    FRACT = 34      # Fractional part
    MIN = 35        # Minimum
    MAX = 36        # Maximum
    CLAMP = 37      # Clamp to range

    # Comparison operations
    LT = 40         # Less than
    GT = 41         # Greater than
    LE = 42         # Less or equal
    GE = 43         # Greater or equal
    EQ = 44         # Equal
    NE = 45         # Not equal

    # Logical operations
    AND = 50        # Logical AND
    OR = 51         # Logical OR
    NOT = 52        # Logical NOT

    # Control flow
    JMP = 60        # Unconditional jump
    JMP_IF = 61     # Jump if true
    JMP_IF_NOT = 62 # Jump if false
    CALL = 63       # Function call
    RET = 64        # Return from function

    # Graphics primitives
    UV = 70         # Push UV coordinates (x, y)
    TIME = 71       # Push time uniform
    RESOLUTION = 72 # Push resolution (width, height)
    MOUSE = 73      # Push mouse position

    # Color operations
    RGB = 80        # Create RGB color (r, g, b) -> rgb
    RGBA = 81       # Create RGBA color (r, g, b, a) -> rgba
    HSV = 82        # Create HSV color (h, s, v) -> rgb
    COLOR = 83      # Output final color (terminates shader)

    # Vector operations
    VEC2 = 90       # Create vec2 from two values
    VEC3 = 91       # Create vec3 from three values
    VEC4 = 92       # Create vec4 from four values
    DOT = 93        # Dot product
    CROSS = 94      # Cross product
    LENGTH = 95     # Vector length
    NORMALIZE = 96  # Normalize vector
    DISTANCE = 97   # Distance between two points

    # Shape primitives
    CIRCLE = 100    # Circle SDF (x, y, radius)
    RECT = 101      # Rectangle SDF (x, y, w, h)
    LINE = 102      # Line SDF

    # Memory operations
    LOAD = 110      # Load from memory
    STORE = 111     # Store to memory

    # Debug
    HALT = 255      # Halt execution


@dataclass
class Instruction:
    """Single VM instruction"""
    opcode: Opcode
    args: List[float] = None

    def __post_init__(self):
        if self.args is None:
            self.args = []


class ShaderVM:
    """
    Shader Virtual Machine - compiles high-level operations to bytecode
    that runs on GPU shaders
    """

    def __init__(self):
        self.instructions: List[Instruction] = []
        self.labels: dict[str, int] = {}
        self.unresolved_jumps: List[Tuple[int, str]] = []

    def emit(self, opcode: Opcode, *args: float) -> int:
        """Emit an instruction and return its index"""
        instruction = Instruction(opcode, list(args))
        idx = len(self.instructions)
        self.instructions.append(instruction)
        return idx

    def label(self, name: str):
        """Define a label at current position"""
        self.labels[name] = len(self.instructions)

    def jump(self, label: str):
        """Emit a jump to a label"""
        idx = self.emit(Opcode.JMP, 0)  # Placeholder
        self.unresolved_jumps.append((idx, label))

    def jump_if(self, label: str):
        """Emit a conditional jump"""
        idx = self.emit(Opcode.JMP_IF, 0)  # Placeholder
        self.unresolved_jumps.append((idx, label))

    def resolve_jumps(self):
        """Resolve all jump labels"""
        for idx, label in self.unresolved_jumps:
            if label not in self.labels:
                raise ValueError(f"Undefined label: {label}")
            target = self.labels[label]
            self.instructions[idx].args[0] = float(target)
        self.unresolved_jumps.clear()

    def compile_to_bytecode(self) -> bytes:
        """
        Compile instructions to binary bytecode for GPU

        Format: each instruction is:
        - 4 bytes: opcode (uint32)
        - N * 4 bytes: arguments (float32, bitcast to uint32)
        """
        self.resolve_jumps()

        bytecode = []
        for instr in self.instructions:
            # Emit opcode
            bytecode.append(struct.pack('I', int(instr.opcode)))

            # Emit arguments
            for arg in instr.args:
                # Convert float to uint32 bits
                bytecode.append(struct.pack('I', struct.unpack('I', struct.pack('f', arg))[0]))

        return b''.join(bytecode)

    def compile_to_uint32_array(self) -> List[int]:
        """
        Compile to array of uint32 values (for WebGPU buffer)
        """
        self.resolve_jumps()

        result = []
        for instr in self.instructions:
            # Add opcode
            result.append(int(instr.opcode))

            # Add arguments as bitcast float->uint32
            for arg in instr.args:
                bits = struct.unpack('I', struct.pack('f', arg))[0]
                result.append(bits)

        return result

    def disassemble(self) -> str:
        """Disassemble bytecode for debugging"""
        lines = []
        for i, instr in enumerate(self.instructions):
            # Check if this is a label target
            label_str = ""
            for label, target in self.labels.items():
                if target == i:
                    label_str = f"{label}:"

            if label_str:
                lines.append(label_str)

            args_str = ", ".join(f"{arg:.2f}" for arg in instr.args)
            lines.append(f"  {i:4d}: {instr.opcode.name:12s} {args_str}")

        return "\n".join(lines)


class EffectCompiler:
    """
    Compiles high-level effect descriptions to shader VM bytecode
    """

    def compile_plasma(self) -> ShaderVM:
        """Classic plasma effect"""
        vm = ShaderVM()

        # Get UV coordinates
        vm.emit(Opcode.UV)           # Stack: [u, v]

        # Calculate plasma: sin(u*10 + time) + cos(v*10 + time)
        vm.emit(Opcode.DUP)          # [u, v, v]
        vm.emit(Opcode.SWAP)         # [u, v, v] -> [v, u, v]

        # First component: sin(u*10 + time)
        vm.emit(Opcode.PUSH, 10.0)   # [v, u, v, 10]
        vm.emit(Opcode.MUL)          # [v, u, v*10]
        vm.emit(Opcode.TIME)         # [v, u, v*10, t]
        vm.emit(Opcode.ADD)          # [v, u, v*10+t]
        vm.emit(Opcode.SIN)          # [v, u, sin(v*10+t)]

        # Second component: cos(u*10 + time)
        vm.emit(Opcode.SWAP)         # [v, sin(...), u]
        vm.emit(Opcode.PUSH, 10.0)   # [v, sin(...), u, 10]
        vm.emit(Opcode.MUL)          # [v, sin(...), u*10]
        vm.emit(Opcode.TIME)         # [v, sin(...), u*10, t]
        vm.emit(Opcode.ADD)          # [v, sin(...), u*10+t]
        vm.emit(Opcode.COS)          # [v, sin(...), cos(u*10+t)]

        # Combine
        vm.emit(Opcode.ADD)          # [v, result]
        vm.emit(Opcode.PUSH, 0.5)    # [v, result, 0.5]
        vm.emit(Opcode.MUL)          # [v, result*0.5]
        vm.emit(Opcode.PUSH, 0.5)    # [v, result*0.5, 0.5]
        vm.emit(Opcode.ADD)          # [v, color] - normalized to 0-1

        # Output as grayscale
        vm.emit(Opcode.DUP)          # [v, color, color]
        vm.emit(Opcode.DUP)          # [v, color, color, color]
        vm.emit(Opcode.PUSH, 1.0)    # [v, color, color, color, 1.0]
        vm.emit(Opcode.COLOR)        # Output RGBA

        return vm

    def compile_gradient(self) -> ShaderVM:
        """Simple UV gradient"""
        vm = ShaderVM()

        vm.emit(Opcode.UV)           # [u, v]
        vm.emit(Opcode.PUSH, 0.5)    # [u, v, 0.5]
        vm.emit(Opcode.PUSH, 1.0)    # [u, v, 0.5, 1.0]
        vm.emit(Opcode.COLOR)        # Output (u, v, 0.5, 1.0)

        return vm

    def compile_pulsing_circle(self) -> ShaderVM:
        """Pulsing circle in center"""
        vm = ShaderVM()

        # Get UV and center it
        vm.emit(Opcode.UV)           # [u, v]
        vm.emit(Opcode.PUSH, 0.5)    # [u, v, 0.5]
        vm.emit(Opcode.SUB)          # [u, v-0.5]
        vm.emit(Opcode.SWAP)         # [v-0.5, u]
        vm.emit(Opcode.PUSH, 0.5)    # [v-0.5, u, 0.5]
        vm.emit(Opcode.SUB)          # [v-0.5, u-0.5]

        # Calculate distance from center
        vm.emit(Opcode.DUP)          # [v-0.5, u-0.5, u-0.5]
        vm.emit(Opcode.MUL)          # [v-0.5, (u-0.5)^2]
        vm.emit(Opcode.SWAP)         # [(u-0.5)^2, v-0.5]
        vm.emit(Opcode.DUP)          # [(u-0.5)^2, v-0.5, v-0.5]
        vm.emit(Opcode.MUL)          # [(u-0.5)^2, (v-0.5)^2]
        vm.emit(Opcode.ADD)          # [dist^2]
        vm.emit(Opcode.SQRT)         # [dist]

        # Pulsing radius: 0.2 + 0.1*sin(time*2)
        vm.emit(Opcode.TIME)         # [dist, t]
        vm.emit(Opcode.PUSH, 2.0)    # [dist, t, 2]
        vm.emit(Opcode.MUL)          # [dist, t*2]
        vm.emit(Opcode.SIN)          # [dist, sin(t*2)]
        vm.emit(Opcode.PUSH, 0.1)    # [dist, sin(t*2), 0.1]
        vm.emit(Opcode.MUL)          # [dist, 0.1*sin(t*2)]
        vm.emit(Opcode.PUSH, 0.2)    # [dist, 0.1*sin(t*2), 0.2]
        vm.emit(Opcode.ADD)          # [dist, radius]

        # Check if inside circle
        vm.emit(Opcode.LT)           # [dist < radius]

        # Convert boolean to color (1.0 if inside, 0.0 if outside)
        vm.emit(Opcode.DUP)          # [inside, inside]
        vm.emit(Opcode.DUP)          # [inside, inside, inside]
        vm.emit(Opcode.PUSH, 1.0)    # [inside, inside, inside, 1.0]
        vm.emit(Opcode.COLOR)

        return vm

    def compile_rainbow_spiral(self) -> ShaderVM:
        """Animated rainbow spiral"""
        vm = ShaderVM()

        # Get centered UV
        vm.emit(Opcode.UV)           # [u, v]
        vm.emit(Opcode.PUSH, 0.5)
        vm.emit(Opcode.SUB)          # [u, v-0.5]
        vm.emit(Opcode.SWAP)
        vm.emit(Opcode.PUSH, 0.5)
        vm.emit(Opcode.SUB)          # [v-0.5, u-0.5]

        # Calculate angle: atan2(y, x)
        vm.emit(Opcode.ATAN2)        # [angle]

        # Calculate distance
        vm.emit(Opcode.UV)
        vm.emit(Opcode.PUSH, 0.5)
        vm.emit(Opcode.SUB)
        vm.emit(Opcode.SWAP)
        vm.emit(Opcode.PUSH, 0.5)
        vm.emit(Opcode.SUB)
        vm.emit(Opcode.DUP)
        vm.emit(Opcode.MUL)
        vm.emit(Opcode.SWAP)
        vm.emit(Opcode.DUP)
        vm.emit(Opcode.MUL)
        vm.emit(Opcode.ADD)
        vm.emit(Opcode.SQRT)         # [angle, dist]

        # Spiral: angle + dist*10 + time
        vm.emit(Opcode.PUSH, 10.0)
        vm.emit(Opcode.MUL)          # [angle, dist*10]
        vm.emit(Opcode.ADD)          # [angle+dist*10]
        vm.emit(Opcode.TIME)
        vm.emit(Opcode.ADD)          # [angle+dist*10+time]

        # Convert to hue (0-1 range)
        vm.emit(Opcode.PUSH, 6.28318)  # 2*pi
        vm.emit(Opcode.DIV)
        vm.emit(Opcode.FRACT)        # [hue]

        # HSV to RGB with S=1, V=1
        vm.emit(Opcode.PUSH, 1.0)    # [hue, 1.0]
        vm.emit(Opcode.PUSH, 1.0)    # [hue, 1.0, 1.0]
        vm.emit(Opcode.HSV)          # [r, g, b]
        vm.emit(Opcode.PUSH, 1.0)    # [r, g, b, 1.0]
        vm.emit(Opcode.COLOR)

        return vm


def test_compiler():
    """Test the shader VM compiler"""
    compiler = EffectCompiler()

    print("🎨 Shader VM Bytecode Compiler")
    print("=" * 60)

    effects = {
        "Plasma": compiler.compile_plasma(),
        "Gradient": compiler.compile_gradient(),
        "Pulsing Circle": compiler.compile_pulsing_circle(),
        "Rainbow Spiral": compiler.compile_rainbow_spiral(),
    }

    for name, vm in effects.items():
        print(f"\n{name}:")
        print(f"  Instructions: {len(vm.instructions)}")

        bytecode = vm.compile_to_uint32_array()
        print(f"  Bytecode size: {len(bytecode)} uint32s ({len(bytecode)*4} bytes)")

        print(f"\n  Disassembly:")
        for line in vm.disassemble().split('\n')[:10]:  # Show first 10 lines
            print(f"    {line}")
        if len(vm.instructions) > 10:
            print(f"    ... ({len(vm.instructions) - 10} more instructions)")


if __name__ == "__main__":
    test_compiler()
