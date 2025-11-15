#!/usr/bin/env python3
"""
pxi_cpu.py - Minimal PXI pixel CPU interpreter.

Each pixel in a PXI image encodes one instruction as RGBA:

    R = opcode
    G,B,A = small operands / immediates

This first version supports a tiny ISA:

    0x00 NOP         : do nothing
    0x10 LOAD        : LOAD R[G] = B        (8 registers, 0-7)
    0x60 DRAW        : draw at (R0, R1) with color (G,B,A) into FRAME
    0xFF HALT        : stop execution

Program counter (PC) is a linear index over the PXI image:
    pc in [0, width*height)
    (x, y) = divmod(pc, width)
"""

from dataclasses import dataclass, field
from typing import List
from PIL import Image

# Opcodes
OP_NOP  = 0x00
OP_LOAD = 0x10
OP_DRAW = 0x60
OP_HALT = 0xFF

@dataclass
class PXICPU:
    pxi_image: Image.Image
    max_steps: int = 10_000

    width: int = field(init=False)
    height: int = field(init=False)
    pc: int = field(default=0, init=False)
    regs: List[int] = field(default_factory=lambda: [0] * 8, init=False)
    halted: bool = field(default=False, init=False)
    frame: Image.Image = field(init=False)

    def __post_init__(self):
        # Normalize to RGBA
        self.pxi_image = self.pxi_image.convert("RGBA")
        self.width, self.height = self.pxi_image.size
        # FRAME starts black, same size as PXI
        self.frame = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 255))

    def _pc_to_xy(self, pc: int):
        y, x = divmod(pc, self.width)
        return x, y

    def step(self):
        if self.halted:
            return

        if not (0 <= self.pc < self.width * self.height):
            # PC out of range: implicit HALT
            self.halted = True
            return

        x, y = self._pc_to_xy(self.pc)
        r, g, b, a = self.pxi_image.getpixel((x, y))
        opcode = r
        self.pc += 1  # default: next instruction

        # ---- Instruction semantics ----
        if opcode == OP_NOP:
            return

        elif opcode == OP_LOAD:
            reg_idx = g & 0x07  # 0-7
            imm = b  # 0-255
            self.regs[reg_idx] = imm

        elif opcode == OP_DRAW:
            x_draw = self.regs[0]
            y_draw = self.regs[1]
            if 0 <= x_draw < self.width and 0 <= y_draw < self.height:
                self.frame.putpixel((x_draw, y_draw), (g, b, a, 255))

        elif opcode == OP_HALT:
            self.halted = True

        else:
            # Unknown opcode → treat as NOP for now
            return

    def run(self):
        steps = 0
        while not self.halted and steps < self.max_steps:
            self.step()
            steps += 1

    @classmethod
    def run_file(cls, pxi_path: str, out_frame_path: str):
        img = Image.open(pxi_path)
        cpu = cls(img)
        cpu.run()
        cpu.frame.save(out_frame_path)
        return cpu
