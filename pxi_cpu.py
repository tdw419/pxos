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
import requests
import zlib
import json
from pathlib import Path
import random

# Opcodes
OP_NOP  = 0x00
OP_LOAD = 0x10
OP_ADD = 0x30
OP_SUB = 0x31
OP_JMP = 0x40
OP_DRAW = 0x60
OP_RAND = 0x90
OP_HALT = 0xFF
OP_SYS_LLM = 0xC8
OP_SYS_DECOMPRESS = 0xCD

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

        # God Pixel auto-loading
        if self.pxi_image.size == (1, 1):
            self.pxi_image = self._load_god_pixel()

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

        elif opcode == OP_ADD:
            reg_dst = g & 0x07
            reg_src = b & 0x07
            self.regs[reg_dst] += self.regs[reg_src]

        elif opcode == OP_SUB:
            reg_dst = g & 0x07
            reg_src = b & 0x07
            self.regs[reg_dst] -= self.regs[reg_src]

        elif opcode == OP_JMP:
            self.pc = b * self.width + g

        elif opcode == OP_DRAW:
            x_draw = self.regs[0]
            y_draw = self.regs[1]
            if 0 <= x_draw < self.width and 0 <= y_draw < self.height:
                self.frame.putpixel((x_draw, y_draw), (g, b, a, 255))

        elif opcode == OP_RAND:
            reg_dst = g & 0x07
            max_val = b
            self.regs[reg_dst] = random.randint(0, max_val)

        elif opcode == OP_HALT:
            self.halted = True

        elif opcode == OP_SYS_LLM:
            prompt_addr = self.regs[0]
            output_addr = self.regs[1]
            max_len = self.regs[2] if self.regs[2] > 0 else 1024

            prompt = self.read_string(prompt_addr)
            if prompt:
                response = query_local_llm(prompt)
                written = self.write_string(output_addr, response, max_len)
                self.regs[0] = written
            else:
                self.regs[0] = 0

        elif opcode == OP_SYS_DECOMPRESS:
            src_addr = self.regs[0]
            dst_addr = self.regs[1]
            compressed_size = self.regs[2]

            if compressed_size > 0:
                compressed_data = self.read_bytes(src_addr, compressed_size)
                try:
                    decompressed_data = zlib.decompress(compressed_data)
                    self.write_bytes(dst_addr, decompressed_data)
                    self.regs[0] = len(decompressed_data) # Return decompressed size
                except zlib.error:
                    self.regs[0] = 0 # Indicate error
            else:
                self.regs[0] = 0
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

    def _load_god_pixel(self):
        """
        Loads and decompresses the universe contained within a God Pixel.
        """
        r, g, b, a = self.pxi_image.getpixel((0, 0))

        registry_path = Path("god_pixel_registry.json")
        if not registry_path.exists():
            raise FileNotFoundError("god_pixel_registry.json not found.")

        with open(registry_path, 'r') as f:
            registry = json.load(f)

        world_meta = None
        for key, meta in registry.items():
            color_parts = [int(c) for c in key.split(',')]
            while len(color_parts) < 4:
                color_parts.append(255) # Default alpha to 255
            reg_r, reg_g, reg_b, reg_a = color_parts
            if r == reg_r and g == reg_g and b == reg_b:
                world_meta = meta
                color_key = key
                break

        if world_meta is None:
            raise ValueError(f"God Pixel color {r},{g},{b} not found in registry.")

        blob_path = Path(world_meta['blob'])

        if not blob_path.exists():
            raise FileNotFoundError(f"Compressed blob not found: {blob_path}")

        with open(blob_path, 'rb') as f:
            compressed_data = f.read()

        decompressed_data = zlib.decompress(compressed_data)

        # The decompressed data is the raw bytes of the new PXI image
        width, height = world_meta['size']
        new_pxi_image = Image.frombytes("RGBA", (width, height), decompressed_data)

        print(f"God Pixel {color_key} loaded: '{world_meta['name']}' ({width}x{height})")
        return new_pxi_image

    def read_bytes(self, addr: int, size: int) -> bytes:
        """Reads a sequence of bytes from the PXI image."""
        byte_data = bytearray()
        for i in range(size):
            pc = addr + i
            if pc >= self.width * self.height:
                break
            x, y = self._pc_to_xy(pc)
            # For simplicity, we'll just read the R channel of each pixel.
            # A more robust implementation would pack data more densely.
            r, _, _, _ = self.pxi_image.getpixel((x, y))
            byte_data.append(r)
        return bytes(byte_data)

    def write_bytes(self, addr: int, data: bytes):
        """Writes a sequence of bytes to the PXI image."""
        for i, byte in enumerate(data):
            pc = addr + i
            if pc >= self.width * self.height:
                break
            x, y = self._pc_to_xy(pc)
            # We'll write the byte to all color channels for visibility.
            self.pxi_image.putpixel((x, y), (byte, byte, byte, 255))

    def read_string(self, addr: int, max_len: int = 1024) -> str:
        """Read null-terminated glyph string from DATA region"""
        s = ""
        for i in range(max_len):
            pc = addr + i
            if pc >= self.width * self.height:
                break
            x, y = self._pc_to_xy(pc)
            r, g, b, a = self.pxi_image.getpixel((x, y))
            if r == 0 and g == 0 and b == 0:  # null terminator
                break
            s += chr(g)  # using G channel as ASCII char
        return s

    def write_string(self, addr: int, text: str, max_len: int = 1024) -> int:
        """Write string as glyphs starting at addr"""
        written = 0
        for i, c in enumerate(text):
            if written >= max_len or i + written >= self.width * self.height:
                break
            pc = addr + written
            x, y = self._pc_to_xy(pc)
            char_code = ord(c) if ord(c) < 256 else 63  # ? fallback
            self.pxi_image.putpixel((x, y), (char_code, char_code, char_code, 255))
            written += 1
        # null terminate
        if written < max_len:
            pc = addr + written
            if pc < self.width * self.height:
                x, y = self._pc_to_xy(pc)
                self.pxi_image.putpixel((x, y), (0, 0, 0, 255))
        return written

def query_local_llm(prompt: str) -> str:
    # In this sandboxed environment, we'll simulate the LLM call
    return "I am a local LLM running on your machine. I have no internet access."
