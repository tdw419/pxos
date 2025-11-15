#!/usr/bin/env python3
"""
PXI_CPU — Pixel-based Virtual Machine
Every pixel is an instruction. RGBA = (opcode, arg1, arg2, arg3)

A computer where the program IS the image.
"""

from PIL import Image
from dataclasses import dataclass
from typing import Optional, Tuple
import requests

# Opcodes (R channel)
OP_NOP      = 0x00  # No operation
OP_HALT     = 0xFF  # Stop execution
OP_LOAD     = 0x10  # Load immediate: R[arg1] = arg2
OP_STORE    = 0x11  # Store: MEM[arg2] = R[arg1]
OP_ADD      = 0x20  # R[arg1] = R[arg2] + R[arg3]
OP_SUB      = 0x21  # R[arg1] = R[arg2] - R[arg3]
OP_JMP      = 0x30  # Jump to pixel address
OP_JNZ      = 0x31  # Jump if R[arg1] != 0
OP_CALL     = 0x32  # Call function: push PC, jump to address
OP_RET      = 0x33  # Return from function: pop PC
OP_PUSH     = 0x34  # Push R[arg1] to stack
OP_POP      = 0x35  # Pop stack to R[arg1]
OP_DRAW     = 0x40  # Draw pixel at (R[arg1], R[arg2]) with color R[arg3]
OP_PRINT    = 0x41  # Print character R[arg1] to screen buffer
OP_SYS_LLM  = 0xC8  # Call local LLM: R0=prompt_addr, R1=output_addr, R2=max_len, R3=model_id
OP_SYS_BLOB = 0xCC  # Load file blob: R0=file_id, R1=dest_addr, R2=max_len, R3=flags

@dataclass
class PXICPU:
    """Pixel Instruction CPU - runs programs stored as PNG images"""

    pxi_image: Image.Image
    regs: list = None
    pc: int = 0
    halted: bool = False
    frame: Image.Image = None  # Output frame (visual state)
    width: int = 0
    height: int = 0

    def __post_init__(self):
        if self.regs is None:
            self.regs = [0] * 16  # R0-R15
        self.width, self.height = self.pxi_image.size
        self.frame = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 255))
        self.screen_buffer = ""  # Text output buffer
        self.stack = []  # Call stack for CALL/RET

    def _pc_to_xy(self, pc: int) -> Tuple[int, int]:
        """Convert linear PC to (x, y) coordinates"""
        x = pc % self.width
        y = pc // self.width
        return (x, y)

    def _xy_to_pc(self, x: int, y: int) -> int:
        """Convert (x, y) to linear PC"""
        return y * self.width + x

    def fetch(self) -> Tuple[int, int, int, int]:
        """Fetch instruction at PC"""
        if self.pc >= self.width * self.height:
            return (OP_HALT, 0, 0, 0)

        x, y = self._pc_to_xy(self.pc)
        r, g, b, a = self.pxi_image.getpixel((x, y))
        return (r, g, b, a)

    def step(self) -> bool:
        """Execute one instruction. Returns False if halted."""
        if self.halted:
            return False

        opcode, arg1, arg2, arg3 = self.fetch()

        if opcode == OP_NOP:
            pass

        elif opcode == OP_HALT:
            self.halted = True
            return False

        elif opcode == OP_LOAD:
            # LOAD R[arg1], arg2
            if arg1 < 16:
                self.regs[arg1] = arg2

        elif opcode == OP_STORE:
            # STORE R[arg1], arg2 → write R[arg1] to pixel at address arg2
            if arg1 < 16 and arg2 < self.width * self.height:
                value = self.regs[arg1]
                x, y = self._pc_to_xy(arg2)
                self.frame.putpixel((x, y), (value, value, value, 255))

        elif opcode == OP_ADD:
            # ADD R[arg1] = R[arg2] + R[arg3]
            if arg1 < 16 and arg2 < 16 and arg3 < 16:
                self.regs[arg1] = (self.regs[arg2] + self.regs[arg3]) & 0xFF

        elif opcode == OP_SUB:
            # SUB R[arg1] = R[arg2] - R[arg3]
            if arg1 < 16 and arg2 < 16 and arg3 < 16:
                self.regs[arg1] = (self.regs[arg2] - self.regs[arg3]) & 0xFF

        elif opcode == OP_JMP:
            # JMP to address arg1 (as offset from current)
            self.pc = arg2 * 256 + arg3
            return True

        elif opcode == OP_JNZ:
            # JNZ R[arg1], address
            if arg1 < 16 and self.regs[arg1] != 0:
                self.pc = arg2 * 256 + arg3
                return True

        elif opcode == OP_CALL:
            # CALL address: push return address, jump
            return_addr = self.pc + 1
            self.stack.append(return_addr)
            self.pc = arg2 * 256 + arg3
            return True

        elif opcode == OP_RET:
            # RET: pop return address, jump back
            if self.stack:
                self.pc = self.stack.pop()
                return True
            else:
                # Stack underflow - halt
                self.halted = True
                return False

        elif opcode == OP_PUSH:
            # PUSH R[arg1]
            if arg1 < 16:
                self.stack.append(self.regs[arg1])

        elif opcode == OP_POP:
            # POP to R[arg1]
            if arg1 < 16 and self.stack:
                self.regs[arg1] = self.stack.pop()

        elif opcode == OP_DRAW:
            # DRAW pixel at (R[arg1], R[arg2]) with color R[arg3]
            if arg1 < 16 and arg2 < 16 and arg3 < 16:
                x, y = self.regs[arg1], self.regs[arg2]
                color = self.regs[arg3]
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.frame.putpixel((x, y), (color, color, color, 255))

        elif opcode == OP_PRINT:
            # PRINT character R[arg1]
            if arg1 < 16:
                char = chr(self.regs[arg1]) if self.regs[arg1] < 256 else '?'
                self.screen_buffer += char
                print(char, end='', flush=True)

        elif opcode == OP_SYS_LLM:
            # SYS_LLM: R0=prompt_addr, R1=output_addr, R2=max_len, R3=model_id (optional)
            self._syscall_llm()

        elif opcode == OP_SYS_BLOB:
            # SYS_BLOB: R0=file_id, R1=dest_addr, R2=max_len, R3=flags
            self._syscall_blob()

        self.pc += 1
        return True

    def _syscall_llm(self):
        """Call local LLM via HTTP with optional model selection"""
        prompt_addr = self.regs[0]
        output_addr = self.regs[1]
        max_len = self.regs[2] if self.regs[2] > 0 else 1024
        model_id = self.regs[3] if self.regs[3] > 0 else None  # Optional model selection

        # Read prompt string from image (using G channel as ASCII)
        prompt = self.read_string(prompt_addr)

        if prompt:
            print(f"\n[SYS_LLM] Prompt: {prompt[:50]}...")
            if model_id:
                print(f"[SYS_LLM] Model ID: 0x{model_id:08X}")
                response = query_local_llm_via_digest(prompt, model_id, max_len)
            else:
                response = query_local_llm(prompt)
            print(f"[SYS_LLM] Response: {response[:50]}...")
            written = self.write_string(output_addr, response, max_len)
            self.regs[0] = written
        else:
            self.regs[0] = 0

    def _syscall_blob(self):
        """
        Load file blob into PXI memory

        R0 = file_id (32-bit ID from sub-boot pixel RGBA)
        R1 = dest_addr (where to write in PXI memory)
        R2 = max_len (max bytes to load)
        R3 = flags (bit 0: decompress, bit 1: text mode)

        Returns in R0: number of bytes loaded (or 0 on error)
        """
        import json
        import zlib
        from pathlib import Path

        file_id = self.regs[0]
        dest_addr = self.regs[1]
        max_len = self.regs[2] if self.regs[2] > 0 else 65536
        flags = self.regs[3]

        decompress = (flags & 0x01) != 0
        text_mode = (flags & 0x02) != 0

        # Try to load registry
        registry_path = Path("file_boot_registry.json")

        if not registry_path.exists():
            print(f"[SYS_BLOB] Error: file_boot_registry.json not found")
            self.regs[0] = 0
            return

        try:
            with open(registry_path, 'r') as f:
                registry = json.load(f)

            entry = registry.get(str(file_id))
            if not entry:
                print(f"[SYS_BLOB] Error: File ID 0x{file_id:08X} not found")
                self.regs[0] = 0
                return

            print(f"[SYS_BLOB] Loading: {entry['name']} (0x{file_id:08X})")

            # Load from packed file or individual blob
            if 'blob_offset' in entry:
                # Packed mode
                project_files = Path("project_files.bin")
                if not project_files.exists():
                    print(f"[SYS_BLOB] Error: project_files.bin not found")
                    self.regs[0] = 0
                    return

                with open(project_files, 'rb') as f:
                    f.seek(entry['blob_offset'])
                    data = f.read(entry['blob_len'])
            else:
                # Individual blob mode
                blob_path = Path(entry['blob_file'])
                data = blob_path.read_bytes()

            # Decompress if needed
            if decompress or entry.get('compressed_size'):
                data = zlib.decompress(data)

            # Truncate to max_len
            data = data[:max_len]

            # Write to PXI memory
            bytes_written = 0
            for i, byte_val in enumerate(data):
                pc = dest_addr + i
                if pc >= self.width * self.height:
                    break

                x, y = self._pc_to_xy(pc)

                if text_mode:
                    # Text mode: write to G channel (ASCII)
                    self.pxi_image.putpixel((x, y), (byte_val, byte_val, byte_val, 255))
                else:
                    # Binary mode: write to R channel
                    self.pxi_image.putpixel((x, y), (byte_val, 0, 0, 255))

                bytes_written += 1

            print(f"[SYS_BLOB] Loaded {bytes_written} bytes to addr {dest_addr}")
            self.regs[0] = bytes_written

        except Exception as e:
            print(f"[SYS_BLOB] Error: {e}")
            self.regs[0] = 0

    def read_string(self, addr: int, max_len: int = 1024) -> str:
        """Read null-terminated string from image (G channel = ASCII)"""
        s = ""
        for i in range(max_len):
            pc = addr + i
            if pc >= self.width * self.height:
                break
            x, y = self._pc_to_xy(pc)
            r, g, b, a = self.pxi_image.getpixel((x, y))
            if g == 0:  # null terminator
                break
            s += chr(g)
        return s

    def write_string(self, addr: int, text: str, max_len: int = 1024) -> int:
        """Write string to image (G channel = ASCII)"""
        written = 0
        for i, c in enumerate(text):
            if written >= max_len or addr + written >= self.width * self.height:
                break
            pc = addr + written
            x, y = self._pc_to_xy(pc)
            char_code = ord(c) if ord(c) < 256 else 63  # '?' fallback
            self.frame.putpixel((x, y), (char_code, char_code, char_code, 255))
            written += 1

        # Null terminate
        if written < max_len and addr + written < self.width * self.height:
            pc = addr + written
            x, y = self._pc_to_xy(pc)
            self.frame.putpixel((x, y), (0, 0, 0, 255))

        return written

    def run(self, max_steps: int = 100000):
        """Run until halt or max steps"""
        steps = 0
        while not self.halted and steps < max_steps:
            if not self.step():
                break
            steps += 1

        if steps >= max_steps:
            print(f"\n[WARNING] Hit max steps ({max_steps})")

        print(f"\n[HALT] Executed {steps} instructions")
        return steps


def query_local_llm(prompt: str, port: int = 1234, model: str = "local-model") -> str:
    """
    Query local LLM via LM Studio or Ollama
    LM Studio: http://localhost:1234/v1/chat/completions
    Ollama: http://localhost:11434/v1/chat/completions
    """
    try:
        r = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 500
            },
            timeout=30
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[LLM error: {e}]"


def query_local_llm_via_digest(prompt: str, model_id: int, max_tokens: int = 512) -> str:
    """
    Query local LLM via PXDigest model ID

    Looks up model configuration in llm_pixel_registry.json and calls appropriate backend.
    """
    import json
    from pathlib import Path

    registry_path = Path("llm_pixel_registry.json")

    if not registry_path.exists():
        return "[Error: LLM pixel registry not found]"

    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)

        model_config = registry.get(str(model_id))
        if not model_config:
            return f"[Error: Model ID 0x{model_id:08X} not found in registry]"

        endpoint = model_config.get("endpoint")
        model_name = model_config.get("model_name", "local-model")
        temperature = model_config.get("temperature", 0.7)
        system_prompt = model_config.get("system_prompt")

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Call the model
        r = requests.post(
            endpoint,
            json={
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            timeout=30
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    except Exception as e:
        return f"[LLM error: {e}]"


if __name__ == "__main__":
    # Test: Create a simple program that prints "HELLO"
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 255))

    def set_pixel(pc, r, g, b, a=0):
        y, x = divmod(pc, 64)
        img.putpixel((x, y), (r, g, b, a))

    # Program: Print "HELLO"
    # 0: LOAD R0, 'H'
    set_pixel(0, OP_LOAD, 0, ord('H'), 0)
    # 1: PRINT R0
    set_pixel(1, OP_PRINT, 0, 0, 0)
    # 2: LOAD R0, 'E'
    set_pixel(2, OP_LOAD, 0, ord('E'), 0)
    # 3: PRINT R0
    set_pixel(3, OP_PRINT, 0, 0, 0)
    # 4: LOAD R0, 'L'
    set_pixel(4, OP_LOAD, 0, ord('L'), 0)
    # 5: PRINT R0
    set_pixel(5, OP_PRINT, 0, 0, 0)
    # 6: PRINT R0 (second L)
    set_pixel(6, OP_PRINT, 0, 0, 0)
    # 7: LOAD R0, 'O'
    set_pixel(7, OP_LOAD, 0, ord('O'), 0)
    # 8: PRINT R0
    set_pixel(8, OP_PRINT, 0, 0, 0)
    # 9: HALT
    set_pixel(9, OP_HALT, 0, 0, 0)

    img.save("/home/user/pxos/test_hello.png")
    print("Booting PXI_CPU...")

    cpu = PXICPU(img)
    cpu.run()

    print("\n✓ PXI_CPU test complete")
